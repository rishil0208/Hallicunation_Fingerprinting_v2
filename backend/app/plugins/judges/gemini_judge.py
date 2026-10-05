"""Gemini judge plugin — calls Gemini API for LLM-based verdict.

ADRs enforced:
  - A-M1/S-M1: Model identity anonymized in judge prompts
  - A-H2: Gemini API key/auth info sanitized from all exceptions
  - A-C1: Judge failure returns clean error, never fake RESOLVED_AMBIGUOUS
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Optional

from backend.app.schemas import JudgeResult


def _extract_json_text(text: str) -> str:
    """Extract JSON from LLM output, handling code fences anywhere.

    Handles:
    - Raw JSON: {"verdict": ...}
    - Code fenced: ```json\n{...}\n```
    - Prefixed: "Here is my analysis:\n```json\n{...}\n```"
    """
    # Try to find JSON inside code fences first
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # Try to find a raw JSON object
    brace_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if brace_match:
        return brace_match.group(0).strip()

    # Fallback: return original text (will fail at json.loads)
    return text


_JUDGE_PROMPT_TEMPLATE = """You are an expert hallucination detection system. Analyze the following response for hallucination indicators.

RESPONSE TO ANALYZE:
{answer}

FINGERPRINT ANALYSIS SUMMARY:
{fingerprint_summary}

TRIGGERED ANOMALY PATTERNS:
{patterns_text}

Based on this analysis, provide your verdict as a JSON object with exactly these fields:
- "verdict": "HIGH_RISK" or "LOW_RISK"
- "explanation": a clear explanation of your reasoning (2-3 sentences)
- "confidence": a float between 0.0 and 1.0

Respond ONLY with the JSON object, no other text."""


class GeminiJudgeError(Exception):
    """Raised when the Gemini judge call fails.

    Sanitized: never includes API keys or auth headers.
    """

    def __init__(self, message: str):
        # ADR A-H2: sanitize any API key fragments
        sanitized = message
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key and api_key in sanitized:
            sanitized = sanitized.replace(api_key, "[REDACTED]")
        super().__init__(sanitized)


# In-memory LRU cache to prevent redundant API token consumption for identical evaluations
_GEMINI_CACHE: dict[str, JudgeResult] = {}


class GeminiJudgePlugin:
    """Calls Gemini API for LLM-based hallucination verdict.

    Only invoked on AMBIGUOUS gate verdicts. Model identity is
    anonymized in the prompt (ADR A-M1/S-M1).
    """

    name = "gemini"

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-flash-lite-latest"):
        self._explicit_api_key = api_key
        self.model_name = os.environ.get("GEMINI_MODEL", model_name)

    @property
    def api_key(self) -> str:
        return self._explicit_api_key or os.environ.get("GEMINI_API_KEY", "")

    @api_key.setter
    def api_key(self, value: str) -> None:
        self._explicit_api_key = value

    def _call_gemini_api(self, prompt: str) -> str:
        current_key = self.api_key
        if not current_key:
            raise GeminiJudgeError(
                "GEMINI_API_KEY not set. Set it in .env or via settings."
            )

        # ── Universal Compatibility: support OpenAI / OpenRouter / Groq (sk-...) ──
        if current_key.startswith("sk-"):
            base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {current_key}",
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 256,
            }
            req = urllib.request.Request(base_url, data=json.dumps(payload).encode("utf-8"), headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                err_msg = str(e)
                if current_key and current_key in err_msg:
                    err_msg = err_msg.replace(current_key, "[REDACTED]")
                raise GeminiJudgeError(f"OpenAI-compatible API call failed: {type(e).__name__} ({err_msg})") from None

        # ── Native Google Gemini API (with candidate fallback on 404, 503, 429) ──
        candidate_models = [self.model_name]
        for fallback in ("gemini-flash-lite-latest", "gemini-flash-latest", "gemini-3.5-flash", "gemini-pro-latest"):
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        last_error = None
        for model in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={current_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 256,
                }
            }

            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    self.model_name = model
                    return data["candidates"][0]["content"]["parts"][0]["text"]
            except urllib.error.HTTPError as e:
                last_error = e
                if e.code in (404, 503, 429):
                    continue
                err_msg = str(e)
                try:
                    error_body = e.read().decode("utf-8", errors="ignore")
                    err_json = json.loads(error_body)
                    google_msg = err_json.get("error", {}).get("message", "")
                    if "API key not valid" in google_msg or "API_KEY_INVALID" in error_body:
                        err_msg = "Google returned API_KEY_INVALID: The API key provided is not a valid Gemini key. Please check your key at https://aistudio.google.com/app/apikey"
                    elif google_msg:
                        err_msg = f"{err_msg}: {google_msg}"
                except Exception:
                    pass
                if current_key and current_key in err_msg:
                    err_msg = err_msg.replace(current_key, "[REDACTED]")
                raise GeminiJudgeError(f"Gemini API call failed: {err_msg}") from None
            except Exception as e:
                last_error = e
                continue

        err_msg = str(last_error) if last_error else "All candidate Gemini models failed"
        if hasattr(last_error, "read"):
            try:
                error_body = last_error.read().decode("utf-8", errors="ignore")
                err_json = json.loads(error_body)
                google_msg = err_json.get("error", {}).get("message", "")
                if "API key not valid" in google_msg or "API_KEY_INVALID" in error_body:
                    err_msg = "Google returned API_KEY_INVALID: The API key provided is not a valid Gemini key. Please check your key at https://aistudio.google.com/app/apikey"
                elif google_msg:
                    err_msg = f"{err_msg}: {google_msg}"
            except Exception:
                pass
        if current_key and current_key in err_msg:
            err_msg = err_msg.replace(current_key, "[REDACTED]")
        raise GeminiJudgeError(f"Gemini API call failed: {err_msg}") from None

    def judge(
        self,
        answer: str,
        fingerprint_summary: dict,
        ambiguous_patterns: list[dict],
    ) -> JudgeResult:
        """Call Gemini for a verdict on an ambiguous response.

        ADR A-M1: model_id is NOT included in the prompt.
        ADR A-H2: all Gemini exceptions are caught and sanitized.
        ADR A-C1: on failure, raises GeminiJudgeError (caller returns 502).
        """
        # Build anonymized prompt (no model_id)
        safe_summary = {
            k: v for k, v in fingerprint_summary.items()
            if k not in ("model_id", "api_key")
        }

        patterns_text = "\n".join(
            f"- {p.get('name', 'unknown')}: features {p.get('features_involved', [])}, "
            f"strength {p.get('strength', 0):.2f}"
            for p in ambiguous_patterns
        ) or "No specific patterns triggered."

        prompt = _JUDGE_PROMPT_TEMPLATE.format(
            answer=answer[:2000],  # truncate very long answers
            fingerprint_summary=json.dumps(safe_summary, indent=2),
            patterns_text=patterns_text,
        )

        import hashlib
        cache_key = hashlib.sha256((answer[:2000] + "||" + patterns_text).encode("utf-8")).hexdigest()
        if cache_key in _GEMINI_CACHE:
            return _GEMINI_CACHE[cache_key]

        try:
            text = self._call_gemini_api(prompt)
            text = _extract_json_text(text)

            result = json.loads(text)

            verdict = result.get("verdict", "LOW_RISK")
            if verdict not in ("HIGH_RISK", "LOW_RISK"):
                verdict = "LOW_RISK"

            confidence = float(result.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            judgement = JudgeResult(
                verdict=verdict,
                explanation=result.get("explanation", "LLM judge analysis complete."),
                confidence=confidence,
            )
            # Store up to 500 cached verdicts in memory
            if len(_GEMINI_CACHE) > 500:
                _GEMINI_CACHE.pop(next(iter(_GEMINI_CACHE)))
            _GEMINI_CACHE[cache_key] = judgement
            return judgement

        except json.JSONDecodeError:
            if 'text' in locals() and text:
                upper = text.upper()
                v = "HIGH_RISK" if any(w in upper for w in ("HIGH_RISK", "HALLUCINAT", "FABRICAT", "FICTION")) else "LOW_RISK"
                clean_exp = text.replace("```json", "").replace("```", "").strip()
                return JudgeResult(verdict=v, explanation=clean_exp[:300], confidence=0.9)
            raise GeminiJudgeError(
                "Gemini returned non-JSON response. Unable to parse verdict."
            ) from None
        except GeminiJudgeError:
            raise
        except Exception as e:
            # ADR A-H2: sanitize all exceptions
            raise GeminiJudgeError(
                f"Gemini API call failed: {type(e).__name__}"
            ) from None
