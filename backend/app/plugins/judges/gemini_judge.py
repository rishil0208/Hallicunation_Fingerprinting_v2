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


class GeminiJudgePlugin:
    """Calls Gemini API for LLM-based hallucination verdict.

    Only invoked on AMBIGUOUS gate verdicts. Model identity is
    anonymized in the prompt (ADR A-M1/S-M1).
    """

    name = "gemini"

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-3.6-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model_name = model_name

    def _call_gemini_api(self, prompt: str) -> str:
        if not self.api_key:
            raise GeminiJudgeError("GEMINI_API_KEY not set. Set it in .env or environment.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }

        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            raise GeminiJudgeError(f"Gemini API call failed: {type(e).__name__}") from None

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

        try:
            text = self._call_gemini_api(prompt)
            text = _extract_json_text(text)

            result = json.loads(text)

            verdict = result.get("verdict", "LOW_RISK")
            if verdict not in ("HIGH_RISK", "LOW_RISK"):
                verdict = "LOW_RISK"

            confidence = float(result.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            return JudgeResult(
                verdict=verdict,
                explanation=result.get("explanation", "LLM judge analysis complete."),
                confidence=confidence,
            )

        except json.JSONDecodeError:
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
