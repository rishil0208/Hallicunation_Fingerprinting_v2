"""Gemini judge plugin — calls Gemini API for LLM-based verdict.

ADRs enforced:
  - A-M1/S-M1: Model identity anonymized in judge prompts
  - A-H2: Gemini API key/auth info sanitized from all exceptions
  - A-C1: Judge failure returns clean error, never fake RESOLVED_AMBIGUOUS
"""
from __future__ import annotations

import json
import os
from typing import Optional

from backend.app.schemas import JudgeResult


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

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model_name = model_name
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise GeminiJudgeError(
                    "GEMINI_API_KEY not set. Set it in .env or environment."
                )
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model_name)
            except Exception as e:
                raise GeminiJudgeError(f"Failed to initialize Gemini client: {type(e).__name__}") from None
        return self._client

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
        # Remove model_id from fingerprint_summary if present
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
            client = self._get_client()
            response = client.generate_content(prompt)
            text = response.text.strip()

            # Parse JSON response
            # Handle markdown code fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

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
