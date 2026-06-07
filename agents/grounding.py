"""Shared observation sanitization and answer grounding for the agent pipeline.

Used by worker loops, orchestrator FINAL checks, and tool output normalization
so factual claims must trace back to tool observations — not task-specific hacks.
"""
from __future__ import annotations

import re

_PLACEHOLDER_RE = re.compile(r"\{\{[^}]+\}\}")
_UNVERIFIED_RE = re.compile(
    r"\b("
    r"could not verify|unable to|cannot determine|can't determine|"
    r"don't know|do not know|not found|couldn't find|no results|"
    r"insufficient|unknown|no data|no price|could not|cannot|"
    r"lookup failed|unavailable|did not include|did not return|"
    r"couldn't extract|could not extract"
    r")\b",
    re.I,
)
# Specific factual tokens: money, decimals, ISO dates, large integers.
_CLAIM_TOKEN_RE = re.compile(
    r"\$?\d+\.\d{2,}"
    r"|\$\d+(?:\.\d+)?"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{5,}\b",
    re.I,
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(
    r"<(?:script|style)[^>]*>.*?</(?:script|style)>",
    re.I | re.DOTALL,
)
_NOISE_OBS_RE = re.compile(
    r"^\[(?:\w+ )?(?:error|unexpected error)\]|"
    r"observation omitted|no results for|no price found|quote source unavailable",
    re.I,
)
_FACTUAL_TASK_RE = re.compile(
    r"\b(stock|share|price|quote|ticker|crypto|weather|temperature|how much|"
    r"current value|worth|cost)\b",
    re.I,
)
_EVASIVE_FINAL_RE = re.compile(
    r"\b(visit|check|go to|refer to|see|look up on|financial websites|"
    r"yahoo finance|google finance|investing\.com|marketwatch|"
    r"for the most accurate|can be found on|please visit)\b",
    re.I,
)


def looks_like_text(text: str) -> bool:
    """True when decoded body is mostly printable (not gzip/binary garbage)."""
    if not text:
        return True
    bad = sum(1 for c in text if ord(c) < 32 and c not in "\t\n\r")
    return bad / len(text) <= 0.15


def sanitize_observation(text: str) -> str:
    if looks_like_text(str(text)):
        return str(text)
    return "[observation omitted: binary or non-text response]"


def clip_observation(text: str, max_chars: int) -> str:
    one_line = " ".join(sanitize_observation(str(text)).split())
    if len(one_line) <= max_chars:
        return one_line
    return one_line[: max_chars - 3] + "..."


def html_to_text(html: str, *, limit: int = 4000) -> str:
    """Best-effort HTML → plain text for http_fetch and similar tools."""
    text = _SCRIPT_STYLE_RE.sub(" ", html)
    text = _HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[: limit - 15] + " ...[truncated]"
    return text


def normalize_tool_output(text: str) -> str:
    """Sanitize and flatten HTML tool responses before they enter the agent loop."""
    s = sanitize_observation(str(text))
    if re.search(r"<(?:html|body|head|div|p|script|!DOCTYPE)\b", s, re.I):
        s = html_to_text(s)
    return s


def extract_claim_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in _CLAIM_TOKEN_RE.finditer(text):
        token = match.group(0).lstrip("$")
        if token:
            tokens.add(token)
    return tokens


def observation_is_useful(obs: str) -> bool:
    """False for errors, empty hits, and sanitized binary placeholders."""
    if not obs or not obs.strip():
        return False
    if not looks_like_text(obs):
        return False
    if _NOISE_OBS_RE.search(obs.strip()):
        return False
    return True


def final_grounding_issue(final: str, evidence: str, *,
                          require_evidence: bool = False) -> str | None:
    """Return rejection message when FINAL is not supported by tool evidence."""
    if _PLACEHOLDER_RE.search(final):
        return (
            "FINAL contains template placeholders; delegate to a worker and "
            "use real tool results."
        )
    if _UNVERIFIED_RE.search(final):
        return None

    claims = extract_claim_tokens(final)
    if not claims:
        return None

    evidence = evidence or ""
    if require_evidence and not evidence.strip():
        return (
            "FINAL cites specific values but no tool produced usable observations "
            "yet; delegate to a worker first."
        )

    if not evidence.strip():
        return (
            "FINAL cites specific values without any tool observations; "
            "delegate to a worker first."
        )

    unmatched = sorted(c for c in claims if c not in evidence)
    if unmatched:
        return (
            f"FINAL values {unmatched} not found in tool observations; "
            "delegate again or cite only what tools returned."
        )
    return None


def task_expects_concrete_answer(task: str) -> bool:
    return bool(_FACTUAL_TASK_RE.search(task))


def final_is_evasive(final: str) -> bool:
    return bool(_EVASIVE_FINAL_RE.search(final))


def final_completeness_issue(task: str, final: str, evidence: str, *,
                             had_delegations: bool) -> str | None:
    """Reject FINAL that deflects without answering when the task needs facts."""
    if not task_expects_concrete_answer(task):
        return None
    if _UNVERIFIED_RE.search(final):
        return None
    if extract_claim_tokens(final):
        return None  # numeric grounding handles mismatch

    evidence_claims = extract_claim_tokens(evidence)
    if evidence_claims and final_is_evasive(final):
        sample = ", ".join(sorted(evidence_claims)[:5])
        return (
            f"FINAL deflects to external sites but observations contain values "
            f"({sample}); cite them directly or state you could not verify."
        )
    if had_delegations and final_is_evasive(final):
        return (
            "FINAL deflects without answering. DELEGATE another worker or state "
            "explicitly that tools could not verify the value."
        )
    if had_delegations and task_expects_concrete_answer(task):
        if not evidence_claims and not _UNVERIFIED_RE.search(final):
            if re.search(r"\b(such as|for example|websites like|please visit)\b", final, re.I):
                return (
                    "FINAL did not answer the question. Synthesize from observations "
                    "or DELEGATE a different worker with a clearer sub-task."
                )
    if had_delegations and task_expects_concrete_answer(task):
        if not extract_claim_tokens(final) and not extract_claim_tokens(evidence):
            if not _UNVERIFIED_RE.search(final):
                return (
                    "FINAL did not answer the factual question and observations "
                    "contain no numeric value; state explicitly that you could not verify."
                )
    return None
