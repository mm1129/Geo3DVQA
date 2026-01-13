"""
Verification pipeline entry points.

Expose a function `verify_answer` that combines:
1) lightweight rule checks
2) optional LLM verifier using the 2x2 panel image

Returns a dict with overall pass/fail and artifacts to store.
"""

from __future__ import annotations

from typing import Dict, Optional

from .conclusion_rules import run_rule_checks
from .llm_verifier import verify_with_llm


def verify_answer(answer_text: str, panel_image_data_url: Optional[str] = None, enable_llm: bool = True, statistics_context: Optional[str] = None, analysis_type: Optional[str] = None) -> Dict[str, object]:
    rules = run_rule_checks(answer_text, statistics_context=statistics_context, analysis_type=analysis_type)
    verdict = {
        "passed": bool(rules.get("passed", False)),
        "rules": rules,
        "llm": None,
    }

    if enable_llm:
        llm = verify_with_llm(answer_text, panel_image_data_url, statistics_context=statistics_context, analysis_type=analysis_type)
        verdict["llm"] = llm
        # Fail-open policy on LLM errors or skips: if LLM indicates a failure to run or was skipped,
        # keep the rules' result and do not additionally gate by LLM.
        llm_rationale = str((llm or {}).get("rationale", "")).lower()
        llm_failed_or_skipped = (
            ("verifier call failed" in llm_rationale) or ("skipped" in llm_rationale)
        )
        if not llm_failed_or_skipped:
            verdict["passed"] = bool(verdict["passed"] and llm.get("passed", False))

    return verdict


