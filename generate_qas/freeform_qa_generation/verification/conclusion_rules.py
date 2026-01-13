"""
Lightweight rule-based checks for validating <OBSERVATION>/<CONCLUSION> consistency.

Responsibilities:
- Parse sections from an answer string
- Ensure no new numbers appear in <CONCLUSION>
- Ensure key semantic claims in <CONCLUSION> are present in <OBSERVATION>

Note: This is a heuristic guardrail to reduce hallucinated conclusions. It is
complementary to the LLM-based verifier which uses the 2x2 panel image.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple, List, Optional
import os
import yaml


SECTION_RE = re.compile(r"<(?P<tag>OBSERVATION|CONCLUSION)>(?P<body>[\s\S]*?)</\1>", re.IGNORECASE)


def extract_sections(answer_text: str) -> Dict[str, str]:
    sections: Dict[str, str] = {"OBSERVATION": "", "CONCLUSION": ""}
    if not isinstance(answer_text, str):
        return sections
    # 1) Primary pass: strict XML-like parse requiring closing tags
    for m in SECTION_RE.finditer(answer_text):
        tag = m.group("tag").upper()
        body = m.group("body").strip()
        if tag in sections:
            sections[tag] = body

    # 2) Fallback: tolerate missing closing tags in generated answers
    # If a section is still empty but an opening tag exists, capture until the next tag or end-of-text.
    # This makes verification robust to minor formatting issues.
    def _fallback_extract(text: str, tag_name: str) -> str:
        lower = (text or "").lower()
        open_tok = f"<{tag_name.lower()}>"
        close_tok = f"</{tag_name.lower()}>"
        start = lower.find(open_tok)
        if start == -1:
            return ""
        start += len(open_tok)
        # Try to find the explicit closing tag first
        end = lower.find(close_tok, start)
        if end != -1:
            return text[start:end].strip()
        # Otherwise, cut at the next opening tag of any known section or end of string
        # Known tags we care about: OBSERVATION, CONCLUSION
        next_obs = lower.find("<observation>", start)
        next_con = lower.find("<conclusion>", start)
        # Choose the nearest positive index greater than start
        candidates = [idx for idx in [next_obs, next_con] if idx != -1]
        if candidates:
            end2 = min(candidates)
            if end2 > start:
                return text[start:end2].strip()
        return text[start:].strip()

    if not sections.get("OBSERVATION") and ("<observation>" in (answer_text or "").lower()):
        sections["OBSERVATION"] = _fallback_extract(answer_text, "OBSERVATION")
    if not sections.get("CONCLUSION") and ("<conclusion>" in (answer_text or "").lower()):
        sections["CONCLUSION"] = _fallback_extract(answer_text, "CONCLUSION")
    return sections


def _find_numbers(text: str) -> List[str]:
    # Capture integers, decimals, and percentages
    return re.findall(r"(?:\d+\.\d+|\d+)(?:\s*%|)", text or "")


def check_no_new_numbers(observation: str, conclusion: str) -> Tuple[bool, List[str]]:
    obs_nums = set(_find_numbers(observation))
    con_nums = set(_find_numbers(conclusion))
    # New numbers are those appearing in conclusion but not in observation
    new_nums = sorted([n for n in con_nums if n not in obs_nums])
    return (len(new_nums) == 0, new_nums)


KEY_TERMS = [
    # land cover & vegetation
    "vegetation", "forest", "grassland", "water", "river", "lake",
    "agricultural", "built", "building", "residential", "commercial",
    "road", "highway", "green", "park",
    # spatial descriptors
    "top-right", "top-left", "top-center", "middle-right", "middle-left", "middle-center",
    "bottom-right", "bottom-left", "bottom-center",
]


def check_terms_supported(observation: str, conclusion: str) -> Tuple[bool, List[str]]:
    obs_lower = (observation or "").lower()
    con_lower = (conclusion or "").lower()
    unsupported: List[str] = []
    for term in KEY_TERMS:
        if term in con_lower and term not in obs_lower:
            unsupported.append(term)
    return (len(unsupported) == 0, unsupported)


def _normalize_grid_name(name: str) -> str:
    if not name:
        return ""
    s = str(name).strip().lower()
    # tolerate both hyphen and space
    s = s.replace(" ", "-")
    # canonicalize center vs centre just in case
    s = s.replace("centre", "center")
    return s


def _extract_grid_mentions(text: str) -> List[str]:
    if not text:
        return []
    lower = text.lower()
    mentions: List[str] = []

    # Support both row-col ("bottom-center") and col-row ("center-bottom") order
    # Also accept "centre" spelling and space/hyphen separators
    # Row aliases: treat "center" as synonym of "middle" when used as row indicator
    row_pattern = r"(top|middle|bottom)"
    col_pattern = r"(left|center|centre|right)"

    import re as _re

    # 1) row-col order
    for m in _re.finditer(fr"\b{row_pattern}[ -]{col_pattern}\b", lower):
        row, col = m.group(1), m.group(2)
        col = col.replace("centre", "center")
        mentions.append(f"{row}-{col}")

    # 2) col-row order (normalize by swapping to row-col)
    for m in _re.finditer(fr"\b{col_pattern}[ -]{row_pattern}\b", lower):
        col, row = m.group(1), m.group(2)
        col = col.replace("centre", "center")
        mentions.append(f"{row}-{col}")

    # 3) row alias handling: allow "center" as a row synonym for "middle"
    #    Examples: "center-left", "center-right", "center-center"
    for m in _re.finditer(fr"\bcenter[ -]{col_pattern}\b", lower):
        col = m.group(1).replace("centre", "center")
        mentions.append(f"middle-{col}")
    for m in _re.finditer(fr"\b{col_pattern}[ -]center\b", lower):
        col = m.group(1).replace("centre", "center")
        mentions.append(f"middle-{col}")

    # 3b) accept 'central' as alias of 'middle' (row)
    for m in _re.finditer(fr"\bcentral[ -]{col_pattern}\b", lower):
        col = m.group(1).replace("centre", "center")
        mentions.append(f"middle-{col}")
    for m in _re.finditer(fr"\b{col_pattern}[ -]central\b", lower):
        col = m.group(1).replace("centre", "center")
        mentions.append(f"middle-{col}")

    # 4) standalone center/middle -> interpret as the central cell (middle-center)
    #    Add only if no explicit row-col mention was captured to avoid inflating counts
    if not mentions:
        if _re.search(r"\bcenter\b", lower) or _re.search(r"\bcentre\b", lower) or _re.search(r"\bmiddle\b", lower) or _re.search(r"\bcentral\b", lower):
            mentions.append("middle-center")

    # Deduplicate while preserving order
    seen = set()
    normalized: List[str] = []
    for gw in mentions:
        canon = _normalize_grid_name(gw)
        if canon not in seen:
            seen.add(canon)
            normalized.append(canon)
    return normalized


def _extract_compass_mentions(text: str) -> List[str]:
    """Detect compass-direction words like north/south/east/west and composites.
    Returns a list of matched direction tokens in canonical lowercase without spaces/hyphens.
    """
    if not text:
        return []
    import re as _re
    lower = text.lower()
    # single and composite directions (allow hyphen or space)
    parts = ["north", "south", "east", "west", "n", "s", "e", "w", "central", "center", "centre"]
    combos = [
        "northeast", "northwest", "southeast", "southwest",
        "north-east", "north-west", "south-east", "south-west",
        "north east", "north west", "south east", "south west",
        # with central
        "central-north", "central-south", "central-east", "central-west",
        "central northeast", "central northwest", "central southeast", "central southwest",
    ]
    found: List[str] = []
    # composites first
    for token in combos:
        pat = _re.escape(token)
        if _re.search(fr"\b{pat}\b", lower):
            found.append(token.replace(" ", "").replace("-", ""))
    # singles
    for p in parts:
        if _re.search(fr"\b{p}\b", lower):
            found.append(p)
    # dedupe
    seen = set()
    uniq: List[str] = []
    for t in found:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _parse_expected_grid_from_statistics(statistics_context: Optional[str], analysis_type: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Return expected grid location based on available statistics keys.

    Preference order (by analysis perspective):
      1) hydrology.best_water_accumulation_position.name or lowest_height_position (water_accumulation)
      2) best_solar_position (renewable_energy)
      3) best_svf_position (fallback for solar/openness)
      4) best_development_position (urban_development)
      5) best_scenic_position (landscape_analysis)
    """
    if not statistics_context:
        return {"expected": None, "source_key": None}
    try:
        data = yaml.safe_load(statistics_context) or {}
    except Exception:
        return {"expected": None, "source_key": None}

    grid_analysis = None
    if isinstance(data, dict):
        grid_analysis = data.get("grid_analysis") or data.get("grid", {}).get("analysis")

    def _extract_name(ga: dict, key: str) -> Optional[str]:
        if not isinstance(ga, dict):
            return None
        obj = ga.get(key)
        if isinstance(obj, dict):
            name = obj.get("name") or obj.get("grid_name")
            if isinstance(name, str):
                return _normalize_grid_name(name)
        # Sometimes provided directly as string
        if isinstance(obj, str):
            return _normalize_grid_name(obj)
        return None

    # Choose priority based on analysis_type when provided
    if analysis_type == "water_accumulation":
        keys_in_priority = [
            "hydrology.best_water_accumulation_position",
            "lowest_height_position",
        ]
    elif analysis_type == "renewable_energy":
        keys_in_priority = [
            "best_solar_position",
            "best_svf_position",
        ]
    elif analysis_type == "urban_development":
        keys_in_priority = [
            "best_development_position",
        ]
    elif analysis_type == "landscape_analysis":
        keys_in_priority = [
            "best_scenic_position",
        ]
    else:
        keys_in_priority = [
            # General priority if analysis_type is unknown
            "hydrology.best_water_accumulation_position",
            "lowest_height_position",
            "best_solar_position",
            "best_svf_position",
            "best_development_position",
            "best_scenic_position",
        ]

    if isinstance(grid_analysis, dict):
        for k in keys_in_priority:
            if k.startswith("hydrology."):
                hyd = grid_analysis.get("hydrology")
                if isinstance(hyd, dict):
                    nm = _extract_name(hyd, k.split(".", 1)[1])
                else:
                    nm = None
            else:
                nm = _extract_name(grid_analysis, k)
            if nm:
                return {"expected": nm, "source_key": k}

    # Backward-compat: sometimes an alternate nesting or naming is used
    # Try flat keys at root
    for k in keys_in_priority:
        nm = None
        try:
            obj = data.get(k)
            if isinstance(obj, dict):
                cand = obj.get("name") or obj.get("grid_name")
                if isinstance(cand, str):
                    nm = _normalize_grid_name(cand)
            elif isinstance(obj, str):
                nm = _normalize_grid_name(obj)
        except Exception:
            nm = None
        if nm:
            return {"expected": nm, "source_key": k}

    return {"expected": None, "source_key": None}


def run_rule_checks(answer_text: str, statistics_context: Optional[str] = None, analysis_type: Optional[str] = None) -> Dict[str, object]:
    sections = extract_sections(answer_text)
    observation = sections.get("OBSERVATION", "")
    conclusion = sections.get("CONCLUSION", "")

    no_new_numbers, new_numbers = check_no_new_numbers(observation, conclusion)
    terms_supported, unsupported_terms = check_terms_supported(observation, conclusion)

    # Statistics-aware grid consistency (feedback-first; non-blocking by default)
    expected = _parse_expected_grid_from_statistics(statistics_context, analysis_type=analysis_type)
    expected_grid = expected.get("expected")
    expected_source = expected.get("source_key")
    found_con = _extract_grid_mentions(conclusion)
    found_obs = _extract_grid_mentions(observation)

    grid_status = "unknown"
    grid_suggestion = None
    grid_issue_detail: Dict[str, object] = {}

    # Compass mention detection (prefer grid names over compass words like 'southeast')
    compass_in_con = _extract_compass_mentions(conclusion)
    compass_issue_detail: Dict[str, object] = {}
    if expected_grid:
        if len(found_con) == 0:
            grid_status = "missing"
            grid_suggestion = f"Mention '{expected_grid}' as the location based on {expected_source}."
        elif len(found_con) > 1:
            grid_status = "multiple"
            grid_suggestion = f"Choose a single sector; prefer '{expected_grid}' from {expected_source}."
        else:
            if _normalize_grid_name(found_con[0]) == expected_grid:
                grid_status = "match"
            else:
                grid_status = "mismatch"
                grid_suggestion = f"Use '{expected_grid}' (from {expected_source}) instead of '{found_con[0]}'."
        grid_issue_detail = {
            "expected": expected_grid,
            "source_key": expected_source,
            "found_in_conclusion": found_con,
            "found_in_observation": found_obs,
            "status": grid_status,
            "suggestion": grid_suggestion,
        }
        if compass_in_con:
            compass_issue_detail = {
                "found_compass": compass_in_con,
                "suggestion": "Use grid sector names like 'middle-left' instead of compass directions like 'southeast'.",
                "status": "compass_discouraged"
            }
    else:
        # No explicit expected grid in statistics; advise to omit grid mentions
        if len(found_con) > 1:
            grid_status = "multiple_without_stats"
            grid_suggestion = "Statistics do not provide a single best sector; omit grid mentions or choose one cautiously."
            grid_issue_detail = {
                "expected": None,
                "source_key": None,
                "found_in_conclusion": found_con,
                "found_in_observation": found_obs,
                "status": grid_status,
                "suggestion": grid_suggestion,
            }
        if compass_in_con:
            compass_issue_detail = {
                "found_compass": compass_in_con,
                "suggestion": "Avoid compass directions; prefer a single named grid sector only if statistics support it.",
                "status": "compass_discouraged"
            }

    # Lenient/strict switch (default: lenient)
    # Set CONCLUSION_RULES_STRICT=1 to enforce original strict behavior
    strict_mode = str(os.getenv("CONCLUSION_RULES_STRICT", "0")).lower() in ("1", "true", "yes")
    if strict_mode:
        # In strict mode, also require grid match if expected is present
        grid_ok = True
        if expected_grid:
            grid_ok = (grid_status in ("match", "unknown"))
        # Optionally disallow compass directions in strict mode
        compass_ok = not bool(compass_in_con)
        passed = bool(no_new_numbers and terms_supported and conclusion.strip() and grid_ok and compass_ok)
    else:
        # Lenient mode: only require that a non-empty CONCLUSION exists
        # Violations are still reported for logging/visibility
        passed = bool(conclusion.strip())
    return {
        "passed": passed,
        "observation": observation,
        "conclusion": conclusion,
        "violations": {
            "new_numbers_in_conclusion": new_numbers,
            "unsupported_terms": unsupported_terms,
            "grid_consistency": grid_issue_detail,
            "compass_terms": compass_issue_detail,
        },
        "grid_feedback": grid_issue_detail,
    }


def canonicalize_answer(answer_text: str) -> str:
    """Return a clean canonical string with exactly one OBSERVATION and one CONCLUSION.

    - Keeps the first valid <OBSERVATION>...</OBSERVATION>
    - Keeps the first valid <CONCLUSION>...</CONCLUSION>
    - Strips any nested or stray tags and extra text
    - If either section missing, returns the original text
    """
    sections = extract_sections(answer_text or "")
    obs = sections.get("OBSERVATION", "").strip()
    con = sections.get("CONCLUSION", "").strip()
    if not obs or not con:
        return answer_text or ""
    # Remove accidental nested tags from bodies
    def _strip_tags(s: str) -> str:
        return SECTION_RE.sub(" ", s).strip()
    obs = _strip_tags(obs)
    con = _strip_tags(con)
    return f"<OBSERVATION>{obs}</OBSERVATION><CONCLUSION>{con}</CONCLUSION>"


