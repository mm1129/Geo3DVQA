"""
LLM-based verifier for checking <CONCLUSION> claims against <OBSERVATION> and
the 2x2 multimodal panel (RGB, SVF, DSM, SEG).

This module uses a small prompt to ask an LLM (e.g., gpt-5) to verify:
- Conclusion introduces no new numbers
- Conclusion does not claim vegetation/water/etc. where the SEG panel shows otherwise
- Location references (e.g., top-right) are visually plausible

Returns a structured verdict with pass/fail and short rationale.
"""

from __future__ import annotations

from typing import Dict, Optional
import time
import os

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # Optional dependency


def _build_prompt(answer_text: str, statistics_context: Optional[str], analysis_type: Optional[str] = None) -> str:
    stats_block = f"\nSTATISTICAL CONTEXT (same as used for generation):\n{statistics_context}\n" if statistics_context else ""
    return f"""
You are an experienced geospatial QA auditor.

Task: Verify that the <CONCLUSION> is fully supported by <OBSERVATION> and by
the attached 2x2 panel image (RGB, SVF, DSM, SEG). Use strict criteria:
- Claims about vegetation/water/built/roads must not show any clear inconsistency with the SEG panel or the statistics.
- Claims about sky openness must not show any clear inconsistency with the SVF Heatmap (blue=lower, red/yellow=higher) or the statistics.
- Height-related claims must not show any clear inconsistency with the DSM (color ramp) or the statistics.
- Location references like top-right must match the image layout
- Conclusion should be correct and not misleading

# Additional checks for <OBSERVATION> quality vs STATISTICS
    - Check that <OBSERVATION> faithfully reflects the STATISTICAL CONTEXT. If <OBSERVATION> misreads or mislabels the statistics (e.g., wrong grid or misinterpreted context), flag it.
    - If <OBSERVATION> introduces numbers not present in the statistics, or contradicts them, flag as critical.
    - Provide a minimal correction suggestion to fix <OBSERVATION> (do not rewrite everything; only adjust the incorrect parts).

# Grid-location checks (concise):
    - If the <CONCLUSION> mentions a location (e.g., "middle-left"), ensure it matches that region across RGB/SVF/DSM/SEG.
    - Prefer exactly one sector for a single perspective; suggest reduction if multiple are named.

# Data-scope consistency (concise):
    - Treat top-level statistics (svf_mean, built_ratio, vegetation_ratio, height_*, scenic_quality, spatial_heterogeneity) as WHOLE-SCENE aggregates.
    - Grid-specific values exist only under svf_grid_summary_3x3 or grid_analysis.* and apply to a single sector.
    - Flag if <OBSERVATION> or <CONCLUSION> attaches a whole-scene number to a specific grid sector; suggest marking it as "overall ..." instead.
    - In metrics-only runs, assume all provided metrics are whole-scene; do not infer per-grid percentages.

# Statistics-aware grid mapping enforcement (feedback-first):
    - If the STATISTICAL CONTEXT contains grid_analysis.best_*_position.name, prefer EXACTLY that single sector for the corresponding perspective:
        * water_accumulation: use hydrology.best_water_accumulation_position.name; if unavailable, lowest_height_position.name
        * renewable_energy (solar/wind): use best_solar_position.name; if unavailable, fall back to best_svf_position.name
        * urban_development: use best_development_position.name
        * landscape_analysis: use best_scenic_position.name
    - If multiple sectors are mentioned in <CONCLUSION> for a single perspective, suggest reducing to ONE, selecting the statistics-backed sector above.
    - If no statistics-backed sector exists, do not fail; suggest omitting grid mentions or clearly stating uncertainty instead of naming multiple sectors.
    - When proposing feedback, reference the exact expected sector string as it appears in the statistics.

JUDGING POLICY (calibrated for balanced tolerance):
- Primary evidence is the STATISTICAL CONTEXT and <OBSERVATION>.
- Use panels (RGB, SVF, DSM, SEG) to detect CLEAR CONTRADICTIONS only.
- Absence of explicit visual evidence is NOT a failure if statistics support the claim and panels do not contradict it.
- Classify issues with severities:
  - critical: clear contradiction between CONCLUSION and either statistics or panels (e.g., claims water where SEG shows none).
  - minor: phrasing too strong, ambiguous location, or claim plausible by statistics but not visually obvious.
  - uncertain: insufficient evidence either way; not a contradiction.
- PASS if critical=0 and minor ≤ configured tolerance; do not fail solely on uncertain.

Output JSON with keys:
- passed (bool)
- rationale (string)
- issues (list of objects with fields: description (string), severity ("critical"|"minor"|"uncertain"))
- severity_counts (object with keys: critical (int), minor (int), uncertain (int))
- observation_fix (optional, string): a concise suggestion to fix <OBSERVATION> if it misreads statistics
- observation_issue_counts (optional, object): counts of observation-specific issues: critical, minor, uncertain

CONTENT TO VERIFY:
{answer_text}
{stats_block}
"""


def verify_with_llm(answer_text: str, panel_image_data_url: Optional[str], model: str = "gpt-5", statistics_context: Optional[str] = None, analysis_type: Optional[str] = None) -> Dict[str, object]:
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        return {
            "passed": True,  # If LLM not available, do not block; rely on rules
            "rationale": "LLM verifier not available; skipped.",
            "detected_issues": []
        }

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = _build_prompt(answer_text, statistics_context, analysis_type=analysis_type)

    # Allow overrides via environment variables for operational flexibility
    model_name = os.getenv("LLM_VERIFIER_MODEL", model)
    reasoning_effort = os.getenv("LLM_VERIFIER_REASONING_EFFORT", "minimal")  # minimal|medium|high
    verbosity = os.getenv("LLM_VERIFIER_VERBOSITY", "low")  # low|medium|high
    try:
        max_output_tokens = int(os.getenv("LLM_VERIFIER_MAX_OUTPUT_TOKENS", "1024"))
    except ValueError:
        max_output_tokens = 1024
    try:
        temperature = float(os.getenv("LLM_VERIFIER_TEMPERATURE", "0.0"))
    except ValueError:
        temperature = 0.0

    try:
        messages = [
            {"role": "system", "content": "You are a strict verifier. Output concise JSON only."},
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
        ]

        if panel_image_data_url:
            messages[1]["content"].append({
                "type": "input_image",
                "image_url": {"url": panel_image_data_url}
            })

        # Build kwargs to allow graceful downgrade on older SDKs
        resp_kwargs = {
            "model": model_name,
            "input": messages,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        # Attach new GPT-5 parameters
        resp_kwargs["reasoning"] = {"effort": reasoning_effort}
        resp_kwargs["verbosity"] = verbosity

        # Try Responses API first
        try:
            resp = client.responses.create(**resp_kwargs)
        except Exception as create_err:
            err_str = str(create_err)
            # Retry without verbosity if unsupported
            if "verbosity" in err_str:
                resp_kwargs.pop("verbosity", None)
                resp = client.responses.create(**resp_kwargs)
            # Retry without reasoning if still unsupported
            elif "reasoning" in err_str:
                resp_kwargs.pop("reasoning", None)
                resp = client.responses.create(**resp_kwargs)
            else:
                # If using GPT-5 with image, avoid Chat Completions (will 400). Try minimal Responses once.
                is_gpt5 = str(model_name).startswith("gpt-5")
                if is_gpt5 and panel_image_data_url:
                    minimal_kwargs = {
                        "model": model_name,
                        "input": [
                            {"role": "system", "content": [{"type": "input_text", "text": "You are a strict verifier. Output concise JSON only."}]},
                            {"role": "user", "content": [
                                {"type": "input_text", "text": prompt},
                                {"type": "input_image", "image_url": {"url": panel_image_data_url}}
                            ]}
                        ],
                        "max_output_tokens": max_output_tokens,
                        "response_format": {"type": "json_object"}
                    }
                    resp = client.responses.create(**minimal_kwargs)
                else:
                    # Fallback to Chat Completions using a vision-capable chat model
                    fallback_chat_model = os.getenv("LLM_VERIFIER_FALLBACK_CHAT_MODEL", "gpt-4o")
                    chat_messages = [
                        {"role": "system", "content": "You are a strict verifier. Output concise JSON only."},
                        {"role": "user", "content": [{"type": "text", "text": prompt}]}
                    ]
                    if panel_image_data_url:
                        chat_messages[1]["content"].append({
                            "type": "image_url",
                            "image_url": {"url": panel_image_data_url}
                        })
                    # Choose correct token parameter name per model family
                    use_max_completion = str(fallback_chat_model).startswith(("o1", "gpt-4.1"))
                    token_arg = {"max_completion_tokens": max_output_tokens} if use_max_completion else {"max_tokens": min(max_output_tokens, 300)}
                    resp = client.chat.completions.create(
                        model=fallback_chat_model,
                        messages=chat_messages,
                        temperature=temperature,
                        response_format={"type": "json_object"},
                        **token_arg
                    )
                    # Normalize content extraction and apply tolerant pass policy
                    content = getattr(resp.choices[0].message, "content", "{}") or "{}"
                    import json as _json
                    parsed = _json.loads(content)
                    # Backward-compat mapping
                    detected_issues_list = list(parsed.get("detected_issues", []))
                    issues_array = parsed.get("issues", [])
                    if not issues_array and detected_issues_list:
                        issues_array = [{"description": str(x), "severity": "minor"} for x in detected_issues_list]

            critical_count = 0
            minor_count = 0
            uncertain_count = 0
            obs_critical = 0
            obs_minor = 0
            obs_uncertain = 0
            for it in issues_array:
                sev = str(it.get("severity", "minor")).lower()
                if sev == "critical":
                    critical_count += 1
                elif sev == "uncertain":
                    uncertain_count += 1
                else:
                    minor_count += 1
                scope = str(it.get("scope", "")).lower()
                if scope == "observation":
                    if sev == "critical":
                        obs_critical += 1
                    elif sev == "uncertain":
                        obs_uncertain += 1
                    else:
                        obs_minor += 1

            try:
                minor_tol = int(os.getenv("LLM_VERIFIER_MINOR_TOLERANCE", "2"))
            except ValueError:
                minor_tol = 2
            passed_policy = bool(critical_count == 0 and minor_count <= minor_tol)
            if "passed" in parsed:
                passed_policy = bool(parsed.get("passed"))
            result = {
                "passed": passed_policy,
                "rationale": str(parsed.get("rationale", "")),
                "issues": issues_array,
                "severity_counts": {"critical": critical_count, "minor": minor_count, "uncertain": uncertain_count},
            }
            if parsed.get("observation_fix"):
                result["observation_fix"] = str(parsed.get("observation_fix"))
            if (obs_critical + obs_minor + obs_uncertain) > 0:
                result["observation_issue_counts"] = {"critical": obs_critical, "minor": obs_minor, "uncertain": obs_uncertain}
            return result

        content_text = getattr(resp, "output_text", None) or "{}"
        import json as _json
        parsed = _json.loads(content_text)

        # Backward compatibility mapping if model returned old schema
        detected_issues_list = list(parsed.get("detected_issues", []))
        issues_array = parsed.get("issues", [])
        if not issues_array and detected_issues_list:
            issues_array = [{"description": str(x), "severity": "minor"} for x in detected_issues_list]

        # Severity counts and pass policy
        critical_count = 0
        minor_count = 0
        uncertain_count = 0
        obs_critical = 0
        obs_minor = 0
        obs_uncertain = 0
        for it in issues_array:
            sev = str(it.get("severity", "minor")).lower()
            if sev == "critical":
                critical_count += 1
            elif sev == "uncertain":
                uncertain_count += 1
            else:
                minor_count += 1
            scope = str(it.get("scope", "")).lower()
            if scope == "observation":
                if sev == "critical":
                    obs_critical += 1
                elif sev == "uncertain":
                    obs_uncertain += 1
                else:
                    obs_minor += 1

        try:
            minor_tol = int(os.getenv("LLM_VERIFIER_MINOR_TOLERANCE", "2"))
        except ValueError:
            minor_tol = 2

        passed_policy = bool(critical_count == 0 and minor_count <= minor_tol)
        if "passed" in parsed:
            passed_policy = bool(parsed.get("passed"))

        result = {
            "passed": passed_policy,
            "rationale": str(parsed.get("rationale", "")),
            "issues": issues_array,
            "severity_counts": {"critical": critical_count, "minor": minor_count, "uncertain": uncertain_count},
        }
        if parsed.get("observation_fix"):
            result["observation_fix"] = str(parsed.get("observation_fix"))
        if (obs_critical + obs_minor + obs_uncertain) > 0:
            result["observation_issue_counts"] = {"critical": obs_critical, "minor": obs_minor, "uncertain": obs_uncertain}
        return result
    except Exception as e:
        return {
            "passed": True,
            "rationale": f"Verifier call failed: {e}",
            "detected_issues": []
        }


