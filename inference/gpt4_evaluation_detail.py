import json
from dotenv import load_dotenv
from openai import OpenAI
import os
import re
import math

# ファイルパス
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="GPT-4 evaluation script for freeform landscape QA")
    parser.add_argument("--ans_file", type=str, 
                       default=None,
                       help="Path to Ground Truth JSONL file")
    parser.add_argument("--pred_file", type=str,
                       default=None,
                       help="Path to model prediction JSON file")
    parser.add_argument("--eval_model", type=str,
                       default="gpt-4.1-mini",
                       help="OpenAI model name to run the evaluator (e.g., gpt-4.1-mini)")
    return parser.parse_args()

args = parse_args()
if args.ans_file is None:
    # "_rgb"以降を削除し、".jsonl"を追加
    TEST_FILE = re.sub(r"_2025.*$", "", args.pred_file) + ".jsonl"
    print(f"TEST_FILE: {TEST_FILE}")
    print(f"args.pred_file: {args.pred_file}")
else:
    TEST_FILE = args.ans_file
PRED_FILE = args.pred_file
today = PRED_FILE.split("/")[-1].split(".")[0][-20:-10]
RESULTS_FILE = f"free_qa_results/gpt4_freeform_eval_results_{today}.jsonl"
QA_OUTPUT_DIR = "output/qa_output_texts"
title = PRED_FILE.split("/")[-1].split(".")[0][-20:-10]

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def extract_final_answer(text):
    """テキストから<FINAL_ANSWER>...</FINAL_ANSWER>部分を抽出"""
    if not text:
        return text
    # <REASONING>...</REASONING>を削除
    text = re.sub(r'<REASONING>.*?</REASONING>', '', text, flags=re.DOTALL)
    # <FINAL_ANSWER>タグで囲まれた部分を抽出
    pattern = r'<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return text
def extract_reasoning(text):
    """テキストから<REASONING>...</REASONING>部分を抽出"""
    if not text:
        return ""
    
    # <REASONING>タグで囲まれた部分を抽出
    pattern = r'<REASONING>\s*(.*?)\s*</REASONING>'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return ""

# ==== New helpers for OBSERVATION / ANALYSIS / CONCLUSION format ====

def extract_tag_section(text: str, tag: str) -> str:
    """Extract content inside a specific XML-like tag (case-insensitive)."""
    if not text:
        return ""
    pattern = rf'<{tag}>\s*(.*?)\s*</{tag}>'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    # return match.group(1).strip() if match else ""
    if match:
        return match.group(1).strip()
    else:
        # 条件をゆるくして再search
        if tag == "OBSERVATION":
            next_tag = "ANALYSIS"
            pattern = rf'{tag}\s*(.*?)\s*{next_tag}'
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
            else:
                return ""
        elif tag == "ANALYSIS":
            next_tag = "CONCLUSION"
            pattern = rf'{tag}\s*(.*?)\s*{next_tag}'
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
            else:
                return ""
        elif tag == "CONCLUSION":
            # Extract everything from "CONCLUSION" to the end
            pattern = rf'{tag}\s*(.*)'
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
            else:
                return ""

def count_words_without_tags(text: str) -> int:
    """Remove tags and count words in the remaining text."""
    if not text:
        return 0
    no_tags = re.sub(r'<[^>]+>', ' ', text)
    no_tags = re.sub(r'\s+', ' ', no_tags).strip()
    if not no_tags:
        return 0
    return len(no_tags.split())

def combine_sections(observation: str, analysis: str, conclusion: str) -> str:
    parts = []
    if observation:
        parts.append(observation)
    if analysis:
        parts.append(analysis)
    if conclusion:
        parts.append(conclusion)
    return "\n\n".join(parts)

def detect_gt_domain_presence(text: str) -> dict:
    """Detect presence of domains in GT to decide NaN applicability.
    NaN should be used ONLY when GT lacks the domain entirely.
    """
    lower = (text or "").lower()
    has_svf = bool(re.search(r'\bsvf\b|sky view factor|sky visibility|sky openness', lower))
    has_elev = bool(re.search(r'\belevation\b|terrain|slope|\bheight\b|\bm\b', lower))
    has_landcover = bool(re.search(r'land ?cover|landcover|vegetation|forest|agricultur|built|urban|water', lower))
    return {
        'svf': has_svf,
        'elevation': has_elev,
        'landcover': has_landcover,
    }

# (Removed presence flags to simplify output and logic)

def extract_scores(eval_result):
    """Extract scores from evaluator output (supports new and legacy keys)."""
    scores = {
        'total_score': 1.0,
        'observation_score': 1.0,
        'conclusion_score': 1.0,
        'svf_analysis': 1.0,
        'land_cover_analysis': 1.0,
        'elevation_analysis': 1.0,
        'logical_consistency': 1.0
    }

    try:
        # Total Score
        total_match = re.search(r'Total Score:\s*([1-5](?:\.\d+)?)', eval_result)
        if total_match:
            scores['total_score'] = float(total_match.group(1))

        # New section scores
        obs_match = re.search(r'Observation Score:\s*([1-5](?:\.\d+)?)', eval_result)
        if obs_match:
            scores['observation_score'] = float(obs_match.group(1))
        con_match = re.search(r'Conclusion Score:\s*([1-5](?:\.\d+)?)', eval_result)
        if con_match:
            scores['conclusion_score'] = float(con_match.group(1))

        # Legacy mapping (fallback)
        answer_match = re.search(r'Answer Score:\s*([1-5](?:\.\d+)?)', eval_result)
        if answer_match and not con_match:
            scores['conclusion_score'] = float(answer_match.group(1))

        # Domain analyses
        svf_match = re.search(r'SVF Analysis:\s*([1-5](?:\.\d+)?|NaN)', eval_result)
        if svf_match:
            if svf_match.group(1).lower() == 'nan':
                scores['svf_analysis'] = float('nan')
            else:
                scores['svf_analysis'] = float(svf_match.group(1))
        land_cover_match = re.search(r'Land Cover Analysis:\s*([1-5](?:\.\d+)?|NaN)', eval_result)
        if land_cover_match:
            if land_cover_match.group(1).lower() == 'nan':
                scores['land_cover_analysis'] = float('nan')
            else:
                scores['land_cover_analysis'] = float(land_cover_match.group(1))
        elevation_match = re.search(r'Elevation Analysis:\s*([1-5](?:\.\d+)?|NaN)', eval_result)
        if elevation_match:
            if elevation_match.group(1).lower() == 'nan':
                scores['elevation_analysis'] = float('nan')
            else:
                scores['elevation_analysis'] = float(elevation_match.group(1))
        logical_match = re.search(r'Logical Consistency:\s*([1-5](?:\.\d+)?)', eval_result)
        if logical_match:
            scores['logical_consistency'] = float(logical_match.group(1))
    except (ValueError, AttributeError) as e:
        print(f"Score extraction error: {e}")
    return scores

def save_results(results, output_file):
    output_dir = os.path.dirname(output_file)
    # Handle case where output_dir is empty (relative path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def save_qa_to_txt(
    question_id,
    question,
    prediction_full,
    answer_full,
    ref_observation,
    ref_analysis,
    ref_conclusion,
    model_observation,
    model_analysis,
    model_conclusion,
    model_word_count,
    scores,
    evaluation_text,
    scene_id,
    image_path,
    output_dir,
    filename,
):
    """Save detailed QA with OBSERVATION/ANALYSIS/CONCLUSION sections."""
    os.makedirs(output_dir, exist_ok=True)
    
    # filename = f"summary_{}.txt"
    filepath = os.path.join(output_dir, filename)
    print(f"summary txt filepath: {filepath}")
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write(f"Question ID: {question_id}\n")
        if scene_id is not None:
            f.write(f"Scene ID: {scene_id}\n")
        if image_path is not None:
            f.write(f"Image: {image_path}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("QUESTION:\n")
        f.write("-" * 20 + "\n")
        f.write(f"{question}\n\n")

        f.write("GROUND TRUTH ANSWER (FULL):\n")
        f.write("-" * 20 + "\n")
        f.write(f"{answer_full}\n\n")
        
        f.write("GROUND TRUTH SECTIONS:\n")
        f.write("-" * 20 + "\n")
        f.write("<OBSERVATION>\n")
        f.write(f"{ref_observation}\n")
        f.write("</OBSERVATION>\n")
        # f.write("<ANALYSIS>\n")
        # f.write(f"{ref_analysis}\n")
        # f.write("</ANALYSIS>\n")
        f.write("<CONCLUSION>\n")
        f.write(f"{ref_conclusion}\n")
        f.write("</CONCLUSION>\n\n")

        f.write("MODEL PREDICTION (FULL):\n")
        f.write("-" * 20 + "\n")
        f.write(f"{prediction_full}\n\n")

        f.write("MODEL SECTIONS:\n")
        f.write("-" * 20 + "\n")
        f.write("<OBSERVATION>\n")
        f.write(f"{model_observation}\n")
        f.write("</OBSERVATION>\n")
        # f.write("<ANALYSIS>\n")
        # f.write(f"{model_analysis}\n")
        # f.write("</ANALYSIS>\n")
        f.write("<CONCLUSION>\n")
        f.write(f"{model_conclusion}\n")
        f.write("</CONCLUSION>\n\n")

        f.write("VALIDATION METRICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Model Total Word Count (sections only): {model_word_count}\n\n")

        f.write("SPECIALIZED DOMAIN SCORES (1-5 Scale):\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Score: {scores['total_score']:.1f}/5\n")
        f.write(f"Observation Score: {scores['observation_score']:.1f}/5\n")
        # f.write(f"Analysis Score: {scores['analysis_score']:.1f}/5\n")
        f.write(f"Conclusion Score: {scores['conclusion_score']:.1f}/5\n")
        f.write(f"SVF Analysis: {scores['svf_analysis']:.1f}/5\n") if not math.isnan(scores['svf_analysis']) else f.write("SVF Analysis: N/A (No relevant GT info)\n")
        f.write(f"Land Cover Analysis: {scores['land_cover_analysis']:.1f}/5\n") if not math.isnan(scores['land_cover_analysis']) else f.write("Land Cover Analysis: N/A (No relevant GT info)\n")
        f.write(f"Elevation Analysis: {scores['elevation_analysis']:.1f}/5\n") if not math.isnan(scores['elevation_analysis']) else f.write("Elevation Analysis: N/A (No relevant GT info)\n")
        f.write(f"Logical Consistency: {scores['logical_consistency']:.1f}/5\n")

        # --- Detailed evaluation output from GPT-4 ---
        if evaluation_text:
            f.write("\nEVALUATION DETAIL (raw output):\n")
            f.write("-" * 20 + "\n")
            f.write(f"{evaluation_text}\n\n")

def calculate_category_averages(results):
    """カテゴリー別の平均スコアを計算（NaN値は除外）"""
    if not results:
        return {}
        
    category_sums = {
        # evaluation scores
        'total_score': 0.0,
        'observation_score': 0.0,
        'conclusion_score': 0.0,
        'svf_analysis': 0.0,
        'land_cover_analysis': 0.0,
        'elevation_analysis': 0.0,
        'logical_consistency': 0.0,
        # categories
        'urban_development': 0.0,
        'energy_installation': 0.0,
        'landscape_analysis': 0.0,
        'water_accumulation': 0.0,
    }

    category_counts = {
        'total_score': 0,
        'observation_score': 0,
        'conclusion_score': 0,
        'svf_analysis': 0,
        'land_cover_analysis': 0,
        'elevation_analysis': 0,
        'logical_consistency': 0,
        'urban_development': 0,
        'energy_installation': 0,
        'landscape_analysis': 0,
        'water_accumulation': 0,    
    }
    
    for result in results:
        if 'scores' in result:
            scores = result['scores']
            for key, value in scores.items():
                if key in category_sums and not math.isnan(value):
                    category_sums[key] += value
                    category_counts[key] += 1
    
    # 平均を計算
    category_averages = {}
    for key in category_sums:
        if category_counts[key] > 0:
            category_averages[key] = category_sums[key] / category_counts[key]
        else:
            category_averages[key] = float('nan')
            
    return category_averages, category_counts
def create_prediction_mapping(pred_data):
    """予測データをscene_id + categoryでマッピング"""
    pred_dict = {}
    for item in pred_data:
        question_id = item["question_id"]
        image = item["image"]
        key = f"{question_id}_{image}"
        pred_dict[key] = item
    return pred_dict

def match_ground_truth_to_prediction(test_data, pred_dict):
    """Ground TruthとPredictionをマッチング"""
    matched_pairs = []
    
    for gt_item in test_data:
        question_id = gt_item["question_id"]
        image = gt_item["image"]
        key = f"{question_id}_{image}"
        
        if key in pred_dict:
            matched_pairs.append({
                "ground_truth": gt_item,
                "prediction": pred_dict[key]
            })
        else:
            print(f"Warning: No prediction found for {key}")
    
    return matched_pairs

def create_evaluation_prompt():
    """Create GPT-4 evaluation prompt for OBSERVATION/ANALYSIS/CONCLUSION format."""
    PROMPT_TEMPLATE = """
    Please evaluate the model's response against the ground truth by analyzing the sections and specialized domains. Rate each aspect on a scale of 1-5.

    CORE PHILOSOPHY (LENIENCY RULES):
    - Prioritize directional/semantic correctness over exact numbers and wording.
    - Numeric tolerance guidance:
      - ≈0–20% relative error with correct trend → treat as near-correct.
      - ≈20–40% error with correct trend → minor penalty only.
      - >40% error but clear correct trend or lacks a few numeric mentioning in prediction → moderate score (do not drop to 1–2 solely for numeric error).
      - Incorrect trend/direction (reversals) or no numeric mentioning in prediction → strong penalty.
    - Spatial references can be 3x3 labels or clear descriptive/coordinate-based mentions. Reward specificity. Penalize 1-2 points for not mentioning spatial references, or wrong spatial references.
    - Structural constraints (tags, word count) are soft: apply mild penalties only if violations are large/persistent.

    EVALUATION CRITERIA:

    1. Observation Score (1-5): Evaluate ONLY the <OBSERVATION>...</OBSERVATION> section
       - Assess correctness/completeness of statistics and visual observations.
       - Reward correct trends and reasonable numeric ranges even if not exact.

    2. Conclusion Score (1-5): Evaluate ONLY the <CONCLUSION>...</CONCLUSION> section
       - How accurate and complete is the final 
       assessment and recommendations?
       - If prediction does not mention grid location despite GT mentions, gives a strong penalty.
       - **no penalty here** for not mentioning statistics of SVF, Elevation, Landcover in the prediction.
       - Word count is soft: within ~60–100 words → at most a small penalty; egregiously short/long → larger penalty.

    3. SVF Analysis (1-5 or NaN): Understanding of sky view factor and spatial openness
       - Emphasize trend/location correctness; allow numeric deviations per tolerance rules above.
       - Assign NaN ONLY if the ground truth contains NO SVF information.

    4. Land Cover Analysis (1-5 or NaN): Vegetation and land use
       - Emphasize correct class identification and relative proportions by area/zone.
       - Assign NaN ONLY if the ground truth contains NO land cover information.

    5. Elevation Analysis (1-5 or NaN): Terrain and elevation
       - Emphasize correct relative elevation/flatness gradients and zones.
       - Assign NaN ONLY if the ground truth contains NO elevation/terrain information.

    6. Logical Consistency (1-5): Consistency between sections
       - Conclusions must be supported by observations; small numeric drifts are acceptable if direction aligns.

    7. Total Score (1-5): Overall quality combining all aspects
       - Favor conceptual/directional understanding; avoid over-penalizing minor numeric or format deviations.
       - If prediction does not mention grid location despite GT mentions, gives a strong penalty.
       - prediction should follow the question guidance when GT follows it.


    CONSTRAINTS AND HOW TO APPLY THEM (SOFT):
    - Structure: Prefer <OBSERVATION>, <CONCLUSION>. Missing tags → small penalty unless content is severely disorganized.
    - Word Count: Target 70–80 words (Model: {model_word_count}; GT: {reference_word_count}). 60–130 words is acceptable with at most a very small penalty; 130–150 words → small penalty.
    - Required Content: Prefer explicit SVF, Elevation, Landcover mentions if present in GT; if one is omitted but trends in others are strong, reduce the score.
    - Location Specificity: Reward explicit spatial references; descriptive equivalents are acceptable. Give additional credit for explicit 3x3 grid codes (A1–C3) or precise zone names.

    CONTENT TO EVALUATE:

    [Question]
    {question}

    [Ground Truth Sections]
    <OBSERVATION>
    {ref_observation}
    </OBSERVATION>
    <CONCLUSION>
    {ref_conclusion}
    </CONCLUSION>

    [Ground Truth Domain Presence]
    - SVF mentioned in GT: {gt_has_svf}
    - Elevation mentioned in GT: {gt_has_elevation}
    - Landcover mentioned in GT: {gt_has_landcover}

    [Model Output Sections]
    <OBSERVATION>
    {model_observation}
    </OBSERVATION>
    <CONCLUSION>
    {model_conclusion}
    </CONCLUSION>

    [Output Format - Use exactly this format]
    Observation Score: [1-5 number]
    Conclusion Score: [1-5 number]
    SVF Analysis: [1-5 number or NaN]
    Land Cover Analysis: [1-5 number or NaN]
    Elevation Analysis: [1-5 number or NaN]
    Logical Consistency: [1-5 number]
    Total Score: [1-5 number]

    Brief Explanation (1 sentence each):
    - Observation Score: [Explain why for the <OBSERVATION> section]
    - Conclusion Score: [Explain why for the <CONCLUSION> section, note any structure/word-count violations]
    - SVF Analysis: [Explain why you gave this score for SVF/sky visibility analysis understanding]
    - Land Cover Analysis: [Explain why you gave this score for land cover analysis understanding]
    - Elevation Analysis: [Explain why you gave this score for elevation/terrain analysis understanding]
    - Logical Consistency: [Explain why you gave this score for consistency across sections]
    - Total Score: [Explain why you gave this overall score considering all aspects]
    """
    return PROMPT_TEMPLATE

def gpt4_eval(client, prompt, model):
    """GPT-4による評価実行"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a strict and fair AI evaluator focusing on semantic understanding and conceptual accuracy for freeform landscape analysis. Prioritize directional consistency and meaning equivalence over exact textual matching. Evaluate based on conceptual understanding rather than precise wording. Always use the exact output format requested."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return f"API Error: {e}"

def main():
    # Use secure API key management
    from utils.api_key_manager import get_openai_api_key, validate_input_path
    
    try:
        api_key = get_openai_api_key()
    except ValueError as e:
        # Use English error message for consistency
        raise ValueError(f"API key error: {str(e)}")
    
    client = OpenAI(api_key=api_key)
    
    # Validate input files
    if TEST_FILE:
        validate_input_path(TEST_FILE, "file")
    if PRED_FILE:
        validate_input_path(PRED_FILE, "file")

    # データ読み込み
    test_data = load_jsonl(TEST_FILE)
    pred_data = load_json(PRED_FILE)
    
    print(f"Loaded {len(test_data)} ground truth questions")
    print(f"Loaded {len(pred_data)} predictions")
    
    # データマッピング
    pred_dict = create_prediction_mapping(pred_data)
    matched_pairs = match_ground_truth_to_prediction(test_data, pred_dict)
    
    print(f"Successfully matched {len(matched_pairs)} question-prediction pairs")
    
    # 評価プロンプトテンプレートを取得
    prompt_template = create_evaluation_prompt()
    
    # 各QAペアで評価
    results = []
    print(f"Processing {len(matched_pairs)} question-prediction pairs...")
    filename = f"summary_{title}.txt"
    if os.path.exists(os.path.join(QA_OUTPUT_DIR, filename)):
        os.remove(os.path.join(QA_OUTPUT_DIR, filename))
        
    for i, pair in enumerate(matched_pairs):
        gt_item = pair["ground_truth"]
        pred_item = pair["prediction"]
        
        # Extract OBSERVATION / CONCLUSION sections (ANALYSIS removed)
        ref_observation = extract_tag_section(gt_item["answer"], "OBSERVATION")
        ref_analysis = ""
        ref_conclusion = extract_tag_section(gt_item["answer"], "CONCLUSION")

        # Model output sections (ANALYSIS removed)
        model_observation = extract_tag_section(pred_item["answer"], "OBSERVATION")
        model_analysis = ""
        model_conclusion = extract_tag_section(pred_item["answer"], "CONCLUSION") or extract_final_answer(pred_item["predict"]) 

        # Word counts (total across sections)
        reference_word_count = count_words_without_tags(combine_sections(ref_observation, "", ref_conclusion))
        model_word_count = count_words_without_tags(combine_sections(model_observation, "", model_conclusion))

        # GT domain presence flags (govern NaN applicability)
        gt_flags = detect_gt_domain_presence(combine_sections(ref_observation, "", ref_conclusion))

        # GPT-4で評価
        prompt = prompt_template.format(
            question=gt_item["text"],
            ref_observation=ref_observation,
            ref_analysis=ref_analysis,
            ref_conclusion=ref_conclusion,
            model_observation=model_observation,
            model_analysis=model_analysis,
            model_conclusion=model_conclusion,
            model_word_count=model_word_count,
            reference_word_count=reference_word_count,
            gt_has_svf=str(gt_flags['svf']),
            gt_has_elevation=str(gt_flags['elevation']),
            gt_has_landcover=str(gt_flags['landcover']),
            
        )
        
        eval_result = gpt4_eval(client, prompt, args.eval_model)
        scores = extract_scores(eval_result)
        
        print(f"Processed {i+1}/{len(matched_pairs)}: {gt_item['category']} - {gt_item['scene_id']}")
        
        # 結果エントリを作成
        result_entry = {
            "scene_id": gt_item["scene_id"],
            "category": gt_item["category"],
            "question_id": gt_item.get("question_id", i+1),
            "question": gt_item["text"],
            "reference_observation": ref_observation,
            # "reference_analysis": ref_analysis,
            "reference_conclusion": ref_conclusion,
            "model_observation": model_observation,
            # "model_analysis": model_analysis,
            "model_conclusion": model_conclusion,
            "model_word_count": model_word_count,
            # Backward-compatible fields
            "reference_final_answer": ref_conclusion,
            "model_final_answer": model_conclusion,
            # "reference_reasoning": ref_analysis,
            # "model_reasoning": model_analysis,
            "evaluation": eval_result,
            "scores": scores
        }
        print("scores", scores)
        
        results.append(result_entry)
        
        # 詳細結果をTXTファイルに保存（新フォーマット対応）
        save_qa_to_txt(
            result_entry["question_id"],
            gt_item["text"],
            pred_item["answer"],
            gt_item["answer"],
            ref_observation,
            ref_analysis,
            ref_conclusion,
            model_observation,
            model_analysis,
            model_conclusion,
            model_word_count,
            scores,
            eval_result,
            gt_item["scene_id"],
            gt_item.get("image", None),
            QA_OUTPUT_DIR,
            filename
        )

    print(f"\\n評価完了: {len(results)}件のQAペアを処理しました")
    
    # 全体統計を計算
    category_averages, category_counts = calculate_category_averages(results)
    
    # カテゴリ別統計を計算
    category_stats = calculate_category_specific_stats(results)
    
    # 結果を表示
    display_results(category_averages, category_counts, category_stats, results)
    
    # 結果を保存
    final_result = {
        "evaluation_philosophy": {
            "semantic_focus": True,
            "accepts_expression_variations": True,
            "prioritizes_directional_consistency": True
        },
        "category_averages": category_averages,
        "category_counts": category_counts,
        "category_specific_stats": category_stats,
        "total_questions": len(results),
        "detailed_results": results
    }
    
    # 結果保存
    save_results(results, RESULTS_FILE)
    
    with open(RESULTS_FILE.replace('.jsonl', '_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"Saved evaluation results to {RESULTS_FILE}")
    print(f"Saved summary to {RESULTS_FILE.replace('.jsonl', '_summary.json')}")

def calculate_category_specific_stats(results):
    """カテゴリ別の詳細統計を計算"""
    categories = {}
    
    for result in results:
        category = result["category"]
        if category not in categories:
            categories[category] = {
                'total_score': [],
                'observation_score': [],
                'conclusion_score': [],
                'svf_analysis': [],
                'land_cover_analysis': [],
                'elevation_analysis': [],
                'logical_consistency': []
            }
        
        scores = result["scores"]
        for key, value in scores.items():
            if key in categories[category] and not math.isnan(value):
                categories[category][key].append(value)
    
    # 各カテゴリの平均値を計算
    category_stats = {}
    for category, score_lists in categories.items():
        category_stats[category] = {}
        for score_type, scores in score_lists.items():
            if scores:
                category_stats[category][score_type] = {
                    'mean': sum(scores) / len(scores),
                    'count': len(scores)
                }
            else:
                category_stats[category][score_type] = {
                    'mean': float('nan'),
                    'count': 0
                }
    
    return category_stats

def display_results(category_averages, category_counts, category_stats, results):
    """結果を表示"""
    print("\\n" + "="*60)
    print("FREEFORM QA EVALUATION RESULTS (1-5 Scale):")
    print("="*60)
    print("GENERAL EVALUATION:")
    print(f"  Total Score: {category_averages['total_score']:.2f}/5 (n={category_counts['total_score']})")
    print(f"  Observation Score: {category_averages['observation_score']:.2f}/5 (n={category_counts['observation_score']})")
    # print(f"  Analysis Score: {category_averages['analysis_score']:.2f}/5 (n={category_counts['analysis_score']})")
    print(f"  Conclusion Score: {category_averages['conclusion_score']:.2f}/5 (n={category_counts['conclusion_score']})")
    print(f"  Logical Consistency: {category_averages['logical_consistency']:.2f}/5 (n={category_counts['logical_consistency']})")
    print()
    print("SPECIALIZED DOMAIN ANALYSIS:")
    
    if not math.isnan(category_averages['svf_analysis']):
        print(f"  SVF Analysis: {category_averages['svf_analysis']:.2f}/5 (n={category_counts['svf_analysis']})")
    else:
        print(f"  SVF Analysis: N/A (no applicable questions)")
    
    if not math.isnan(category_averages['land_cover_analysis']):
        print(f"  Land Cover Analysis: {category_averages['land_cover_analysis']:.2f}/5 (n={category_counts['land_cover_analysis']})")
    else:
        print(f"  Land Cover Analysis: N/A (no applicable questions)")
    
    if not math.isnan(category_averages['elevation_analysis']):
        print(f"  Elevation Analysis: {category_averages['elevation_analysis']:.2f}/5 (n={category_counts['elevation_analysis']})")
    else:
        print(f"  Elevation Analysis: N/A (no applicable questions)")
    
    print()
    print("CATEGORY-SPECIFIC PERFORMANCE:")
    category_names = {
        'urban_development_application': 'Urban Development',
        'renewable_energy_installation': 'Renewable Energy',
        'landscape_analysis': 'Landscape Analysis',
        'water_accumulation': 'Water Accumulation'
    }
    
    for category, stats in category_stats.items():
        display_name = category_names.get(category, category)
        total_score = stats['total_score']
        if total_score['count'] > 0:
            print(f"  {display_name}: {total_score['mean']:.2f}/5 (n={total_score['count']})")
        else:
            print(f"  {display_name}: N/A (no questions)")
    
    print("="*60)
    
if __name__ == "__main__":
    main()
    