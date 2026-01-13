import json
from dotenv import load_dotenv
from openai import OpenAI
import os
import re
import math
from collections import defaultdict

# ファイルパス
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="GPT-4 evaluation script for all captions landscape QA")
    parser.add_argument("--answer_file", type=str, 
                       default="svf_qa_re/all_captions_qa_format_qa_format.jsonl",
                       help="Path to test dataset JSONL file")
    parser.add_argument("--pred_file", type=str,
                       default="svf_qa_re/all_captions_qa_format_qa_format_rgb_20250715_gpt_4o_output_new_20250715_050455.json",
                       help="Path to model output JSON file")
    return parser.parse_args()

args = parse_args()
RESULTS_FILE = "gpt4_captions_eval_results.jsonl"
QA_OUTPUT_DIR = "qa_output_texts"
TEST_FILE = args.answer_file
OUTPUT_FILE = args.pred_file
title = OUTPUT_FILE.split("/")[-1].split(".")[0][:5]

# データ読み込み
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def extract_scores(eval_result):
    """評価結果からSky View Factor分析専用スコアを抽出"""
    scores = {
        'total_score': 1.0,
        'content_accuracy': 1.0,
        'svf_numerical_accuracy': 1.0,
        'elevation_data_accuracy': 1.0,
        'land_cover_percentage_accuracy': 1.0,
        'sky_view_factor_analysis': 1.0,
        'logical_consistency': 1.0
    }
    
    try:
        # Total Score抽出 (1-5の範囲)
        total_match = re.search(r'Total Score:\s*([1-5](?:\.\d+)?)', eval_result)
        if total_match:
            scores['total_score'] = float(total_match.group(1))
            
        # Content Accuracy抽出 (1-5の範囲)
        content_match = re.search(r'Content Accuracy:\s*([1-5](?:\.\d+)?)', eval_result)
        if content_match:
            scores['content_accuracy'] = float(content_match.group(1))
            
        # Sky View Factor分析専用スコア抽出
        svf_categories = [
            'SVF Numerical Accuracy',
            'Elevation Data Accuracy',
            'Land Cover Percentage Accuracy',
            'Sky View Factor Analysis'
        ]
        
        for category in svf_categories:
            pattern = category.replace(' ', r'\s+') + r':\s*([1-5](?:\.\d+)?|NaN)'
            match = re.search(pattern, eval_result)
            if match:
                score_key = category.lower().replace(' ', '_')
                if match.group(1).lower() == 'nan':
                    scores[score_key] = float('nan')
                else:
                    scores[score_key] = float(match.group(1))
            
        # Logical Consistency抽出 (1-5の範囲)
        logical_match = re.search(r'Logical Consistency:\s*([1-5](?:\.\d+)?)', eval_result)
        if logical_match:
            scores['logical_consistency'] = float(logical_match.group(1))
            
    except (ValueError, AttributeError) as e:
        print(f"Score extraction error: {e}")
        
    return scores

def save_results(results, output_file):
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def save_qa_to_txt(question_id, question, prediction, answer, scores, scene_id, image_path, template_id, category, output_dir, filename):
    """各質問のquestion, prediction, answerをtxtファイルに保存"""
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write(f"Question ID: {question_id}\n")
        if scene_id is not None:
            f.write(f"Scene ID: {scene_id}\n")
        if image_path is not None:
            f.write(f"Image: {image_path}\n")
        if template_id is not None:
            f.write(f"Template ID: {template_id}\n")
        if category is not None:
            f.write(f"Category: {category}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("QUESTION:\n")
        f.write("-" * 20 + "\n")
        f.write(f"{question}\n\n")
        
        f.write("GROUND TRUTH ANSWER:\n")
        f.write("-" * 20 + "\n")
        f.write(f"{answer}\n\n")
        
        f.write("MODEL PREDICTION:\n")
        f.write("-" * 20 + "\n")
        f.write(f"{prediction}\n\n")

        f.write("TEMPLATE-BASED SCORES (1-5 Scale):\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Score: {scores['total_score']:.1f}/5\n")
        f.write(f"Content Accuracy: {scores['content_accuracy']:.1f}/5\n")
        
        # Sky View Factor分析専用スコア表示
        svf_scores = {
            'svf_numerical_accuracy': 'SVF Numerical Accuracy',
            'elevation_data_accuracy': 'Elevation Data Accuracy',
            'land_cover_percentage_accuracy': 'Land Cover Percentage Accuracy',
            'sky_view_factor_analysis': 'Sky View Factor Analysis'
        }
        
        for score_key, display_name in svf_scores.items():
            score_value = scores[score_key]
            if math.isnan(score_value):
                f.write(f"{display_name}: N/A (Not applicable)\n")
            else:
                f.write(f"{display_name}: {score_value:.1f}/5\n")
            
        f.write(f"Logical Consistency: {scores['logical_consistency']:.1f}/5\n\n")

def calculate_template_averages(results):
    """Template ID別の平均スコアを計算（NaN値は除外）"""
    if not results:
        return {}, {}
        
    # Template ID別に分類
    template_results = defaultdict(list)
    for result in results:
        template_id = result.get('template_id', 'unknown')
        template_results[template_id].append(result)
    
    # 各Template IDでの平均を計算
    template_averages = {}
    template_counts = {}
    
    for template_id, template_data in template_results.items():
        template_sums = {
            'total_score': 0.0,
            'content_accuracy': 0.0,
            'svf_numerical_accuracy': 0.0,
            'elevation_data_accuracy': 0.0,
            'land_cover_percentage_accuracy': 0.0,
            'sky_view_factor_analysis': 0.0,
            'logical_consistency': 0.0
        }
        
        template_score_counts = {key: 0 for key in template_sums.keys()}
        
        for result in template_data:
            if 'scores' in result:
                for category in template_sums:
                    score = result['scores'].get(category, float('nan'))
                    if not math.isnan(score):
                        template_sums[category] += score
                        template_score_counts[category] += 1
        
        # 平均を計算（NaN値は除外）
        template_avg = {}
        template_cnt = {}
        for category in template_sums:
            if template_score_counts[category] > 0:
                template_avg[category] = template_sums[category] / template_score_counts[category]
            else:
                template_avg[category] = float('nan')
            template_cnt[category] = template_score_counts[category]
            
        template_averages[template_id] = template_avg
        template_counts[template_id] = template_cnt
    
    return template_averages, template_counts

def calculate_overall_averages(results):
    """全体の平均スコアを計算（NaN値は除外）"""
    if not results:
        return {}, {}
        
    overall_sums = {
        'total_score': 0.0,
        'content_accuracy': 0.0,
        'svf_numerical_accuracy': 0.0,
        'elevation_data_accuracy': 0.0,
        'land_cover_percentage_accuracy': 0.0,
        'sky_view_factor_analysis': 0.0,
        'logical_consistency': 0.0
    }
    
    overall_counts = {key: 0 for key in overall_sums.keys()}
    
    for result in results:
        if 'scores' in result:
            for category in overall_sums:
                score = result['scores'].get(category, float('nan'))
                if not math.isnan(score):
                    overall_sums[category] += score
                    overall_counts[category] += 1
    
    # 平均を計算（NaN値は除外）
    overall_averages = {}
    for category in overall_sums:
        if overall_counts[category] > 0:
            overall_averages[category] = overall_sums[category] / overall_counts[category]
        else:
            overall_averages[category] = float('nan')
            
    return overall_averages, overall_counts

def get_template_mapping():
    """Template IDとCategory名のマッピングを取得"""
    return {
        "environmental_solar_landscape": "environmental_energy_analysis",
        "scenic_landscape_character": "landscape_character_analysis", 
        "ecological_landscape_integration": "ecological_analysis",
        "comprehensive_landscape_description": "comprehensive_analysis",
        "spatial_organization_analysis": "spatial_structure_analysis",
        "urban_planning_perspective": "urban_development_analysis"
    }

def main():
    # .envからAPIキーを取得
    load_dotenv()
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
    output_data = load_json(OUTPUT_FILE)
    
    # output_dataがリストかどうか確認して処理
    if isinstance(output_data, list):
        output_dict = {str(item["question_id"]): item for item in output_data}
    else:
        output_dict = output_data

    # プロンプトテンプレート
    PROMPT_TEMPLATE = """
Please evaluate the model's landscape description response against the ground truth by analyzing content accuracy and template-specific requirements. Rate each aspect on a scale of 1-5:

**EVALUATION PHILOSOPHY:**
- **CRITICAL: Balance numerical accuracy with conceptual understanding**
- **CRITICAL: Apply tolerance-based evaluation - do not penalize minor numerical differences**
- **CRITICAL: Prioritize semantic equivalence over exact terminology**
- **CRITICAL: Reward partial accuracy rather than demanding complete information**

**TOLERANCE GUIDELINES - STRICTLY APPLY THESE:**
- **SVF Values**: 
  * ±0.03 tolerance for exact values (0.90 vs 0.93 = EQUIVALENT)
  * ±0.10 tolerance for mean values (0.85 vs 0.90 = EQUIVALENT)
  * Range overlaps of 50%+ = ACCEPTABLE (0.3-0.8 vs 0.01-1.0 = ACCEPTABLE)
- **Elevation Data**: 
  * ±10% tolerance for specific values (15m vs 16.5m = EQUIVALENT)
  * ±20% tolerance for mean values (50m vs 60m = EQUIVALENT)
  * General descriptors acceptable if specific values missing
- **Land Cover Percentages**: 
  * ±5% tolerance for exact percentages (70% vs 75% = EQUIVALENT)
  * ±10% tolerance for approximate percentages (60-70% vs 75% = ACCEPTABLE)
- **Landcover Classifications**: 
  * ALWAYS accept semantic equivalence using this comprehensive mapping:
  * AGRICULTURAL: farmland, cropland, agriculture, farming, crops, cultivation, arable
  * BUILT/URBAN: urban, buildings, residential, commercial, built_up, developed, constructed, infrastructure, settlement
  * FOREST: woodland, trees, forestland, wooded, tree_cover, timber, canopy
  * WATER: aquatic, river, lake, water_body, wetland, hydrographic, streams, ponds
  * GRASSLAND: pasture, meadow, grassland, grazing, grass, prairie, savanna
  * ROADS/TRANSPORT: transportation, infrastructure, pathways, transport, streets, highways

**SCORING FRAMEWORK - USE THIS EXACT SCALE:**
- **5 (Excellent)**: Exact match or within strict tolerance
- **4 (Good)**: Close match within expanded tolerance, good conceptual understanding
- **3 (Average)**: Reasonable approximation, basic conceptual understanding
- **2 (Below Average)**: Partial accuracy, some conceptual understanding
- **1 (Poor)**: Significant inaccuracy, poor conceptual understanding

**EVALUATION CRITERIA:**

1. **Content Accuracy (1-5)**: Overall accuracy of Sky View Factor analysis
   - Does the model capture the key SVF characteristics (openness, sky visibility)?
   - Are the main landscape features correctly identified?
   - **Give 3+ points if basic concepts are correct, even if details are missing**
   - **Do not penalize for missing specific numerical values if general trends are correct**

2. **Numerical Accuracy Assessment (1-5)**:

   **SVF Numerical Accuracy (1-5)**: 
   - **If specific values within tolerance: 4-5 points**
   - **If general ranges overlap significantly: 3-4 points**
   - **If conceptual understanding correct but values missing: 2-3 points**
   - **Only give 1 point if completely wrong conceptually**

   **Elevation Data Accuracy (1-5)**:
   - **If specific values within ±10% tolerance: 4-5 points**
   - **If general terrain description correct: 3-4 points**
   - **If basic understanding shown but values missing: 2-3 points**
   - **Only give 1 point if terrain description is completely wrong**

   **Land Cover Percentage Accuracy (1-5)**:
   - **If percentages within ±5% tolerance: 4-5 points**
   - **If semantic equivalence applied (agricultural=farmland): 3-4 points**
   - **If general landcover types correct but percentages off: 2-3 points**
   - **Only give 1 point if landcover types are completely wrong**

   **Sky View Factor Analysis (1-5)**:
   - **Focus primarily on conceptual understanding**
   - **If SVF relationships to landscape features are correct: 3+ points**
   - **If spatial patterns mentioned correctly: 4+ points**
   - **Reward understanding even if specific numbers are missing**

3. **Logical Consistency (1-5)**: Internal consistency and logical flow
   - Is the description internally consistent?
   - Do different parts of the response support each other logically?
   - **Give 3+ points if basic logic is sound, even if details are incomplete**

4. **Total Score (1-5)**: Overall quality considering all aspects
   - **Weight: Conceptual Understanding (40%) + Numerical Accuracy (30%) + Completeness (20%) + Logic (10%)**
   - **If model shows good conceptual understanding: minimum 3 points**
   - **Apply tolerance improvements to boost overall score**

**CRITICAL REMINDERS:**
- **Never give 1 point unless response is fundamentally wrong**
- **Always apply tolerance ranges before scoring**
- **Reward partial accuracy and conceptual understanding**
- **Focus on what the model gets right, not what it misses**

**SPECIFIC EXAMPLES FOR FAIR SCORING:**
- SVF range "0.3-0.8" vs GT "0.011-1.000" = 3-4 points (significant overlap, conceptual understanding)
- "Agricultural 60-70%" vs GT "75%" = 4 points (within expanded tolerance)
- "Buildings" vs GT "Built environment" = 5 points (perfect semantic equivalence)
- "Relatively flat terrain" vs GT "14.9m elevation" = 3 points (general descriptor acceptable)
- Missing specific numbers but correct relationships = 3+ points (conceptual understanding)

**CONTENT TO EVALUATE:**

[Question]
{question}

[Ground Truth Answer]
{reference_answer}

[Model Output]
{model_answer}

[Template ID: {template_id}]
[Category: {category}]

[Output Format - Use exactly this format]
Content Accuracy: [1-5 number]
SVF Numerical Accuracy: [1-5 number]
Elevation Data Accuracy: [1-5 number]
Land Cover Percentage Accuracy: [1-5 number]
Sky View Factor Analysis: [1-5 number]
Logical Consistency: [1-5 number]
Total Score: [1-5 number]

Brief Explanation (1 sentence each):
- Content Accuracy: [Explain why you gave this score for overall content accuracy]
- SVF Numerical Accuracy: [Explain the accuracy of SVF values]
- Elevation Data Accuracy: [Explain the accuracy of elevation data]
- Land Cover Percentage Accuracy: [Explain the accuracy of land cover percentages]
- Sky View Factor Analysis: [Explain the quality of SVF analysis]
- Logical Consistency: [Explain why you gave this score for internal consistency]
- Total Score: [Explain why you gave this overall score]
"""

    def gpt4_eval(prompt):
        try:
            response = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": "You are a fair and balanced AI evaluator for landscape analysis. Your primary goal is to assess conceptual understanding and practical accuracy. **CRITICAL INSTRUCTIONS: 1) Apply tolerance ranges strictly - do not penalize minor numerical differences. 2) Prioritize semantic equivalence over exact terminology. 3) Reward partial accuracy and conceptual understanding. 4) Focus on what the model gets right, not what it misses. 5) Never give 1 point unless the response is fundamentally wrong.** Evaluate based on template-specific requirements and assign NaN to non-applicable template categories. Always use the exact output format requested."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=512
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return f"API Error: {e}"

    # 各QAペアで評価
    results = []
    print(f"Processing {len(test_data)} questions...")
    filename = f"summary_{title}.txt"
    if os.path.exists(os.path.join(QA_OUTPUT_DIR, filename)):
        os.remove(os.path.join(QA_OUTPUT_DIR, filename))
        
    for test_item in test_data:
        qid = str(test_item["question_id"])
        if qid not in output_dict:
            print(f"Warning: Question ID {qid} not found in output data")
            continue
            
        output_item = output_dict[qid]
        
        # Template IDとCategoryを取得
        template_id = test_item.get("template_id", "unknown")
        category = test_item.get("category", "unknown")
        
        # GPT-4.1で評価
        prompt = PROMPT_TEMPLATE.format(
            question=test_item["text"],
            reference_answer=test_item["answer"],
            model_answer=output_item["answer"],
            template_id=template_id,
            category=category
        )
        
        eval_result = gpt4_eval(prompt)
        scores = extract_scores(eval_result)
        
        print(f"QID {qid} ({template_id}):\n{eval_result}\n{'='*40}")
        
        # scene_idとimage情報を抽出（存在する場合）
        scene_id = test_item.get("scene_id") or output_item.get("scene_id") or None
        image_path = test_item.get("image") or output_item.get("image") or None
        
        result_entry = {
            "question_id": qid,
            "question": test_item["text"],
            "reference_answer": test_item["answer"],
            "model_answer": output_item["answer"],
            "template_id": template_id,
            "category": category,
            "evaluation": eval_result,
            "scores": scores
        }
        
        # scene_idがある場合は追加
        if scene_id is not None:
            result_entry["scene_id"] = scene_id
            
        # image情報がある場合は追加
        if image_path is not None:
            result_entry["image"] = image_path
        
        results.append(result_entry)
        
        # question, prediction, answerをtxtファイルに保存
        save_qa_to_txt(
            qid, 
            test_item["text"], 
            output_item["answer"], 
            test_item["answer"],
            scores,
            scene_id,
            image_path,
            template_id,
            category,
            QA_OUTPUT_DIR, 
            filename
        )

    # Template ID別平均スコアを計算
    template_averages, template_counts = calculate_template_averages(results)
    overall_averages, overall_counts = calculate_overall_averages(results)
    
    print("\n" + "="*80)
    print("TEMPLATE-BASED EVALUATION RESULTS (1-5 Scale):")
    print("="*80)
    
    print(" OVERALL AVERAGES:")
    print(f"  Total Score: {overall_averages['total_score']:.2f}/5 (n={overall_counts['total_score']})")
    print(f"  Content Accuracy: {overall_averages['content_accuracy']:.2f}/5 (n={overall_counts['content_accuracy']})")
    print(f"  Logical Consistency: {overall_averages['logical_consistency']:.2f}/5 (n={overall_counts['logical_consistency']})")
    print()
    
    print(" SKY VIEW FACTOR ANALYSIS AVERAGES:")
    svf_names = {
        'svf_numerical_accuracy': ' SVF Numerical Accuracy',
        'elevation_data_accuracy': ' Elevation Data Accuracy',
        'land_cover_percentage_accuracy': ' Land Cover Percentage Accuracy',
        'sky_view_factor_analysis': ' Sky View Factor Analysis'
    }
    
    for score_key, display_name in svf_names.items():
        if not math.isnan(overall_averages[score_key]):
            print(f"  {display_name}: {overall_averages[score_key]:.2f}/5 (n={overall_counts[score_key]})")
        else:
            print(f"  {display_name}: N/A (no applicable questions)")
    
    print("\n" + "="*80)
    print(" TEMPLATE ID BREAKDOWN:")
    print("="*80)
    
    for template_id, averages in template_averages.items():
        counts = template_counts[template_id]
        print(f"\n {template_id.upper()}:")
        print(f"  Questions: {counts['total_score']}")
        print(f"  Total Score: {averages['total_score']:.2f}/5")
        print(f"  Content Accuracy: {averages['content_accuracy']:.2f}/5")
        print(f"  Logical Consistency: {averages['logical_consistency']:.2f}/5")
        
        # SVF分析専用スコアのみ表示
        for score_key, display_name in svf_names.items():
            if not math.isnan(averages[score_key]) and counts[score_key] > 0:
                print(f"  {display_name}: {averages[score_key]:.2f}/5 (n={counts[score_key]})")
    
    print("="*80)
    
    # 結果に平均スコアを追加
    final_result = {
        "evaluation_type": "template_based_captions",
        "overall_averages": overall_averages,
        "overall_counts": overall_counts,
        "template_averages": template_averages,
        "template_counts": template_counts,
        "total_questions": len(results),
        "template_distribution": {template_id: len(data) for template_id, data in template_counts.items()},
        "detailed_results": results
    }
    
    # 結果保存
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"Saved evaluation results to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
