#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import re
import argparse
from datetime import datetime
from collections import OrderedDict
import math
import random



# local helpers
try:
    from prediction_scanner import list_jsonl_files
    from csv_writer import append_rows_csv
    from category_mappings import (
        build_csv_fieldnames_with_subcategories,
        get_major_to_subcategories,
    )
except Exception:
    # Fallback for different working directories
    from .prediction_scanner import list_jsonl_files  # type: ignore
    from .csv_writer import append_rows_csv  # type: ignore
    from .category_mappings import (  # type: ignore
        build_csv_fieldnames_with_subcategories,
        get_major_to_subcategories,
    )

def calculate_random_baseline(category, total_questions):
    """
    Calculate expected accuracy for random answers by question type
    
    :param category: Question type
    :param total_questions: Total number of questions in this category
    :return: Expected accuracy (percentage)
    """
    if category in ['openness_assessment', "regional_svf_variability", "sky_visibility", "sun_exposure", "urban_density", "visibility_range"]:
        return 25.00
    CHOICE_COUNTS = {
        'hard_pixel': 10,
        'hard_ranking': 6,
        'height_average': math.nan,
        'highest_region': 4,
        'land_use': math.nan,
        'landcover_type': math.nan,
    }
    
    num_choices = CHOICE_COUNTS[category] if category in CHOICE_COUNTS else math.nan
    if math.isnan(num_choices):
        return math.nan
    return 100.0 / num_choices
def extract_prompt_type(question_text):
    """
    Extract prompt type from question text (part before first \n)
    
    :param question_text: Question text
    :return: Prompt type
    """
    if not question_text:
        return "Unknown"
    
    first_line = question_text.split('\n')[0].strip()
    return first_line if first_line else "Unknown"

def load_answer_file(file_path):
    answers = {}
    line_count = 0
    
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    if line.strip():
                        data = json.loads(line)
                        question_id = data.get('question_id', line_num)
                        answers[question_id] = data
                        line_count += 1
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error (line {line_num}): {e}")
                    continue
    except Exception as e:
        print(f"File reading error: {e}")
        return {}
    
    print(f"Loaded {line_count} items from answer file")
    return answers
def judge_correct(pred_norm, ans_norm, category, question_text="", correct_answer="", extracted_prediction=""):
    """
    Judge if prediction is correct
    
    :param pred_norm: Normalized prediction answer
    :param ans_norm: Normalized correct answer
    :param category: Question category
    :param question_text: Question text (for height_inference)
    :param correct_answer: Correct answer (for landcover_type, land_use)
    :param extracted_prediction: Extracted prediction answer (for landcover_type, land_use)
    :return: Boolean indicating if answer is correct
    """
    # height_inference subtype aggregation
    if category == "height_inference":
        height_inference_subtypes = [
            "What is the average height in this image?",
            "What height patterns are observable?",
            "What elevation characteristics can be inferred?",
            "What is the typical vertical dimension?",
            "What altitude information can be determined?"
        ]
        return pred_norm == ans_norm and pred_norm != ""
    elif category == "height_average":
        import re
        pred_height = re.search(r'\d+', pred_norm)
        ans_height = re.search(r'\d+', ans_norm)
        if pred_height and ans_height:
            pred_val = int(pred_height.group())
            ans_val = int(ans_height.group())
            # Allow error within 30% of ans_val
            if ans_val == 0:
                return pred_val == 0
            if ans_val > 30:
                return abs(pred_val - ans_val) <= ans_val * 0.3
            else:
                return abs(pred_val - ans_val) < 10
        return False
    # landcover_type Jaccard scoring
    elif category == "landcover_type":
        gt_set = [x.strip().lower() for x in correct_answer.split(",") if x.strip()]
        pred_set = [x.strip().lower() for x in extracted_prediction.split(",") if x.strip()]
        
        def jaccard_similarity_local(set1, set2):
            s1 = set([x.strip().lower() for x in set1])
            s2 = set([x.strip().lower() for x in set2])
            if not s1 and not s2:
                return 1.0
            if not s1 or not s2:
                return 0.0
            intersection = s1 & s2
            union = s1 | s2
            return len(intersection) / len(union)
        
        jac = jaccard_similarity_local(gt_set, pred_set)
        landcover_type_jaccard_threshold = 0.7
        return jac >= landcover_type_jaccard_threshold
    
    # land_use set matching (order-independent exact match)
    elif category == "land_use":
        gt_set = set([x.strip().lower() for x in correct_answer.split(",") if x.strip()])
        pred_set = set([x.strip().lower() for x in extracted_prediction.split(",") if x.strip()])
        return gt_set == pred_set
    elif category == "hard_ranking":
        # print(f"DEBUG: pred_norm: {pred_norm}, ans_norm: {ans_norm}")
        # For hard_ranking, accept if the order of regions matches, supporting both "Region X" and "X" formats
        import re

        def extract_region_order(text):
            # Extract list like ['region a', 'region b', ...] in order of appearance (case-insensitive)
            regions = [m.group(0).lower() for m in re.finditer(r'region\s+[a-z]', text, re.IGNORECASE)]
            if regions:
                return regions
            # If no "Region X" format found, extract single letters that could be region identifiers
            single_letters = [m.group(0).lower() for m in re.finditer(r'\b[a-z]\b', text, re.IGNORECASE)]
            # Convert single letters to "region x" format for comparison
            return [f'region {letter}' for letter in single_letters]

        pred_regions = extract_region_order(pred_norm)
        ans_regions = extract_region_order(ans_norm)
        if pred_regions and ans_regions:
            return pred_regions == ans_regions
        return pred_norm == ans_norm
    elif "region" in ans_norm:
        import re
        if "region" in pred_norm and "region" in ans_norm:
            region_match = re.search(r'region\s+[a-z]', pred_norm, re.IGNORECASE)
            if region_match:
                pred_region = region_match.group(0).lower()
                return pred_region == ans_norm
        elif "region" not in pred_norm and "region" in ans_norm:
            # remove "region" from ans_norm
            ans_norm = ans_norm.replace("region", "")
            ans_norm = ans_norm.strip()
            pred_norm = pred_norm.strip()
            return pred_norm == ans_norm
        elif pred_norm == ans_norm:
            return True
        else:
            return False
    elif category == "hard_pixel":
        return abs(float(pred_norm) - float(ans_norm)) <= 0.05
    else:
        return pred_norm == ans_norm

def normalize_answer(s):
    s = s.lower()
    parts = [part.strip() for part in s.split(',')]
    parts = [p for p in parts if p]
    return ', '.join(parts)
def get_major_category(category):
    if category in ['height_average', 'highest_region']:
        return 'height_inference'
    elif category in ['land_use', 'landcover_type']:
        return 'land_use_landcover_type'
    elif category in ['sky_visibility', 'urban_density', 'visibility_range']:
        return 'multi_inference'
    else:
        return 'svf_inference'
def calculate_category_accuracy_from_files(
    ans_file,
    pred_file,
    output_dir=None,
    show_details=True,
    include_observation=True,
    return_breakdown=False,
):
    """
    Calculate category-wise accuracy by comparing answer and prediction files
    
    :param ans_file: Path to answer file (contains category field)
    :param pred_file: Path to prediction file
    :param output_dir: Output directory path (None to skip output)
    :param show_details: Whether to show detailed results
    :return: Overall accuracy (percentage)
    """
    print(f"\n===== Category-wise Accuracy Calculation =====")
    print(f"Answer file: {ans_file}")
    print(f"Prediction file: {pred_file}")
    
    answers = load_answer_file(ans_file)
    predictions = load_answer_file(pred_file)
    
    if not answers:
        print("Could not load answer file")
        return 0.0
    
    if not predictions:
        print("Could not load prediction file")
        return 0.0
    ans_has_question_id = any('question_id' in data for data in answers.values())
    pred_has_question_id = any('question_id' in data for data in predictions.values())
    category_results = {}
    major_category_results = {}
    category_accuracy = {}
    major_category_accuracy = {}
    category_counts = {}
    category_prompt_type_accuracy = {}
    category_format_errors = {}
    correct = 0
    total = 0
    answered = 0
    non_answer_count = 0  
    format_error_count = 0  
    detailed_results = []
    landcover_type_jaccard_sum = 0.0
    landcover_type_jaccard_count = 0
    landcover_type_jaccard_correct = 0
    landcover_type_jaccard_threshold = 0.7
    def results_to_detailed_results(ans_data, pred_data):
        nonlocal correct, answered, category_accuracy, category_prompt_type_accuracy, category_counts, category_results, major_category_results, major_category_accuracy
        category = ans_data.get('category', 'unknown')
        question_text = ans_data.get('text', '')
        prompt_type = extract_prompt_type(ans_data.get('text', ''))
        if category not in category_results:
            category_results[category] = []
            category_accuracy[category] = [0, 0]
            category_counts[category] = 0
            category_prompt_type_accuracy[category] = {}
            category_format_errors[category] = 0
        major_category = get_major_category(category)
        if major_category not in major_category_results:
            major_category_results[major_category] = []
            major_category_accuracy[major_category] = [0, 0]
            
        if prompt_type not in category_prompt_type_accuracy[category]:
            category_prompt_type_accuracy[category][prompt_type] = [0, 0]
        
        correct_answer = ""
        for field in ['answer', 'label', 'text']:
            if field in ans_data and ans_data[field]:
                correct_answer = str(ans_data[field]).strip()
                break
        pred_answer = ""
        for field in ['predict', 'answer', 'response', 'text']:
            if field in pred_data and pred_data[field]:
                pred_answer = str(pred_data[field]).strip()
                break
        # Skip <OBSERVATION> answers when requested
        if not include_observation and correct_answer.startswith("<OBSERVATION>"):
            return None
        answered += 1

        category = ans_data.get('category', 'unknown')
        question_text = ans_data.get('text', '')
        prompt_type = extract_prompt_type(question_text)
        ans_norm = normalize_answer(correct_answer)
        pred_norm = normalize_answer(pred_answer)
        try:
            if len(pred_norm) <= 0 or len(pred_norm) - len(ans_norm) > 40:
                raise ValueError("pred_norm is too long")
            is_correct = judge_correct(pred_norm, ans_norm, category, question_text, correct_answer, pred_answer)
            format_error = False
        except (ValueError, TypeError) as e:
            is_correct = False
            format_error = True
            category_format_errors[category] += 1
            nonlocal format_error_count
            format_error_count += 1
        if is_correct:
            correct += 1
            category_accuracy[category][0] += 1
            category_prompt_type_accuracy[category][prompt_type][0] += 1
            major_category_accuracy[major_category][0] += 1
            
        category_accuracy[category][1] += 1
        category_prompt_type_accuracy[category][prompt_type][1] += 1
        category_counts[category] += 1
        major_category_accuracy[major_category][1] += 1
        
        detailed_results_item = ({
            "question_id": f"行{question_id}" if ans_has_question_id else f"行{i+1}",
            "prediction": pred_answer,
            "extracted_prediction": pred_norm,
            "correct_answer": correct_answer,
            "correct": is_correct,
            "note": "format error" if format_error else None
        })
        category_results[category].append(detailed_results_item)
        major_category_results[major_category].append(detailed_results_item)
        detailed_results.append(detailed_results_item)
        return detailed_results_item
    if not ans_has_question_id or not pred_has_question_id:
        print("no question_id in answer or prediction file, so we assume the order of the questions is the same")
        ans_list = []
        pred_list = []
        with open(ans_file, "r", encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    if line.strip():
                        data = json.loads(line)
                        data['question_id'] = line_num
                        ans_list.append(data)
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error (line {line_num}): {e}")
                    continue
        with open(pred_file, "r", encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    if line.strip():
                        data = json.loads(line)
                        data['question_id'] = line_num
                        pred_list.append(data)
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error (line {line_num}): {e}")
                    continue
        min_length = min(len(ans_list), len(pred_list))
        print(f"correct answer file: {len(ans_list)} questions, prediction file: {len(pred_list)} questions")
        print(f"compared questions: {min_length} questions (from top to bottom)")
        correct = 0
        total = min_length
        answered = 0
        detailed_results = []
        
        for i in range(min_length):
            ans_data = ans_list[i]
            pred_data = pred_list[i]
            results_to_detailed_results(ans_data, pred_data)
        
    else:
        common_ids = set(answers.keys()) & set(predictions.keys())
        if not common_ids:
            print("No common question_id found")
            return 0.0
        print(f"Number of questions to compare: {len(common_ids)}")
        correct = 0
        total = len(common_ids)
        answered = 0
        detailed_results = []
        def custom_sort_key(qid):
            if isinstance(qid, int):
                return (0, qid)
            else:
                return (1, str(qid))
        for question_id in sorted(common_ids, key=custom_sort_key):
            ans_data = answers[question_id]
            pred_data = predictions[question_id]
            results_to_detailed_results(ans_data, pred_data)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(output_dir, f"answer_comparison_result_{timestamp}.txt")
        with open(result_file, "w", encoding='utf-8') as f:
            f.write(f"Answer file: {ans_file}\n")
            f.write(f"Prediction file: {pred_file}\n")
            f.write(f"Calculation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Overall accuracy: {correct}/{total} ({(correct/total*100):.2f}%)\n")
            f.write(f"Accuracy for answered questions: {correct}/{answered} ({(correct/answered*100):.2f}% if answered > 0 else 0%)\n")
            f.write(f"Unanswered / Do not know: {total - answered}\n\n")
            f.write(f"Format errors: {format_error_count}/{total} ({(format_error_count/total*100):.2f}% if total > 0 else 0%)\n")
            
            if show_details:
                for category, (correct_count, total_count) in sorted(category_accuracy.items()):
                    if total_count > 0:
                        random_baseline = calculate_random_baseline(category, total_count)
                        f.write(f"  {category}: {correct_count}/{total_count} ({(correct_count/total_count*100):.2f}%) [Random: {random_baseline:.2f}%]\n")
                        print(f"  {category}: {correct_count}/{total_count} ({(correct_count/total_count*100):.2f}%) [Random: {random_baseline:.2f}%]")
                
                f.write("Major category: ")
                print("Major category: ")
                for major_category, results in major_category_results.items():
                    f.write(f"  {major_category}: {major_category_accuracy[major_category][0]}/{major_category_accuracy[major_category][1]} ({(major_category_accuracy[major_category][0]/major_category_accuracy[major_category][1]*100):.2f}%)\n")
                    print(f"  {major_category}: {major_category_accuracy[major_category][0]}/{major_category_accuracy[major_category][1]} ({(major_category_accuracy[major_category][0]/major_category_accuracy[major_category][1]*100):.2f}%)")
                f.write("Format errors: ")
                print(f"Format errors: {format_error_count}/{total} ({(format_error_count/total*100):.2f}% if total > 0 else 0%)")
                print("Format errors: ")
                
                for category, (correct_count, total_count) in sorted(category_accuracy.items()):
                    if category_format_errors[category] > 0:
                        f.write(f"  {category}: {category_format_errors[category]} ({(category_format_errors[category]/total_count*100):.2f}%)\n")
                        print(f"  {category}: {category_format_errors[category]} ({(category_format_errors[category]/total_count*100):.2f}%)")
                f.write("Detailed results:\n")
                f.write("-" * 80 + "\n")
                for result in detailed_results:
                    f.write(f"Question ID: {result['question_id']}\n")
                    f.write(f"Prediction (before extraction): {result['prediction'][:200]}{'...' if len(result['prediction']) > 200 else ''}\n")
                    f.write(f"Extracted prediction: {result['extracted_prediction']}\n")
                    f.write(f"Correct answer: {result['correct_answer']}\n")
                    f.write(f"Result: {'✓ Correct' if result['correct'] else '✗ Incorrect'}\n")
                    if result['note']:
                        f.write(f"Note: {result['note']}\n")
                    f.write("-" * 80 + "\n")
        print(f"Saved detailed results to file: {result_file}")
        for category, results in category_results.items():
            if not results:
                continue

            category_output_file = os.path.join(output_dir, f"{category}_results.txt")
            with open(category_output_file, "w", encoding="utf-8") as f:
                f.write(f"Category: {category}\n")
                f.write(f"Accuracy: {category_accuracy[category][0]}/{category_accuracy[category][1]} ({(category_accuracy[category][0]/category_accuracy[category][1]*100):.2f}%)\n")
                f.write(f"Number of answers: {category_counts[category]}\n")
                f.write(f"Format errors: {category_format_errors[category]}\n")
                f.write(f"Accuracy by prompt type:\n")
                for prompt_type, accuracy in category_prompt_type_accuracy[category].items():
                    f.write(f"  {prompt_type}: {accuracy[0]}/{accuracy[1]} ({(accuracy[0]/accuracy[1]*100):.2f}%)\n")
                f.write("\n")
                for result in results:
                    f.write(f"Question ID: {result['question_id']}\n")
                    f.write(f"Prediction (before extraction): {result['prediction'][:200]}{'...' if len(result['prediction']) > 200 else ''}\n")
                    f.write(f"Extracted prediction: {result['extracted_prediction']}\n")
                    f.write(f"Correct answer: {result['correct_answer']}\n")
                    f.write(f"Result: {'✓ Correct' if result['correct'] else '✗ Incorrect'}\n")
                    if result['note']:
                        f.write(f"Note: {result['note']}\n")
                    f.write("-" * 80 + "\n")
    print(f"Total questions: {total}")
    print(f"Answered questions: {answered}")
    print(f"Unanswered / Do not know: {total - answered}")
    print(f"Overall accuracy: {correct}/{total} ({(correct/total*100):.2f}%)")
    if answered > 0:
        print(f"Accuracy for answered questions: {correct}/{answered} ({(correct/answered*100):.2f}%)")
    
    if show_details:
        print(f"\nCorrect: {correct}")
        print(f"Incorrect: {answered - correct}")
        
        incorrect_samples = [r for r in detailed_results if not r['correct'] and not r['note']]
        if incorrect_samples:
            print(f"\nIncorrect samples (first 5):")
            for i, result in enumerate(incorrect_samples[:5], 1):
                print(f"{i}. {result['question_id']}: Prediction='{result['extracted_prediction']}' vs Correct='{result['correct_answer']}'")
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    if return_breakdown:
        # Convert major category counts to percentages
        major_pct = {}
        for k, (c_cnt, t_cnt) in major_category_accuracy.items():
            major_pct[k] = (c_cnt / t_cnt * 100) if t_cnt > 0 else 0.0
        # Convert fine category counts to percentages
        sub_pct = {}
        for k, (c_cnt, t_cnt) in category_accuracy.items():
            sub_pct[k] = (c_cnt / t_cnt * 100) if t_cnt > 0 else 0.0
        return accuracy, major_pct, sub_pct
    return accuracy


def _extract_timestamp_from_name(filename: str) -> str:
    m = re.search(r"(20\d{6}_\d{6})", filename)
    return m.group(1) if m else ""


def _infer_modality_from_name(filename: str) -> str:
    from pathlib import Path as _Path
    stem = _Path(filename).stem
    parts = stem.split("_")
    if parts:
        tail = parts[-1].lower()
        if tail in {"all", "svf", "dsm", "rgb"}:
            return tail
    return ""


def aggregate_predictions_to_csv(
    ans_file: str,
    pred_dir: str,
    csv_path: str,
    name_contains: str = "results",
    recursive: bool = True,
    show_details: bool = False,
    output_dir: str = None,
    max_files: int = 0,
):
    files = list_jsonl_files(pred_dir, name_contains=name_contains, recursive=recursive)
    if not files:
        print(f"No prediction files found under: {pred_dir} (filter='{name_contains}')")
        return

    # Optionally keep only most-recent N files by mtime
    if max_files and max_files > 0:
        files = sorted(files, key=lambda p: os.stat(p).st_mtime, reverse=True)[:max_files]

    fieldnames = build_csv_fieldnames_with_subcategories()

    rows = []
    for pred_file in files:
        try:
            acc, major_pct, sub_pct = calculate_category_accuracy_from_files(
                ans_file,
                pred_file,
                output_dir=output_dir,
                show_details=show_details,
                include_observation=False,
                return_breakdown=True,
            )
        except Exception as e:
            print(f"Failed to evaluate {pred_file}: {e}")
            continue
        ts = _extract_timestamp_from_name(os.path.basename(pred_file))
        modality = _infer_modality_from_name(os.path.basename(pred_file))
        base_row = {
            "timestamp": ts,
            "modality": modality,
            "pred_file": os.path.basename(pred_file),
            "pred_dir": str(__import__('pathlib').Path(pred_file).parent),
            "accuracy": f"{acc:.2f}",
        }
        # Fill major and subcategory columns in defined order
        major_to_sub = get_major_to_subcategories()
        for major, subs in major_to_sub.items():
            base_row[f"acc_{major}"] = f"{major_pct.get(major, 0.0):.2f}"
            for sub in subs:
                base_row[f"acc_{sub}"] = f"{sub_pct.get(sub, 0.0):.2f}"
        rows.append(base_row)

    if rows:
        append_rows_csv(csv_path, fieldnames=fieldnames, rows=rows)
        print(f"Appended {len(rows)} rows to CSV: {csv_path}")
    else:
        print("No rows to append.")

def main():
    parser = argparse.ArgumentParser(description="Calculate accuracy from answer and prediction files")
    parser.add_argument("--ans_file", help="Path to the answer file")
    parser.add_argument("--pred_file", help="Path to the prediction file")
    parser.add_argument("--output_dir", help="Output directory for results")
    parser.add_argument("--category_analysis", action="store_true", help="Show category details")
    # aggregation options
    parser.add_argument("--aggregate", action="store_true", help="Aggregate all prediction files under a directory into a CSV")
    parser.add_argument("--pred_dir", type=str, default="", help="Directory containing prediction files (.json/.jsonl)")
    parser.add_argument("--csv_path", type=str, default="", help="Target CSV path. Defaults to <pred_dir>/aggregated_accuracy.csv")
    parser.add_argument("--name_contains", type=str, default="results", help="Filter prediction files by substring")
    parser.add_argument("--no_recursive", action="store_true", help="Disable recursive search in pred_dir")
    parser.add_argument("--max_files", type=int, default=0, help="If >0, only evaluate N most-recent files")
    
    args = parser.parse_args()
    config_file = "inference_code/gpt4o_svf_qa_multi.py"
    if not args.ans_file:
        for line in open(config_file).readlines():
            if "parser.add_argument('--questions_file', type=str, default=" in line:
                args.ans_file = line.split("default=")[1].split(",")[0].strip().strip('"')
                break
    
    # aggregation mode
    if args.aggregate:
        if not args.ans_file:
            print("ans_file is required for aggregation")
            return
        pred_dir = args.pred_dir
        if not pred_dir:
            if args.pred_file:
                from pathlib import Path as _Path
                pred_dir = str(_Path(args.pred_file).parent)
            else:
                print("pred_dir or pred_file is required for aggregation")
                return
        csv_path = args.csv_path or str(__import__('pathlib').Path(pred_dir) / "aggregated_accuracy.csv")
        recursive = not args.no_recursive
        print(
            f"Aggregate mode enabled\n  ans_file: {args.ans_file}\n  pred_dir: {pred_dir}\n  csv_path: {csv_path}\n  filter: {args.name_contains}\n  recursive: {recursive}\n  max_files: {args.max_files}\n  category_analysis: {args.category_analysis}\n  output_dir: {args.output_dir}"
        )
        aggregate_predictions_to_csv(
            args.ans_file,
            pred_dir,
            csv_path,
            name_contains=args.name_contains,
            recursive=recursive,
            show_details=args.category_analysis,
            output_dir=args.output_dir,
            max_files=args.max_files,
        )
        return

    print(f"ans_file: {args.ans_file}")
    print(f"pred_file: {args.pred_file}")
    print(f"output_dir: {args.output_dir}")
    print(f"category_analysis: {args.category_analysis}")
    calculate_category_accuracy_from_files(args.ans_file, args.pred_file, args.output_dir, args.category_analysis)
    
if __name__ == "__main__":
    main()