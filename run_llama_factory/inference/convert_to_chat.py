import json
import os
import random
import re
from typing import List, Dict, Any
import argparse
import random

def extract_region_coordinates(question_text: str) -> Dict[str, str]:
    """
    質問テキストから各リージョンの座標を抽出
    
    Args:
        question_text: 質問文
        
    Returns:
        Dict[str, str]: リージョン名と座標のマッピング（例: {"Region A": "[24%, 8%, 48%, 32%]"}）
    """
    coordinates_map = {}
    
    pattern1 = re.compile(r'Region ([A-Z]):\s*(\[[\d%,\s]+\])')
    matches = pattern1.findall(question_text)
    for region, coords in matches:
        coordinates_map[f"Region {region}"] = coords
    
    pattern2 = re.compile(r'([A-Z]):\s*\[xmin=([\d%]+),\s*ymin=([\d%]+),\s*xmax=([\d%]+),\s*ymax=([\d%]+)\]')
    matches = pattern2.findall(question_text)
    for region, xmin, ymin, xmax, ymax in matches:
        coordinates_map[f"Region {region}"] = f"[{xmin}, {ymin}, {xmax}, {ymax}]"
    
    return coordinates_map

def convert_answer_to_coordinates(answer: str, question_text: str) -> str:
    coordinates_map = extract_region_coordinates(question_text)
    
    if answer in coordinates_map:
        return f"{answer}: {coordinates_map[answer]}"
    
    return answer

def convert_to_chat_format(input_file: str, output_file: str, ft_mode: bool = False) -> None:
    """
    Convert JSONL file to chat format where each entry has:
    {
        "messages": [
            {
                "content": "<image>\n{question}",
                "role": "user"
            },
            {
                "content": "{answer}",
                "role": "assistant"
            }
        ],
        "images": [
            "{image_path}"
        ]
    }
    For 30% of the data, the format will include category information in the prompt and answer.
    For specific categories with high-performance prompts, rephrase to best-performing prompt.
    
    Args:
        input_file: 入力JSONLファイルパス
        output_file: 出力JSONファイルパス
        ft_mode: FTモードを有効にするかどうか
    """
    raw_data = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    question = json.loads(line)
                    if not isinstance(question, dict):
                        print(f"Warning: Line {line_num} is not a valid JSON object")
                        continue
                    
                    if 'image' not in question or 'text' not in question or 'answer' not in question or 'category' not in question:
                        print(f"Warning: Line {line_num} is missing required fields (image, text, answer, category)")
                        continue

                    raw_data.append(question)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON at line {line_num}: {e}")
                    continue
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    if not raw_data:
        print("Warning: No valid data was processed")
        return

    random.shuffle(raw_data)
    chat_data = []
    split_index = int(len(raw_data) * 0.15)
    split_index2 = int(len(raw_data) * 0.85)

    for i, question in enumerate(raw_data):
        image_path = question['image']
        category = question['category']
        original_text = question['text']
        original_answer = question['answer']
        
        if ft_mode:
            if random.random() < 0.7:
                converted_answer = convert_answer_to_coordinates(original_answer, original_text)
                if "Region" in converted_answer and converted_answer != original_answer:
                    question['text'] = original_text + "\nPlease provide the answer in the format of Region X: [coordinates]"
            else:
                converted_answer = original_answer
                question['text'] = original_text
        else:
            converted_answer = original_answer
        
        if i < split_index:
            user_content = f"<image>\n{question['text']}\nPlease first state the category in the format <category> </category> and then provide only the short answer to the question, without explanation."
            assistant_content = f"<category>{category}</category> {converted_answer}"
        elif i > split_index2:
            user_content = f"<image>\n{question['text']}\nPlease first state the category in the format <category> </category> and then provide only the short answer to the question, without explanation."
            assistant_content = f"<category>{category}</category> {converted_answer}"
        else:
            user_content = f"<image>\n{question['text']}\nPlease provide only the short answer to the question, without explanation."
            assistant_content = converted_answer

        chat_entry = {
            "messages": [
                {
                    "content": user_content,
                    "role": "user"
                },
                {
                    "content": assistant_content,
                    "role": "assistant"
                }
            ],
            "images": [
                f"/workspace/GeoNRW/{image_path}"
            ]
        }
        chat_data.append(chat_entry)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)
        print(f"Successfully processed {len(chat_data)} entries")
    except Exception as e:
        print(f"Error writing output file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert SVF QA data to chat format")
    parser.add_argument("--input_file", type=str, default="svf_qo_re0625/svf_15x_medium_answers_train_mixed_hr0.3_0627_hr0.3.jsonl",
                        help="Input JSONL file path")
    parser.add_argument("--output_file", type=str,
                        help="Output JSON file path (default: input_file with _factory.json suffix)")
    parser.add_argument("--ft-mode", action="store_true",
                        help="Enable FT mode: Convert Region X answers to Region X: [coordinates] format")
    
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file if args.output_file else input_file.replace(".jsonl", "_factory.json")
    
    if args.ft_mode and args.output_file is None:
        output_file = input_file.replace(".jsonl", "_ft_factory.json")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    convert_to_chat_format(input_file, output_file, ft_mode=args.ft_mode)
    print(f"Conversion completed. Output written to {output_file}")

if __name__ == "__main__":
    main() 