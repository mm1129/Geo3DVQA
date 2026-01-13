import json
import yaml
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel
from datetime import datetime
import argparse
import os

def load_config_from_yaml(yaml_path):
    """YAMLファイルから設定を読み込む"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def extract_inference_config(training_config):
    """学習用YAMLから推論用設定を抽出"""
    inference_config = {
        'base_model_path': training_config.get('model_name_or_path', 'Qwen/Qwen2.5-VL-7B-Instruct'),
        'lora_model_path': training_config.get('output_dir', 'data/saves/default/lora'),
        'image_max_pixels': training_config.get('image_max_pixels', 262144),
        'video_max_pixels': training_config.get('video_max_pixels', 16384),
        'cutoff_len': training_config.get('cutoff_len', 1024),
        'template': training_config.get('template', 'qwen2_vl'),
        'media_dir': training_config.get('media_dir', '/workspace/GeoNRW')
    }
    return inference_config

parser = argparse.ArgumentParser(description="Qwen2.5-VL-7B-Instruct Inference Script")

parser.add_argument('--config', type=str, default="configs/0728_qwen2_5_vl.yaml", help='YAML config file path')

parser.add_argument('--base_model_path', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help='Base model path')
parser.add_argument('--input_file', type=str, default="data/svf_15x_large_answers_test_mixed_hr0.3_0809_hr0.3.jsonl", help='Input JSONL file')
group = parser.add_mutually_exclusive_group(required=False)
group.add_argument('--use_lora', dest='use_lora', action='store_true', help='Use LoRA')
group.add_argument('--no_use_lora', dest='use_lora', action='store_false', help='Do not use LoRA')
parser.set_defaults(use_lora=True)

args = parser.parse_args()

if "free" in args.input_file:
    is_free = True
    print("Free-form questions")
else:
    is_free = False

if args.config and args.use_lora:
    print(f"Loading config from: {args.config}")
    yaml_config = load_config_from_yaml(args.config)
    inference_config = extract_inference_config(yaml_config)
    
    for key, value in inference_config.items():
        if not hasattr(args, key) or getattr(args, key) == parser.get_default(key):
            setattr(args, key, value)
    
    print(f" Configuration loaded:")
    print(f"  - Base model: {args.base_model_path}")
    print(f"  - LoRA path: {args.lora_model_path}")
    print(f"  - Image max pixels: {args.image_max_pixels}")

tokenizer = AutoTokenizer.from_pretrained(
    args.base_model_path,
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    args.base_model_path,
    trust_remote_code=True
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.base_model_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype="auto"
)
if args.use_lora:
    model = PeftModel.from_pretrained(model, args.lora_model_path)
    model = model.to("cuda").eval()
    if args.lora_model_path.endswith('/lora/sft'):
        lora_prefix = args.lora_model_path.split('/')[-3]
    elif args.lora_model_path.endswith('/lora'):
        lora_prefix = args.lora_model_path.split('/')[-2]
    else:
        lora_prefix = args.lora_model_path.split('/')[-1]
    print(f"LoRA prefix: {lora_prefix}")
else:
    print(f"Loading base model: {args.base_model_path}")
    model = model.eval()
from datetime import datetime

today = datetime.now().strftime("%Y%m%d")
use_lora = "lora" if args.use_lora else "base"
lora_prefix = args.lora_prefix if hasattr(args, 'lora_prefix') and args.lora_prefix else ""
lora_prefix = args.config.split("/")[-1].split("_")[0] if args.config else lora_prefix
output_file = args.input_file.split("/")[-1].split(".")[0] + f"_{today}_{use_lora}_{lora_prefix[:10]}_results.jsonl"

result_file = f"data/{output_file}"
if os.path.exists(result_file):
    base, ext = os.path.splitext(result_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_result_file = f"{base}_new_{timestamp}{ext}"
    print(f"既存の結果ファイルが存在するため、新しいファイル名に変更します: {new_result_file}")
    result_file = new_result_file
print(f"出力先ファイル: {result_file}") 
with open(args.input_file, "r", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f if line.strip()]

for idx, item in enumerate(tqdm(test_data)):
    
    image_path = f"/workspace/GeoNRW/{item['image']}"
    image = Image.open(image_path).convert("RGB")
    if is_free:
        item_text = item["text"] + "\nPlease structure your response using only <OBSERVATION> and <CONCLUSION>:\n<OBSERVATION>Statistical data and image-grounded observations only (SVF values, elevation in meters, land cover %).</OBSERVATION>\n<CONCLUSION>Base the decision strictly on <OBSERVATION> and answer consisely to the question; please mention at least one grid for the answer.</CONCLUSION>."
    else:
        item_text = item["text"] + "\nPlease provide only your final short answer to the question concisely."   
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user", 
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": item_text}
            ]
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None
        )
    input_len = inputs["input_ids"].shape[1]
    pred = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)

    result = {
        "question_id": item["question_id"] if "question_id" in item else None,
        "prompt": text,
        "predict": pred,
        "label": item.get("output", "")
    }
    with open(result_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

print(f"推論完了: {len(test_data)}件の結果を{result_file}に保存しました")