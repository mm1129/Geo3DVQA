# -*- coding: utf-8 -*-
import json
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from datetime import datetime
import argparse
import os


# argparseで引数を受け取る
parser = argparse.ArgumentParser(description="LLaVA-OneVision Inference Script (Transformers)")

# モデルと入出力
parser.add_argument(
    '--base_model_path',
    type=str,
    default="llava-hf/llava-onevision-qwen2-7b-ov-hf",
    help='Base model path (e.g., llava-hf/llava-onevision-qwen2-7b-ov-hf)'
)
parser.add_argument(
    '--input_file',
    type=str,
    default="data/svf_15x_large_answers_test_mixed_hr0.3_0809_hr0.3.jsonl",
    help='Input JSONL file'
)
parser.add_argument(
    '--media_root',
    type=str,
    default="/workspace/GeoNRW",
    help='Root directory for images referenced in the input JSONL'
)
parser.add_argument(
    '--max_new_tokens',
    type=int,
    default=256,
    help='Maximum number of new tokens to generate'
)

args = parser.parse_args()

if 'free' in args.input_file:
    is_free = True
else:
    is_free = False

# プロセッサ・モデルのロード
processor = AutoProcessor.from_pretrained(
    args.base_model_path,
    trust_remote_code=True
)
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    args.base_model_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16
).eval()


# 出力ファイル名を事前に決定（tmpファイルは使わない）
today = datetime.now().strftime("%Y%m%d")
output_file = (
    args.input_file.split("/")[-1].split(".")[0]
    + f"_{args.base_model_path.split('/')[-1]}_{today}_results.jsonl"
)

# 保存先ディレクトリを作成
result_file = f"data/{output_file}"
os.makedirs(os.path.dirname(result_file), exist_ok=True)

if os.path.exists(result_file):
    base, ext = os.path.splitext(result_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_result_file = f"{base}_new_{timestamp}{ext}"
    print(f"既存の結果ファイルが存在するため、新しいファイル名に変更します: {new_result_file}")
    result_file = new_result_file
print(f"出力先ファイル: {result_file}")


# テストデータの読み込み
with open(args.input_file, "r", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f if line.strip()]


for idx, item in enumerate(tqdm(test_data)):
    # 画像の読み込み
    image_path = os.path.join(args.media_root, item['image'])
    image = Image.open(image_path).convert("RGB")

    # プロンプト生成（メッセージ形式→テンプレート適用）
    # LLaVA-OneVisionはHFのチャットテンプレートに対応
    if is_free:
        prompt_text = item.get("text", "") + "\nPlease structure your response using only <OBSERVATION> and <CONCLUSION>:\n<OBSERVATION>Statistical data and image-grounded observations only (SVF values, elevation in meters, land cover %).</OBSERVATION>\n<CONCLUSION>Base the decision strictly on <OBSERVATION> and answer consisely to the question; please mention at least one grid for the answer.</CONCLUSION>."
    else:
        if not is_free:
        prompt_text = item.get("text", "") + "\nPlease provide only your final short answer to the question concisely."
  
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    text_input = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    # 入力の前処理
    inputs = processor(
        text=text_input,
        images=image,
        return_tensors="pt"
    )
    # toデバイス/精度
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device, dtype=torch.float16) if hasattr(v, 'dtype') and torch.is_floating_point(v) else v.to(model.device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 推論
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None
        )

    # 出力デコード（入力長をスキップして生成分のみ）
    input_len = inputs["input_ids"].shape[1]
    pred = processor.decode(output_ids[0, input_len:], skip_special_tokens=True).strip()

    # 結果を保存
    result = {
        "question_id": item["question_id"] if "question_id" in item else None,
        "prompt": item.get("text", ""),
        "predict": pred,
        "label": item.get("output", "")
    }
    with open(result_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


print(f"推論完了: {len(test_data)}件の結果を{result_file}に保存しました")


