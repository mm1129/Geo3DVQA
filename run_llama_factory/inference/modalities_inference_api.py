import json
import yaml
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel
from datetime import datetime
import argparse
import os
import glob
import hashlib

# Multimodal helpers
from typing import List
from modality_utils.path_resolver import get_related_image_paths
from modality_utils.messaging import prepare_multimodal_message

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

# Resolve LoRA path to a concrete checkpoint if top-level has no adapter files
def resolve_lora_path(path: str) -> str:
    """Return a directory containing adapter_config.json (prefer latest checkpoint)."""
    cfg_path = os.path.join(path, "adapter_config.json")
    if os.path.exists(cfg_path):
        return path
    candidates = glob.glob(os.path.join(path, "checkpoint-*", "adapter_config.json"))
    best_dir = None
    best_step = -1
    best_mtime = -1.0
    for cfg in candidates:
        ckpt_dir = os.path.dirname(cfg)
        base = os.path.basename(ckpt_dir)
        step = -1
        if base.startswith("checkpoint-"):
            try:
                step = int(base.split("-")[-1])
            except Exception:
                step = -1
        mtime = os.path.getmtime(cfg)
        if step > best_step or (step == best_step and mtime > best_mtime):
            best_step = step
            best_mtime = mtime
            best_dir = ckpt_dir
    if best_dir:
        # Prefer directories that also contain adapter weights
        for fname in ("adapter_model.safetensors", "adapter_model.bin"):
            if os.path.exists(os.path.join(best_dir, fname)):
                return best_dir
        return best_dir
    raise ValueError(f"adapter_config.json not found under {path}. Set --lora_model_path to a checkpoint containing adapter files.")

parser = argparse.ArgumentParser(description="Qwen2.5-VL-7B-Instruct Inference Script (multimodal)")

parser.add_argument('--config', type=str, default="configs/0728_qwen2_5_vl.yaml", help='YAML config file path')

parser.add_argument('--base_model_path', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help='Base model path')
parser.add_argument('--input_file', type=str, default="data/svf_combined_3000q_free.jsonl", help='Input JSONL file')
parser.add_argument('--output_file', type=str, default=None, help='Explicit result JSONL path (optional)')
parser.add_argument('--limit', type=int, default=-1, help='Max questions to process (-1 for all)')
parser.add_argument('--verbose', action='store_true', help='Enable verbose logs')
parser.add_argument('--add_modality_note', action='store_true', help='Append modality note to the question text')

parser.add_argument('--image_dir', type=str, default="/workspace/GeoNRW/", help='Base directory for RGB images')
parser.add_argument('--dsm_dir', type=str, default="/workspace/GeoNRW/", help='Base directory for DSM GeoTIFFs')
parser.add_argument('--svf_dir', type=str, default="svf/skyview_umep_test/", help='Base directory for SVF GeoTIFFs')
parser.add_argument('--seg_dir', type=str, default="/workspace/GeoNRW/", help='Base directory for segmentation GeoTIFFs')
parser.add_argument('--use_all_modalities', action='store_true', help='Use all modalities (rgb,dsm,svf,seg)')
parser.add_argument('--modalities', type=str, default='rgb', help='Comma-separated modalities (e.g., rgb,dsm,svf)')
parser.add_argument('--dsm_colormap', type=str, default='terrain', help='Colormap for DSM (e.g., terrain, jet, viridis)')
parser.add_argument('--dsm_show_colorbar', default=True, action='store_true', help='Show DSM colorbar with min/max in meters')
parser.add_argument('--dsm_vmin', type=float, default=None, help='Explicit DSM min (meters) for color mapping')
parser.add_argument('--dsm_vmax', type=float, default=None, help='Explicit DSM max (meters) for color mapping')
parser.add_argument('--svf_colormap', type=str, default='plasma', help='Colormap for SVF (e.g., plasma, viridis, inferno)')
parser.add_argument('--svf_show_colorbar', action='store_true', help='Show SVF colorbar with min/max')
parser.add_argument('--svf_vmin', type=float, default=None, help='Explicit SVF min (default 0.0)')
parser.add_argument('--svf_vmax', type=float, default=None, help='Explicit SVF max (default 1.0)')
parser.add_argument('--preview_dir', type=str, default="preview_convert", help='If set, export preview of first K items (images + prompt)')
parser.add_argument('--preview_count', type=int, default=5, help='Number of preview items to export')
parser.add_argument('--single_modality_test', action='store_true', help='Test each modality or combination separately')
parser.add_argument('--modality_combinations', type=str, default=None, help='Comma-separated combos (e.g., rgb,dsm,rgb+dsm,all)')
group = parser.add_mutually_exclusive_group(required=False)
group.add_argument('--use_lora', dest='use_lora', action='store_true', help='Use LoRA')
group.add_argument('--no_use_lora', dest='use_lora', action='store_false', help='Do not use LoRA')
parser.set_defaults(use_lora=True)

args = parser.parse_args()

if "free" in args.input_file and "combined" not in args.input_file:
    is_free = True
    print("Free-form questions")
else:
    is_free = False

if args.modality_combinations and not args.single_modality_test:
    args.single_modality_test = True
    print("single_modality_test を自動有効化: modality_combinations が指定されました")

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
    original_lora_path = args.lora_model_path
    resolved_lora_path = resolve_lora_path(original_lora_path)
    print(f"Using LoRA from: {resolved_lora_path}")
    model = PeftModel.from_pretrained(model, resolved_lora_path)
    model = model.to("cuda").eval()
    if original_lora_path.endswith('/lora/sft'):
        lora_prefix = original_lora_path.split('/')[-3]
    elif original_lora_path.endswith('/lora'):
        lora_prefix = original_lora_path.split('/')[-2]
    else:
        lora_prefix = original_lora_path.split('/')[-1]
    print(f"LoRA prefix: {lora_prefix}")
else:
    print(f"Loading base model: {args.base_model_path}")
    model = model.eval()
today = datetime.now().strftime("%Y%m%d")
use_lora = "lora" if args.use_lora else "base"
lora_prefix = args.lora_prefix if hasattr(args, 'lora_prefix') and args.lora_prefix else ""
lora_prefix = args.config.split("/")[-1].split("_")[0] if args.config else lora_prefix
base_input_name = args.input_file.split("/")[-1].split(".")[0]
if args.output_file:
    result_file = args.output_file
else:
    if not args.single_modality_test:
        if args.use_all_modalities:
            modality_info_for_name = "all"
        else:
            _mods = [m.strip() for m in (args.modalities or "rgb").split(',') if m.strip()]
            modality_info_for_name = "_".join(_mods) if _mods else "rgb"
        output_file = base_input_name + f"_{today}_{use_lora}_{lora_prefix[:10]}_{modality_info_for_name}_results.jsonl"
    else:
        output_file = base_input_name + f"_{today}_{use_lora}_{lora_prefix[:10]}_results.jsonl"
    result_file = f"data/{output_file}"
if os.path.exists(result_file):
    base, ext = os.path.splitext(result_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_result_file = f"{base}_new_{timestamp}{ext}"
    print(f"既存の結果ファイルが存在するため、新しいファイル名に変更します: {new_result_file}")
    result_file = new_result_file
print(f"出力先ファイル: {result_file}")

if not args.image_dir:
    args.image_dir = getattr(args, 'media_dir', None)

if args.use_lora and args.config:
    pass

if args.use_all_modalities:
    modalities: List[str] = ["rgb", "dsm", "svf", "seg"]
else:
    modalities = [m.strip() for m in (args.modalities or "rgb").split(',') if m.strip()]
print(f"使用するモダリティ: {modalities}")


def _validate_env_and_inputs(selected_modalities: List[str]):
    """Fail-fast checks for environment and directory readiness."""
    try:
        import rasterio  # noqa: F401
    except Exception as e:
        if any(m in selected_modalities for m in ["dsm", "svf", "seg"]):
            raise RuntimeError("rasterio is required for dsm/svf/seg conversions") from e
    try:
        import matplotlib  # noqa: F401
    except Exception as e:
        if any(m in selected_modalities for m in ["dsm", "svf"]):
            raise RuntimeError("matplotlib is required for dsm/svf conversions") from e

    if "dsm" in selected_modalities and not args.dsm_dir:
        raise ValueError("'dsm' modality requires --dsm_dir to be set")
    if "svf" in selected_modalities and not args.svf_dir:
        raise ValueError("'svf' modality requires --svf_dir to be set")

    if args.dsm_dir and "dsm" in selected_modalities:
        dsm_candidates = glob.glob(os.path.join(args.dsm_dir, "**", "*.tif"), recursive=True)
        if len(dsm_candidates) == 0:
            print(f"Warning: No DSM .tif files found under --dsm_dir: {args.dsm_dir}")
    if args.svf_dir and "svf" in selected_modalities:
        svf_candidates = glob.glob(os.path.join(args.svf_dir, "**", "*.tif"), recursive=True)
        if len(svf_candidates) == 0:
            print(f"Warning: No SVF .tif files found under --svf_dir: {args.svf_dir}")


_validate_env_and_inputs(modalities)

def run_inference_loop(test_data, selected_modalities, result_path):
    preview_emitted = 0
    def _assert_unique_and_exists(paths: dict, req_mods: List[str]):
        existing = []
        for m in req_mods:
            p = paths.get(m)
            if not p or not os.path.exists(p):
                raise FileNotFoundError(f"[{m}] path invalid: {p}")
            existing.append(p)
        if len(set(existing)) != len(existing):
            raise RuntimeError(f"Modalities map to same file: {existing}")

    def _sha256(path: str) -> str:
        try:
            with open(path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return ""

    if args.limit and args.limit > 0:
        data_iter = test_data[:args.limit]
    else:
        data_iter = test_data
    for idx, item in enumerate(tqdm(data_iter)):
        rel_image = item.get('image', '')
        if args.image_dir:
            rgb_path = os.path.join(args.image_dir, rel_image)
        else:
            rgb_path = f"/workspace/GeoNRW/{rel_image}"

        image_paths = get_related_image_paths(
            rgb_path,
            args.dsm_dir or args.image_dir,
            args.svf_dir or args.image_dir,
            args.seg_dir or args.image_dir,
        )

        req_mods = [m for m in selected_modalities if m in ["rgb", "dsm", "svf", "seg"]]
        present = [m for m in req_mods if image_paths.get(m)]
        if len(present) == 0:
            raise RuntimeError(f"No modality images resolved. rgb={image_paths.get('rgb')} dsm={image_paths.get('dsm')} svf={image_paths.get('svf')} seg={image_paths.get('seg')}")
        _assert_unique_and_exists(image_paths, present)
        if args.verbose:
            dsm_p, svf_p = image_paths.get('dsm'), image_paths.get('svf')
            print(f"[{idx}] paths: dsm={dsm_p}, svf={svf_p}")
            if dsm_p and svf_p:
                print(f"[{idx}] hashes: dsm={_sha256(dsm_p)[:8]}, svf={_sha256(svf_p)[:8]}")

        item_text = item.get("text", "")
        if args.add_modality_note:
            modal_note = f"[Modalities={','.join(selected_modalities)} | DSM={args.dsm_colormap} | SVF={args.svf_colormap}]"
            item_text = (item_text + "\n" + modal_note).strip()
        messages, used_modalities = prepare_multimodal_message(
            image_paths=image_paths,
            text=item_text,
            modalities=selected_modalities,
            dsm_colormap=args.dsm_colormap,
            svf_colormap=args.svf_colormap,
            is_free=is_free,
            dsm_show_colorbar=args.dsm_show_colorbar,
            dsm_vmin=args.dsm_vmin,
            dsm_vmax=args.dsm_vmax,
            svf_show_colorbar=args.svf_show_colorbar,
            svf_vmin=args.svf_vmin,
            svf_vmax=args.svf_vmax,
        )
        if not used_modalities:
            raise RuntimeError(f"No images attached to the message. Resolved: {image_paths}")
        if args.verbose:
            print(f"[{idx}] used_modalities={used_modalities}")

        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text_prompt],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens if hasattr(args, 'max_new_tokens') else 256,
            )
            if hasattr(args, 'temperature') and args.temperature and args.temperature > 0:
                gen_kwargs.update(dict(do_sample=True, temperature=args.temperature))
            else:
                gen_kwargs.update(dict(do_sample=False, temperature=None))
            output = model.generate(
                **inputs,
                **gen_kwargs,
            )
        input_len = inputs["input_ids"].shape[1]
        pred = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)

        result = {
            "question_id": item.get("question_id"),
            "prompt": text_prompt,
            "predict": pred,
            "label": item.get("output", ""),
            "used_modalities": used_modalities,
        }
        with open(result_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        try:
            if args.preview_dir and preview_emitted < max(0, int(args.preview_count or 0)):
                os.makedirs(args.preview_dir, exist_ok=True)
                idx_str = f"{preview_emitted:04d}"
                for m in used_modalities:
                    p = image_paths.get(m)
                    if not p or not os.path.exists(p):
                        continue
                    out_img = os.path.join(args.preview_dir, f"{idx_str}_{m}{os.path.splitext(p)[1]}")
                    try:
                        import shutil
                        shutil.copyfile(p, out_img)
                    except Exception:
                        pass
                try:
                    meta_json = {
                        "index": preview_emitted,
                        "modalities": used_modalities,
                        "image_paths": {m: image_paths.get(m) for m in used_modalities},
                        "prompt": text_prompt,
                    }
                    with open(os.path.join(args.preview_dir, f"{idx_str}_preview.json"), "w", encoding="utf-8") as f2:
                        f2.write(json.dumps(meta_json, ensure_ascii=False, indent=2))
                except Exception:
                    pass
                preview_emitted += 1
        except Exception:
            pass

    print(f"推論完了: {len(test_data)}件の結果を{result_path}に保存しました")

# Input data
with open(args.input_file, "r", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f if line.strip()]
if args.verbose:
    print(f"Loaded {len(test_data)} items from {args.input_file}")

if args.single_modality_test:
    all_modalities = ["rgb", "dsm", "svf", "seg"]
    if args.modality_combinations:
        combinations = [c.strip() for c in args.modality_combinations.split(',') if c.strip()]
    else:
        combinations = all_modalities + ["all"]

    for combo in combinations:
        if combo == "all":
            selected = all_modalities
            modality_info = "all"
        else:
            selected = [m.strip() for m in combo.replace('+', ',').split(',') if m.strip()]
            modality_info = "_".join(selected)

        base, ext = os.path.splitext(result_file)
        combo_result_file = f"{base}_{modality_info}{ext}"
        if os.path.exists(combo_result_file):
            base2, ext2 = os.path.splitext(combo_result_file)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            combo_result_file = f"{base2}_new_{timestamp}{ext2}"
        print(f"\n=== モダリティ '{modality_info}' で実行 ===")
        print(f"出力先ファイル: {combo_result_file}")
        run_inference_loop(test_data, selected, combo_result_file)
else:
    run_inference_loop(test_data, modalities, result_file)