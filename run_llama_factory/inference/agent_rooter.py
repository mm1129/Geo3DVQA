# rgb ---encoder--> {rgb,seg,svf,dsmのうち必要なもの}  ---Qwen FTed--> answer
# 1. rgb encode to {modality needed}
# ref: test_pretrained.py
# 2. select {modalities needed} -> add images -> inferenc
# import: llama-factory-0802/configs/modalities_inference_api.py
# ref: llama-factory-0802/configs/convert_to_multimodal_dataset.py
"""Agent router for modality-aware inference (per item).

This CLI:
- Reads JSONL input and infers required modalities for each item
- Resolves modality image paths and prepares a multimodal message per item
- Runs Qwen VL generation in-process and writes one merged JSONL
"""

from __future__ import annotations

import argparse
import json
import os
import glob
import hashlib
import yaml
from datetime import datetime
from typing import Dict, List

import torch
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel

from modality_utils.path_resolver import get_related_image_paths
from modality_utils.messaging import prepare_multimodal_message
from tqdm import tqdm
from modality_utils.image_converters import dsm_to_rgb, svf_to_rgb, seg_to_rgb


# Import heuristics from converter without triggering its main()
try:
    from .convert_to_multimodal_dataset import (
        _extract_first_question_sentence,
        _infer_modalities_from_question,
    )
except Exception:
    from convert_to_multimodal_dataset import (  # type: ignore
        _extract_first_question_sentence,
        _infer_modalities_from_question,
    )


CANONICAL_MODALITIES: List[str] = ["rgb", "svf", "dsm", "seg"]


def load_config_from_yaml(yaml_path: str):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def extract_inference_config(training_config):
    return {
        'base_model_path': training_config.get('model_name_or_path', 'Qwen/Qwen2.5-VL-7B-Instruct'),
        'lora_model_path': training_config.get('output_dir', 'data/saves/default/lora'),
        'image_max_pixels': training_config.get('image_max_pixels', 262144),
        'video_max_pixels': training_config.get('video_max_pixels', 16384),
        'cutoff_len': training_config.get('cutoff_len', 1024),
        'template': training_config.get('template', 'qwen2_vl'),
        'media_dir': training_config.get('media_dir', '/workspace/GeoNRW')
    }


def resolve_lora_path(path: str) -> str:
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
        for fname in ("adapter_model.safetensors", "adapter_model.bin"):
            if os.path.exists(os.path.join(best_dir, fname)):
                return best_dir
        return best_dir
    raise ValueError(f"adapter_config.json not found under {path}. Set --lora_model_path to a checkpoint containing adapter files.")


def _canonicalize_modalities(mods: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for m in CANONICAL_MODALITIES:
        if m in mods and m not in seen:
            ordered.append(m)
            seen.add(m)
    return ordered


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-item modality-aware inference")
    parser.add_argument("--input_file", default="data/svf_combined_3000q_free.jsonl", type=str, help="Input JSONL path")
    parser.add_argument("--output_file", required=False, type=str, help="Merged output JSONL path")
    parser.add_argument("--config", required=False, type=str, help="YAML config file path")
    parser.add_argument("--base_model_path", default="Qwen/Qwen2.5-VL-7B-Instruct", type=str)
    parser.add_argument("--image_dir", default="/workspace/GeoNRW", type=str)
    parser.add_argument("--dsm_dir", default="/workspace/GeoNRW", type=str)
    parser.add_argument("--svf_dir", default="svf/skyview_umep_test/", type=str)
    parser.add_argument("--seg_dir", default="/workspace/GeoNRW", type=str)
    parser.add_argument("--dsm_colormap", default="terrain", type=str)
    parser.add_argument("--svf_colormap", default="plasma", type=str)
    parser.add_argument("--dsm_show_colorbar", action="store_true", help="Show DSM colorbar in viz")
    parser.add_argument("--svf_show_colorbar", action="store_true", help="Show SVF colorbar in viz")
    parser.add_argument("--dsm_vmin", type=float, default=None)
    parser.add_argument("--dsm_vmax", type=float, default=None)
    parser.add_argument("--svf_vmin", type=float, default=None)
    parser.add_argument("--svf_vmax", type=float, default=None)
    parser.add_argument("--add_modality_note", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--infer_mode", choices=["override", "merge"], default="override")
    parser.add_argument("--use_lora", dest="use_lora", action="store_true")
    parser.add_argument("--no_use_lora", dest="use_lora", action="store_false")
    parser.set_defaults(use_lora=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=-1, help="Process first N items (-1 for all)")
    # Preview export (mirrors converter script style, minimal subset)
    parser.add_argument("--preview_dir", type=str, default=None, help="If set, export per-item preview images and prompt")
    parser.add_argument("--preview_count", type=int, default=5, help="Number of preview items to export")
    parser.add_argument("--preview_format", type=str, default="jpeg", help="Preview image format: png or jpeg")
    parser.add_argument("--preview_quality", type=int, default=90, help="JPEG quality for preview (if format=jpeg)")
    parser.add_argument("--preview_max_edge", type=int, default=1024, help="Resize preview so that the longer edge <= this pixels (0 to disable)")
    return parser.parse_args()


def _read_jsonl(path: str) -> List[dict]:
    data: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                continue
    return data


def _write_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
def extract_first_question_sentence(text: str) -> str:
    s = text.lstrip()
    if s.startswith("\n"):
        s = s[1:]
    s = s.lstrip()
    # take first line or until '?'
    qmark = s.find("?")
    if qmark >= 0:
        s = s[: qmark + 1]
    else:
        s = s.splitlines()[0] if s.splitlines() else s
    return s.strip()

def _infer_modalities_for_item(item: dict, mode: str) -> List[str]:
    item_text = item.get("text", "")
    inferred = _infer_modalities_from_question(extract_first_question_sentence(item_text))
    if mode == "merge":
        base = ["rgb"]
        for m in CANONICAL_MODALITIES:
            if m in inferred and m not in base:
                base.append(m)
        return _canonicalize_modalities(base)
    # override (default)
    out = ["rgb"]
    for m in inferred:
        if m != "rgb" and m not in out:
            out.append(m)
    return _canonicalize_modalities(out)


def _validate_env_and_inputs(selected_modalities: List[str], args: argparse.Namespace):
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


def run_inference_per_item(items: List[dict], args: argparse.Namespace, tokenizer, processor, model, is_free: bool, result_path: str) -> None:
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
        data_iter = items[: args.limit]
    else:
        data_iter = items
    preview_emitted = 0
    for idx, item in enumerate(tqdm(data_iter, total=len(data_iter))):
        item_text = item.get("text", "")
        sel_modalities = _infer_modalities_for_item(item, args.infer_mode)
        _validate_env_and_inputs(sel_modalities, args)

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

        req_mods = [m for m in sel_modalities if m in ["rgb", "dsm", "svf", "seg"]]
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
            modal_note = f"[Modalities={','.join(sel_modalities)} | DSM={args.dsm_colormap} | SVF={args.svf_colormap}]"
            item_text = (item_text + "\n" + modal_note).strip()

        messages, used_modalities = prepare_multimodal_message(
            image_paths=image_paths,
            text=item_text,
            modalities=sel_modalities,
            dsm_colormap=args.dsm_colormap,
            svf_colormap=args.svf_colormap,
            is_free=is_free,
            dsm_show_colorbar=bool(getattr(args, 'dsm_show_colorbar', False)),
            dsm_vmin=args.dsm_vmin,
            dsm_vmax=args.dsm_vmax,
            svf_show_colorbar=bool(getattr(args, 'svf_show_colorbar', False)),
            svf_vmin=args.svf_vmin,
            svf_vmax=args.svf_vmax,
        )
        if not used_modalities:
            raise RuntimeError(f"No images attached to the message. Resolved: {image_paths}")

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
            gen_kwargs = dict(max_new_tokens=int(args.max_new_tokens))
            if args.temperature and args.temperature > 0:
                gen_kwargs.update(dict(do_sample=True, temperature=float(args.temperature)))
            else:
                gen_kwargs.update(dict(do_sample=False, temperature=None))
            output = model.generate(**inputs, **gen_kwargs)

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
            if args.preview_dir and preview_emitted < max(0, int(getattr(args, "preview_count", 0) or 0)):
                os.makedirs(args.preview_dir, exist_ok=True)
                item_idx = preview_emitted
                saved_rel_paths = []
                for m in used_modalities:
                    p = image_paths.get(m)
                    if not p or not os.path.exists(p):
                        continue
                    out_ext = ".png" if (str(args.preview_format).lower() == "png") else ".jpg"
                    out_path = os.path.join(args.preview_dir, f"{item_idx:04d}_{m}{out_ext}")
                    img_obj = None
                    if m == "rgb":
                        try:
                            from PIL import Image  # local import to avoid hard dep at module import
                            img_obj = Image.open(p).convert("RGB")
                        except Exception:
                            img_obj = None
                    elif m == "dsm":
                        img_obj = dsm_to_rgb(
                            p,
                            colormap=args.dsm_colormap,
                            show_colorbar=bool(getattr(args, 'dsm_show_colorbar', False)),
                            vmin=args.dsm_vmin,
                            vmax=args.dsm_vmax,
                        )
                    elif m == "svf":
                        img_obj = svf_to_rgb(
                            p,
                            colormap=args.svf_colormap,
                            show_colorbar=bool(getattr(args, 'svf_show_colorbar', False)),
                            vmin=args.svf_vmin,
                            vmax=args.svf_vmax,
                        )
                    elif m == "seg":
                        img_obj = seg_to_rgb(p)
                    if img_obj is not None:
                        try:
                            if args.preview_max_edge and int(args.preview_max_edge) > 0:
                                try:
                                    max_edge = int(args.preview_max_edge)
                                    w, h = img_obj.size
                                    if max(w, h) > max_edge:
                                        img_obj.thumbnail((max_edge, max_edge))
                                except Exception:
                                    pass
                            save_kwargs = {}
                            if out_ext == ".jpg":
                                try:
                                    img_obj = img_obj.convert("RGB")
                                except Exception:
                                    pass
                                save_kwargs.update(dict(quality=int(getattr(args, 'preview_quality', 90) or 90), optimize=True))
                            img_obj.save(out_path, **save_kwargs)
                            saved_rel_paths.append({"modality": m, "file": out_path})
                        except Exception:
                            pass
                try:
                    meta_out = {
                        "index": item_idx,
                        "modalities": used_modalities,
                        "images": saved_rel_paths,
                        "prompt": text_prompt,
                        "question_id": item.get("question_id"),
                    }
                    with open(os.path.join(args.preview_dir, f"{item_idx:04d}_preview.json"), "w", encoding="utf-8") as f:
                        f.write(json.dumps(meta_out, ensure_ascii=False, indent=2))
                except Exception:
                    pass
                preview_emitted += 1
        except Exception:
            pass


def main() -> None:
    args = _parse_args()

    # Load YAML config if provided and set defaults
    if args.config and args.use_lora:
        if args.verbose:
            print(f"Loading config from: {args.config}")
        yaml_cfg = load_config_from_yaml(args.config)
        inf_cfg = extract_inference_config(yaml_cfg)
        for key, value in inf_cfg.items():
            if not hasattr(args, key) or getattr(args, key) == _parse_args.__defaults__:
                setattr(args, key, value)
        if args.verbose:
            print(f"  - Base model: {getattr(args, 'base_model_path', None)}")

    # Load tokenizer/processor/model
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
        # Expect lora_model_path from YAML
        lora_path = getattr(args, 'lora_model_path', None)
        if lora_path:
            resolved = resolve_lora_path(lora_path)
            if args.verbose:
                print(f"Using LoRA from: {resolved}")
            model = PeftModel.from_pretrained(model, resolved)
            if getattr(model, "hf_device_map", None) is not None:
                model = model.eval()
            else:
                model = model.to("cuda").eval()
        else:
            model = model.eval()
    else:
        model = model.eval()

    # Determine output path
    out_path = args.output_file or (args.input_file.replace(".jsonl", "_agent_results.jsonl"))
    if os.path.exists(out_path):
        base, ext = os.path.splitext(out_path)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"{base}_new_{ts}{ext}"
        if args.verbose:
            print(f"Output exists; writing to: {out_path}")

    # Free-form detection
    is_free = ("free" in args.input_file and "combined" not in args.input_file)

    items = _read_jsonl(args.input_file)
    if args.verbose:
        print(f"Loaded {len(items)} items from {args.input_file}")

    # Ensure fresh file
    try:
        if os.path.exists(out_path):
            os.remove(out_path)
    except Exception:
        pass

    run_inference_per_item(items, args, tokenizer, processor, model, is_free, out_path)
    print(f"Wrote results: {out_path} ({len(items)} items)")


if __name__ == "__main__":
    main()
"""
  python llama-factory-0802/configs/agent_rooter.py \
    --config llama-factory-0802/configs/comb_1019_multi_select.yaml \
    --image_dir /workspace/GeoNRW \
    --dsm_dir /workspace/GeoNRW \
    --svf_dir /workspace/GeoNRW/svf/skyview_umep_test \
    --seg_dir /workspace/GeoNRW \
    --infer_mode override \
    --use_lora \
    --max_new_tokens 256 \
    --temperature 0 \
    --verbose
"""
