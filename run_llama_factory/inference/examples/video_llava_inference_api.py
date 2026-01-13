#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from datetime import datetime
from typing import List, Tuple

import torch
from PIL import Image
from tqdm import tqdm

# Use TEOChat's Video-LLaVA utilities
from videollava.eval.eval import load_model as load_videollava_model
from videollava.eval.inference import run_inference_single as videollava_generate


def _resolve_output_path(input_file: str, model_name: str) -> str:
    today = datetime.now().strftime("%Y%m%d")
    input_stem = os.path.basename(input_file).split(".")[0]
    output_file = f"{input_stem}_{model_name}_{today}_results.jsonl"
    result_file = os.path.join("data", output_file)
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    if os.path.exists(result_file):
        base, ext = os.path.splitext(result_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"{base}_new_{timestamp}{ext}"
        print(f"既存の結果ファイルが存在するため、新しいファイル名に変更します: {result_file}")
    print(f"出力先ファイル: {result_file}")
    return result_file


def _read_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _ensure_paths_exist(paths: List[str]) -> List[str]:
    valid_paths = []
    for p in paths:
        if os.path.exists(p):
            valid_paths.append(p)
        else:
            print(f"警告: ファイルが存在しません: {p}")
    return valid_paths


def _collect_video_frames(
    video_path_or_dir: str,
    image_root: str,
    max_frames: int,
) -> Tuple[List[str], List[str]]:
    """
    Return (image_paths, timestamps). Video-LLaVA uses image sequences as video proxy.
    - If `video_path_or_dir` is a directory, treat each file as a frame, sorted.
    - If it's a file path to a video, attempt nearby extracted frame directory: <video>.frames/ or <stem>_frames/.
    """
    timestamps: List[str] = []
    # Build absolute path
    abs_path = video_path_or_dir
    if not os.path.isabs(abs_path):
        abs_path = os.path.join(image_root, video_path_or_dir)

    frames_dir_candidates: List[str] = []
    if os.path.isdir(abs_path):
        frames_dir_candidates.append(abs_path)
    else:
        stem, _ = os.path.splitext(abs_path)
        frames_dir_candidates.extend([
            f"{abs_path}.frames",
            f"{stem}_frames",
            os.path.join(os.path.dirname(abs_path), os.path.basename(stem)),
        ])

    selected_dir = None
    for d in frames_dir_candidates:
        if os.path.isdir(d):
            selected_dir = d
            break

    image_paths: List[str] = []
    if selected_dir is not None:
        all_files = [os.path.join(selected_dir, f) for f in os.listdir(selected_dir)]
        all_files = [p for p in all_files if os.path.isfile(p)]
        # Basic sort that works for frame_000001.png etc.
        all_files.sort()
        if max_frames > 0 and len(all_files) > max_frames:
            # Uniform sample indices
            step = len(all_files) / float(max_frames)
            indices = [int(i * step + step / 2) for i in range(max_frames)]
            image_paths = [all_files[i] for i in indices]
        else:
            image_paths = all_files
    else:
        # Fallback: if abs_path is an image file, use it as a single-frame video
        if os.path.isfile(abs_path):
            image_paths = [abs_path]
        else:
            print(f"警告: フレームディレクトリ/ファイルが見つかりません: {abs_path}")
            image_paths = []

    image_paths = _ensure_paths_exist(image_paths)
    return image_paths, timestamps


def main():
    parser = argparse.ArgumentParser(description="Video-LLaVA Inference Script (image-sequence video)")

    # Model and backend
    parser.add_argument('--model_path', type=str, default='LanguageBind/Video-LLaVA-7B-hf', help='Video-LLaVA model id/path')
    parser.add_argument('--device', type=str, default='cuda', help='Device for inference')
    parser.add_argument('--load_8bit', action='store_true', help='Load model in 8bit')

    # Data IO
    parser.add_argument('--input_file', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--media_root', type=str, default='/workspace/GeoNRW', help='Root directory for images/videos')

    # Generation
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.2)

    # Video sampling
    parser.add_argument('--max_frames', type=int, default=16, help='Uniformly sampled frames per video')
    parser.add_argument('--prompt_strategy', type=str, default='interleave', choices=['interleave', None], help='How to expand <video> to image tokens')
    parser.add_argument('--chronological_prefix', action='store_true', help='Chronological phrase tweak in prompt')

    args = parser.parse_args()

    # Load model (tokenizer, model, processor)
    print("Video-LLaVAモデルを読み込み中...")
    tokenizer, model, processor = load_videollava_model(
        model_path=args.model_path,
        model_base=None,
        load_8bit=args.load_8bit,
        device=args.device,
    )
    print("Video-LLaVAモデルの読み込み完了")

    # Build output path
    model_short = os.path.basename(args.model_path)
    result_file = _resolve_output_path(args.input_file, model_short)

    # Read data
    data = _read_jsonl(args.input_file)

    # Infer per item
    for item in tqdm(data):
        text = item.get("text", "")
        is_free = "free" in args.input_file
        if is_free:
            prompt_text = (
                text
                + "\nPlease structure your response using only <OBSERVATION> and <CONCLUSION>:\n"
                + "<OBSERVATION>Statistical data and image-grounded observations only (SVF values, elevation in meters, land cover %).</OBSERVATION>\n"
                + "<CONCLUSION>Base the decision strictly on <OBSERVATION> and answer consisely to the question; please mention at least one grid for the answer.</CONCLUSION>."
            )
        else:
            prompt_text = text + "\nPlease provide only your final short answer to the question concisely."

        # Resolve media: expect `video` field, fallback to `image`
        video_rel = item.get("video") or item.get("image")
        if video_rel is None:
            print("警告: 入力項目に 'video' も 'image' も見つかりません。スキップします。")
            continue

        frame_paths, timestamps = _collect_video_frames(video_rel, args.media_root, args.max_frames)
        if len(frame_paths) == 0:
            print(f"警告: フレーム取得失敗: {video_rel}。スキップします。")
            continue

        # Sanity check images
        verified_paths: List[str] = []
        for p in frame_paths:
            try:
                with Image.open(p) as im:
                    _ = im.size
                verified_paths.append(p)
            except Exception as e:
                print(f"警告: 画像読み込み失敗: {p} ({e})")

        if len(verified_paths) == 0:
            print(f"警告: 使用可能なフレームがありません: {video_rel}")
            continue

        # Build prompt with <video> token expected by Video-LLaVA conversation template
        prompt = f"<video> {prompt_text}"

        try:
            response = videollava_generate(
                model,
                processor,
                tokenizer,
                prompt,
                verified_paths,
                conv_mode="v1",
                timestamps=timestamps,
                prompt_strategy=args.prompt_strategy,
                chronological_prefix=args.chronological_prefix,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as e:
            print(f"生成中にエラーが発生: {e}")
            response = f"エラー: {e}"

        result = {
            "question_id": item.get("question_id"),
            "prompt": text,
            "predict": response.strip() if isinstance(response, str) else str(response),
            "label": item.get("output", ""),
        }
        with open(result_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"推論完了: {len(data)}件の結果を{result_file}に保存しました")


if __name__ == "__main__":
    main()


