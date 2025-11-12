#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
from PIL import Image
import argparse
from tqdm import tqdm
import numpy as np
from pathlib import Path
import rasterio
from rasterio.plot import reshape_as_image
import matplotlib.pyplot as plt
import time
import signal
from datetime import datetime
import cv2
import seaborn as sns
from scipy import ndimage
# from modelscope import AutoModelForCausalLM, AutoTokenizer
# from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils.svf_qa_utils import generate_modality_description
from attention_visualizer import (
    extract_attention_weights,
    visualize_attention_heatmap,
    save_attention_metadata,
    visualize_attention_patterns,
    create_cross_modal_attention_summary
)

def load_json_data(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
            else:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            data.append(item)
                        except json.JSONDecodeError as e:
                            print(f"JSON parsing error: {e}. Line: {line[:50]}...")
    except Exception as e:
        print(f"File reading error: {e}")
    return data

def process_vision_info(messages):
    images = []
    videos = None
    
    for message in messages:
        if not isinstance(message, dict) or "content" not in message:
            continue
            
        for content in message.get("content", []):
            if not isinstance(content, dict):
                continue
                
            if content.get("type") == "image":
                image_path = content.get("image")
                if not image_path:
                    continue
                    
                try:
                    if isinstance(image_path, Image.Image):
                        images.append(image_path)
                    elif os.path.exists(image_path):
                        image = Image.open(image_path).convert('RGB')
                        images.append(image)
                    else:
                        print(f"Image not found: {image_path}")
                except Exception as e:
                    print(f"Image loading error: {e}, path: {image_path}")
    
    return images, videos

def normalize_array(array, min_val=None, max_val=None):
    if min_val is None:
        min_val = np.nanmin(array)
    if max_val is None:
        max_val = np.nanmax(array)
    
    array = np.nan_to_num(array, nan=min_val)
    
    if min_val == max_val:
        return np.zeros_like(array)
    
    normalized = (array - min_val) / (max_val - min_val)
    normalized = np.clip(normalized, 0, 1)
    return normalized

def dsm_to_rgb(dsm_path, colormap='terrain'):
    try:
        with rasterio.open(dsm_path) as src:
            dsm_data = src.read(1)
            normalized = normalize_array(dsm_data)
            cm = plt.get_cmap(colormap)
            colored = cm(normalized)
            
            # RGBAからRGBに変換
            rgb_image = Image.fromarray((colored[:, :, :3] * 255).astype(np.uint8))
            return rgb_image
    except Exception as e:
        print(f"DSM loading error: {e}, path: {dsm_path}")
        return None

def svf_to_rgb(svf_path, colormap='plasma'):
    try:
        with rasterio.open(svf_path) as src:
            svf_data = src.read(1)
            normalized = normalize_array(svf_data, min_val=0, max_val=1)
            cm = plt.get_cmap(colormap)
            colored = cm(normalized)
            rgb_image = Image.fromarray((colored[:, :, :3] * 255).astype(np.uint8))
            return rgb_image
    except Exception as e:
        print(f"SVF loading error: {e}, path: {svf_path}")
        return None

def seg_to_rgb(seg_path):
    """セグメンテーションデータをRGB画像に変換する
    GeoNRWのセグメンテーションクラス:
    0. background
    1. forest
    2. water
    3. agricultural
    4. residential,commercial,industrial
    5. grassland,swamp,shrubbery
    6. railway,trainstation
    7. highway,squares
    8. airport,shipyard
    9. roads
    10. buildings
    """
    try:
        with rasterio.open(seg_path) as src:
            seg_data = src.read()
            
            if seg_data.shape[0] == 1:
                seg_data = seg_data[0]
                colored = np.zeros((seg_data.shape[0], seg_data.shape[1], 3), dtype=np.uint8)
                
                colors = {
                    0: [0, 0, 0],       # 背景: 黒
                    1: [0, 100, 0],     # 森林: 深緑
                    2: [0, 0, 255],     # 水域: 青
                    3: [255, 255, 0],   # 農地: 黄色
                    4: [128, 128, 128], # 住宅/商業/工業: グレー
                    5: [144, 238, 144], # 草地/湿地/低木: ライトグリーン
                    6: [165, 42, 42],   # 鉄道/駅: 茶色
                    7: [192, 192, 192], # 高速道路/広場: ライトグレー
                    8: [255, 165, 0],   # 空港/造船所: オレンジ
                    9: [128, 128, 0],   # 道路: オリーブ
                    10: [255, 0, 0],    # 建物: 赤
                }
                
                for cls, color in colors.items():
                    mask = (seg_data == cls)
                    colored[mask] = color
                
                rgb_image = Image.fromarray(colored)
            else:
                seg_data = reshape_as_image(seg_data)
                if seg_data.shape[2] > 3:
                    seg_data = seg_data[:, :, :3]
                rgb_image = Image.fromarray(seg_data.astype(np.uint8))
            
            return rgb_image
    except Exception as e:
        print(f"Segmentation loading error: {e}, path: {seg_path}")
        return None

def get_related_image_paths(rgb_path, dsm_dir, svf_dir, seg_dir):
    try:
        rgb_path = Path(rgb_path)
        if not rgb_path.exists():
            print(f"Warning: RGB image not found: {rgb_path}")
            return {"rgb": None, "dsm": None, "svf": None, "seg": None}
        
        base_dir = rgb_path.parent
        city = base_dir.name
        
        if "_rgb.jp2" in rgb_path.name:
            base_name = rgb_path.name.replace("_rgb.jp2", "")
        else:
            base_name = rgb_path.stem
        
        dsm_path = os.path.join(dsm_dir, city, f"{base_name}_dem.tif") if dsm_dir else None
        svf_path = os.path.join(svf_dir, city, f"{base_name}_dem_svf_umep.tif") if svf_dir else None
        seg_path = os.path.join(seg_dir, city, f"{base_name}_seg.tif") if seg_dir else None
        
        paths = {
            "rgb": str(rgb_path) if rgb_path.exists() else None,
            "dsm": dsm_path if dsm_path and os.path.exists(dsm_path) else None,
            "svf": svf_path if svf_path and os.path.exists(svf_path) else None,
            "seg": seg_path if seg_path and os.path.exists(seg_path) else None,
        }
        
        return paths
    except Exception as e:
        print(f"Image path resolution error: {e}, RGB path: {rgb_path}")
        return {"rgb": None, "dsm": None, "svf": None, "seg": None}

def get_geonrw_class_names():
    return {
        0: "Background", 1: "Forest", 2: "Water", 3: "Agricultural",
        4: "Residential/Commercial/Industrial", 5: "Grassland/Swamp/Shrubbery",
        6: "Railway/Train station", 7: "Highway/Squares", 8: "Airport/Shipyard",
        9: "Roads", 10: "Buildings"
    }

def get_geonrw_class_colors():
    return {
        0: [0, 0, 0], 1: [0, 100, 0], 2: [0, 0, 255], 3: [255, 255, 0],
        4: [128, 128, 128], 5: [144, 238, 144], 6: [165, 42, 42],
        7: [192, 192, 192], 8: [255, 165, 0], 9: [128, 128, 0], 10: [255, 0, 0]
    }

def prepare_multi_image_message(image_paths, text, modalities, dsm_colormap, svf_colormap, save_images=True, text_only=False):
    message_content = []
    used_modalities = []
    temp_files = []
    
    if text_only:
        available_modalities = [m for m in modalities if m in image_paths and image_paths[m]]
        if available_modalities:
            modality_info = f"[Available modalities: {', '.join(available_modalities)}]\n\n"
            text = modality_info + text
        
        message_content.append({"type": "text", "text": text})
        message = [{"role": "user", "content": message_content}]
        return message, [], []
    
    if save_images:
        os.makedirs("temp_images", exist_ok=True)
    
    for modality in modalities:
        if modality in image_paths and image_paths[modality]:
            path = image_paths[modality]
            
            if modality == "rgb":
                message_content.append({"type": "image", "image": path})
                used_modalities.append(modality)
            elif modality == "dsm" and path:
                img = dsm_to_rgb(path, colormap=dsm_colormap)
                if img:
                    temp_path = f"temp_{modality}_{os.path.basename(path)}.png"
                    img.save(temp_path)
                    message_content.append({"type": "image", "image": temp_path})
                    used_modalities.append(modality)
                    temp_files.append(temp_path)
            elif modality == "svf" and path:
                img = svf_to_rgb(path, colormap=svf_colormap)
                if img:
                    temp_path = f"temp_{modality}_{os.path.basename(path)}.png"
                    img.save(temp_path)
                    message_content.append({"type": "image", "image": temp_path})
                    used_modalities.append(modality)
                    temp_files.append(temp_path)
            elif modality == "seg" and path:
                img = seg_to_rgb(path)
                if img:
                    temp_path = f"temp_{modality}_{os.path.basename(path)}.png"
                    img.save(temp_path)
                    message_content.append({"type": "image", "image": temp_path})
                    used_modalities.append(modality)
                    temp_files.append(temp_path)
    
    if set(used_modalities) == set(["rgb", "dsm", "svf", "seg"]):
        interpret_guide = "Interpret viridis (purple→blue→green→yellow) as low→high elevation, plasma (purple→red→orange→yellow→red) as low→high SVF, and each GeoNRW RGB color as its corresponding segmentation class."
        text = interpret_guide + "\n\n" + text
    
    modality_desc = generate_modality_description(used_modalities, dsm_colormap, svf_colormap)
    
    if modality_desc:
        enhanced_text = f"{modality_desc}\n\n**Question:** {text}\n\nPlease analyze the provided image(s) and answer the question. After your analysis, provide your final answer in the format: <final short answer>your answer</final short answer>"
    else:
        enhanced_text = f"**Question:** {text}\n\nPlease answer the question concisely. After your analysis, provide your final answer in the format: <final short answer>your answer</final short answer>"
    
    message_content.append({"type": "text", "text": enhanced_text})
    message = [{"role": "user", "content": message_content}]
    
    return message, used_modalities, temp_files

def prepare_messages(image_paths, text, modalities, dsm_colormap='terrain', svf_colormap='plasma', temp_dir=None):
    content = []
    
    processed_images, temp_files = prepare_multimodal_images(
        image_paths, modalities, dsm_colormap, svf_colormap, temp_dir
    )
    
    added_images = 0
    actual_modalities = []
    for modality in modalities:
        if modality in processed_images:
            image_path = processed_images[modality]
            
            if not os.path.exists(image_path):
                print(f"Warning: Processed image file not found: {image_path}")
                continue
                
            if os.path.getsize(image_path) == 0:
                print(f"Warning: Processed image file size is 0: {image_path}")
                continue
                
            base64_image = encode_image_to_base64(image_path)
            if base64_image:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })
                added_images += 1
                actual_modalities.append(modality)
                print(f"  Added {modality} image to message")
            else:
                print(f"Warning: Could not encode {modality} image {image_path}")
    
    if added_images == 0:
        print("Warning: No valid images were added")
    
    modality_desc = generate_modality_description(actual_modalities, dsm_colormap, svf_colormap)
    
    if modality_desc:
        enhanced_text = f"{modality_desc}\n\n**Question:** {text}\n\nPlease analyze the provided image(s) and answer the question concisely with brief reasoning."
    else:
        enhanced_text = f"**Question:** {text}\n\nPlease answer the question concisely with brief reasoning."
    
    content.append({
        "type": "text", 
        "text": enhanced_text
    })
    
    message = {
        "role": "user",
        "content": content
    }
    
    return message, temp_files

def main(args):
    print(f"Loading question data: {args.questions_file}")
    questions = load_json_data(args.questions_file)
    print(f"Loaded {len(questions)} questions")
    
    temp_output_file = f"{args.output_file}.temp"
    completed_question_ids = set()
    results = []
    
    if os.path.exists(temp_output_file) and not args.restart:
        print(f"Loading intermediate results: {temp_output_file}")
        temp_results = load_json_data(temp_output_file)
        for result in temp_results:
            completed_question_ids.add(result.get("question_id", 0))
            results.append(result)
        print(f"{len(completed_question_ids)} questions already processed")
    
    print("Loading Qwen2.5-VL model...")
    torch.cuda.set_per_process_memory_fraction(0.8)
    torch.cuda.empty_cache()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    print("Model loaded")
    
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_name)
    print("Processor loaded")
    
    dsm_dir = args.dsm_dir or args.image_dir
    svf_dir = args.svf_dir or args.image_dir
    seg_dir = args.seg_dir or args.image_dir
    
    if args.use_all_modalities:
        modalities = ["rgb", "dsm", "svf", "seg"]
        print("Using all modalities")
    else:
        modalities = [m.strip() for m in args.modalities.split(",")]
    print(f"Using modalities: {modalities}")
    
    if args.text_only:
        print("Text-only mode enabled")
    
    batch_size = args.batch_size
    question_slice = questions[:args.limit] if args.limit > 0 else questions
    total_questions = len(question_slice)
    remaining_questions = [q for q in question_slice if q.get("question_id", 0) not in completed_question_ids]
    print(f"Remaining questions: {len(remaining_questions)}/{total_questions}")
    
    os.makedirs("temp_images", exist_ok=True)
    last_save_time = time.time()
    
    try:
        for i in range(0, len(remaining_questions), batch_size):
            batch_questions = remaining_questions[i:i+batch_size]
            batch_messages = []
            batch_question_info = []
            all_temp_files = []
            
            batch_num = i // batch_size + 1
            total_batches = (len(remaining_questions) - 1) // batch_size + 1
            print(f"Processing batch {batch_num}/{total_batches}... ({(i + len(batch_questions))}/{len(remaining_questions)} questions)")
            
            for question in batch_questions:
                question_id = question.get("question_id", 0)
                image_rel_path = question.get("image", "")
                text = question.get("text", "")
                category = question.get("category", "unknown")
                
                rgb_path = os.path.join(args.image_dir, image_rel_path)
                image_paths = get_related_image_paths(rgb_path, dsm_dir, svf_dir, seg_dir)
                
                messages, used_modalities, temp_files = prepare_multi_image_message(
                    image_paths, text, modalities, args.dsm_colormap, args.svf_colormap, 
                    args.save_images, args.text_only
                )
                displayed_text = text
                if set(used_modalities) == set(["rgb", "dsm", "svf", "seg"]):
                    interpret_guide = "Interpret viridis (purple→blue→green→yellow) as low→high elevation, plasma (purple→red→orange→yellow) as low→high SVF, and each GeoNRW RGB color as its corresponding segmentation class."
                    displayed_text = interpret_guide + "\n\n" + text
                
                batch_messages.append(messages)
                batch_question_info.append({
                    "question_id": question_id,
                    "image_rel_path": image_rel_path,
                    "text": text,
                    "category": category,
                    "used_modalities": used_modalities
                })
                all_temp_files.extend(temp_files)
            
            batch_results = process_batch(model, processor, batch_messages, args)
            
            for idx, output_text in enumerate(batch_results):
                if "</think>" in output_text:
                    output_text = output_text.split("</think>")[1]
                    if output_text.startswith("\n\n"):
                        output_text = output_text[2:]
                else:
                    output_text = output_text
                info = batch_question_info[idx]
                result = {
                    "question_id": info["question_id"],
                    "image": info["image_rel_path"],
                    "text": info["text"],
                    "category": info["category"],
                    "used_modalities": info["used_modalities"],
                    "answer": output_text
                }
                results.append(result)
                
                if args.verbose:
                    print(f"\nQuestion ID: {info['question_id']}")
                    print(f"Used modalities: {', '.join(info['used_modalities'])}")
                    print(f"Question: {info['text']}")
                    print(f"Answer: {output_text}")
            
            if not args.save_images:
                for temp_file in all_temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except Exception as e:
                        print(f"Temporary file deletion error: {e}, file: {temp_file}")
            
            current_time = time.time()
            if current_time - last_save_time > 60:
                save_results(results, temp_output_file)
                last_save_time = current_time
                print(f"Saved intermediate results: {len(results)}/{total_questions} questions completed")
        
        if results:
            output_file = args.output_file
            print(f"Saving results to file: {output_file}")
            save_results(results, output_file)
            
            if os.path.exists(temp_output_file):
                os.remove(temp_output_file)
            
            print(f"Processing complete. Generated {len(results)} answers.")
    
    finally:
        if results and not os.path.exists(args.output_file):
            save_results(results, temp_output_file)
            print(f"Interrupted but saved {len(results)} results to intermediate file: {temp_output_file}")
        
        if args.save_images:
            import glob
            for temp_file in glob.glob("temp_*.png"):
                try:
                    os.remove(temp_file)
                except:
                    pass

def save_results(results, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def process_batch(model, processor, batch_messages, args):
    """Process multiple messages in batch efficiently
    1. Prepare text prompts for each message in the batch
    2. Extract images from each message
    3. Process images and text with Qwen2.5-VL model
    4. Execute inference for each prompt
    5. Return generated text output
    6. (Optional) Run attention visualization
    
    Args:
        model: Qwen2.5-VLモデル
        processor: モデルのプロセッサ
        batch_messages: 処理するメッセージシーケンスのリスト
        args: コマンドライン引数
        
    Returns:
        生成されたテキスト応答のリスト
    """
    # タイムアウトハンドラー
    class TimeoutException(Exception):
        pass

    import platform
    is_windows = platform.system() == 'Windows'

    def timeout_handler(signum, frame):
        raise TimeoutException("Processing timeout")

    output_texts = []
    
    if args.visualize_attention:
        attention_output_dir = os.path.join(os.path.dirname(args.output_file), "attention_visualizations")
        os.makedirs(attention_output_dir, exist_ok=True)
        print(f"Attention visualization output directory: {attention_output_dir}")
    
    for i, messages in enumerate(batch_messages):
        print(f"  Processing question {i+1}/{len(batch_messages)} in batch...")
        
        try:
            print(f"    Starting model generation...")
            start_time = time.time()
            
            try:
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
                inputs = inputs.to(model.device)
                
                if args.visualize_attention and image_inputs:
                    print(f"    Running attention visualization...")
                    try:
                        attention_weights, generation_outputs = extract_attention_weights(model, inputs)
                        
                        if hasattr(generation_outputs, 'sequences'):
                            generated_ids = generation_outputs.sequences
                        else:
                            with torch.no_grad():
                                generated_ids = model.generate(
                                    **inputs,
                                    max_new_tokens=args.max_new_tokens,
                                    do_sample=args.temperature > 0,
                                    temperature=args.temperature if args.temperature > 0 else None,
                                    top_p=0.95 if args.temperature > 0 else None,
                                    top_k=20 if args.temperature > 0 else None,
                                )
                        
                        input_length = inputs.input_ids.shape[1]
                        generated_ids_trimmed = generated_ids[:, input_length:]
                        
                        output = processor.batch_decode(
                            generated_ids_trimmed, 
                            skip_special_tokens=True, 
                            clean_up_tokenization_spaces=False
                        )[0]
                        
                        question_id = f"batch_{i}"
                        question_text = text[:200] + "..." if len(text) > 200 else text
                        
                        used_modalities = []
                        for message in messages:
                            if "content" in message:
                                for content in message["content"]:
                                    if content.get("type") == "image":
                                        image_path = content.get("image", "")
                                        if "temp_dsm" in image_path:
                                            used_modalities.append("dsm")
                                        elif "temp_svf" in image_path:
                                            used_modalities.append("svf")
                                        elif "temp_seg" in image_path:
                                            used_modalities.append("seg")
                                        else:
                                            used_modalities.append("rgb")
                        
                        if not used_modalities:
                            used_modalities = ["rgb"]
                        
                        visualize_attention_heatmap(
                            attention_weights, 
                            image_inputs, 
                            attention_output_dir, 
                            question_id, 
                            used_modalities
                        )
                        
                        save_attention_metadata(
                            attention_weights, 
                            attention_output_dir, 
                            question_id, 
                            question_text, 
                            output
                        )
                        
                    except Exception as attention_error:
                        print(f"    Attention visualization error: {attention_error}")
                        with torch.no_grad():
                            generated_ids = model.generate(
                                **inputs,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=args.temperature > 0,
                                temperature=args.temperature if args.temperature > 0 else None,
                                top_p=0.95 if args.temperature > 0 else None,
                                top_k=20 if args.temperature > 0 else None,
                            )
                        
                        input_length = inputs.input_ids.shape[1]
                        generated_ids_trimmed = generated_ids[:, input_length:]
                        
                        output = processor.batch_decode(
                            generated_ids_trimmed, 
                            skip_special_tokens=True, 
                            clean_up_tokenization_spaces=False
                        )[0]
                
                else:
                    with torch.no_grad():
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=args.temperature > 0,
                            temperature=args.temperature if args.temperature > 0 else None,
                            top_p=0.95 if args.temperature > 0 else None,
                            top_k=20 if args.temperature > 0 else None,
                        )
                    
                    input_length = inputs.input_ids.shape[1]
                    generated_ids_trimmed = generated_ids[:, input_length:]
                    
                    output = processor.batch_decode(
                        generated_ids_trimmed, 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=False
                    )[0]
                
                elapsed_time = time.time() - start_time
                print(f"    Model generation completed (elapsed time: {elapsed_time:.2f}s)")
                
                content = output
                
            except Exception as e:
                print(f"    Error during generation: {e}")
                content = f"Error: Model generation failed - {str(e)}"
                
        except Exception as outer_e:
            print(f"    Unexpected error during processing: {outer_e}")
            content = f"Error: Processing failed - {str(outer_e)}"
        
        output_texts.append(content)
    
    return output_texts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2.5-VL model for landscape question answering")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Qwen2.5-VL model name")
    parser.add_argument("--questions_file", type=str, default="svf_qa/svf_questions_test_0612.jsonl",
                        help="Path to question data JSON file")
    parser.add_argument("--image_dir", type=str, default="GeoNRW/",
                        help="Base directory for RGB image files")
    parser.add_argument("--dsm_dir", type=str, default=None,
                        help="DSM data directory")
    parser.add_argument("--svf_dir", type=str, default="svf/skyview_umep_test/",
                        help="SVF data directory")
    parser.add_argument("--seg_dir", type=str, default=None,
                        help="Segmentation data directory")
    parser.add_argument("--use_all_modalities", action="store_true",
                        help="Use all modalities (RGB, DSM, SVF, segmentation)")
    parser.add_argument("--modalities", type=str, default="rgb",
                        help="Comma-separated modalities (e.g., rgb,dsm,svf)")
    parser.add_argument("--dsm_colormap", type=str, default="terrain",
                        help="DSM colormap (e.g., terrain, jet, viridis)")
    parser.add_argument("--svf_colormap", type=str, default="plasma",
                        help="SVF colormap (e.g., plasma, viridis, inferno)")
    parser.add_argument("--temperature", type=float, default=0,
                        help="Temperature parameter for generation")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--output_file", type=str, default="svf_answers_qwen_vl_output.json",
                        help="Output JSON file path")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Maximum number of questions to process (-1 for all)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed output")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (number of questions to process simultaneously)")
    parser.add_argument("--generation_timeout", type=int, default=90,
                        help="Timeout for generation processing per batch (seconds)")
    parser.add_argument("--restart", action="store_true",
                        help="Load intermediate result file")
    parser.add_argument("--save_images", action="store_true", default=False,
                        help="Save images as temporary files")
    parser.add_argument("--enable_thinking", action="store_true", default=False,
                        help="Enable thinking mode")
    parser.add_argument("--text_only", action="store_true", default=False,
                        help="Enable text-only mode (do not pass images to LLM)")
    parser.add_argument("--single_modality_test", action="store_true", default=False,
                        help="Enable single modality test mode")
    parser.add_argument("--modality_combinations", type=str, default=None,
                        help="Comma-separated modality combinations to test (e.g., rgb,svf,rgb+svf)")
    parser.add_argument("--visualize_attention", action="store_true", default=False,
                        help="Run attention visualization")
    parser.add_argument("--attention_layer_indices", type=str, default=None,
                        help="Specific layer indices for attention visualization (comma-separated, e.g., -1,-2,-3)")
    parser.add_argument("--attention_head_average", action="store_true", default=True,
                        help="Average multiple attention heads")
    
    args = parser.parse_args()
    today = datetime.now().strftime("%Y%m%d")
    input_filename = os.path.basename(args.questions_file)
    input_basename = os.path.splitext(input_filename)[0]
    
    modality_info = ""
    if args.modalities and not args.use_all_modalities and not args.text_only:
        modality_info = f"{args.modalities.replace(',', '_')}_"
    
    args.output_file = f"qwen_{input_basename}_{modality_info}{today}_output.json"
    
    print(f"Input file: {args.questions_file}")
    print(f"Output file will be: {args.output_file}")
    args.output_file = os.path.join(os.path.dirname(args.questions_file), "qwen", args.output_file)
    
    if args.single_modality_test:
        all_modalities = ["rgb", "dsm", "svf", "seg"]
        
        if args.modality_combinations:
            combinations = args.modality_combinations.split(',')
        else:
            combinations = all_modalities + ["all"]
        
        original_output_file = args.output_file
        original_questions_file = args.questions_file
        
        for combo in combinations:
            if combo == "all":
                args.use_all_modalities = True
                args.modalities = ",".join(all_modalities)
                modality_info = "all_"
            else:
                args.use_all_modalities = False
                args.modalities = combo.replace('+', ',')
                modality_info = f"{args.modalities.replace(',', '_')}_"
            
            input_basename = os.path.splitext(os.path.basename(original_questions_file))[0]
            args.output_file = f"{input_basename}_{modality_info}{today}_output.json"
            args.output_file = os.path.join(os.path.dirname(original_questions_file), args.output_file)
            
            print(f"\n=== Testing with modalities '{args.modalities}' ===")
            print(f"Output file: {args.output_file}")
            
            main(args)
    else:
        main(args)