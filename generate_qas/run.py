# -*- coding: utf-8 -*-
import os
import random
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time
from datetime import datetime
import argparse
import sys
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from concurrent.futures import ProcessPoolExecutor
import gc
import warnings
import re
from pathlib import Path
from utils_sampling import filter_by_distance, filter_by_metric_gap, select_by_quartiles, select_by_quartiles_with_top
from svf_questions_rgb_estimated import ConstructSVFQuestionRGB # Added import
from svf_questions_region_based import ConstructSVFQuestionRegionBased
from utils_seed import get_seed_manager, set_global_seed, get_global_seed

try:
    from visualize_qa_results import QAVisualizer
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False
    print("Warning: Could not import visualize_qa_results.py. Visualization features are not available.")

plot_counter = 0
MAX_PLOTS = 50
temp_data_buffer = {
    'conversation_data': [],
    'question_count': 0,
    'last_save_time': time.time(),
    'temp_files': []
}

batch_buffer = {
    'answers': [],
    'questions': [],
    'detailed': [],
    'conversations': [],
    'buffer_size': 0,
    'max_buffer_size': 500
}

performance_tracker = {
    'category_times': {},
    'total_questions': {},
    'slow_categories': [],
    'file_processing_times': {},
}

try:
    from PIL import Resampling
    NEAREST = Resampling.NEAREST
    BICUBIC = Resampling.BICUBIC
except ImportError:
    NEAREST = Image.NEAREST
    BICUBIC = Image.BICUBIC

import logging

logging.basicConfig(filename='process_log.log',
                    level=logging.DEBUG,
                    format='%(asctime)s %(process)d %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def tqdm_safe_print(*args, **kwargs):
    """Safely print even when tqdm is active"""
    try:
        if tqdm is not None:
            tqdm.write(*args, **kwargs)
        else:
            print(*args, **kwargs)
    except Exception as e:
        pass

def record_category_performance(category, elapsed_time, question_count=1):
    """Record performance for each category"""
    global performance_tracker
    
    if isinstance(elapsed_time, list):
        if len(elapsed_time) > 0:
            elapsed_time = sum(elapsed_time) / len(elapsed_time)
        else:
            return
    
    if not isinstance(elapsed_time, (int, float)):
        return
    
    if category not in performance_tracker['category_times']:
        performance_tracker['category_times'][category] = []
        performance_tracker['total_questions'][category] = 0
    
    performance_tracker['category_times'][category].append(elapsed_time)
    performance_tracker['total_questions'][category] += question_count
    
    if elapsed_time > 5.0 and category not in performance_tracker['slow_categories']:
        performance_tracker['slow_categories'].append(category)
        tqdm_safe_print(f"Warning: Slow category detected: {category} ({elapsed_time:.2f}s)")

def print_performance_summary():
    """Display performance summary"""
    global performance_tracker
    
    if not performance_tracker['category_times']:
        return
    
    tqdm_safe_print("\n" + "="*80)
    tqdm_safe_print("Category-wise Performance Analysis")
    tqdm_safe_print("="*80)
    
    category_stats = []
    for category, times in performance_tracker['category_times'].items():
        total_time = sum(times)
        avg_time = total_time / len(times)
        max_time = max(times)
        question_count = performance_tracker['total_questions'][category]
        
        category_stats.append({
            'category': category,
            'total_time': total_time,
            'avg_time': avg_time,
            'max_time': max_time,
            'question_count': question_count,
            'calls': len(times)
        })
    
    category_stats.sort(key=lambda x: x['total_time'], reverse=True)
    
    tqdm_safe_print(f"{'Category':<25} {'Total':<8} {'Avg':<8} {'Max':<8} {'QCount':<6} {'Calls':<5}")
    tqdm_safe_print("-" * 80)
    
    for stat in category_stats:
        tqdm_safe_print(f"{stat['category']:<25} {stat['total_time']:<8.1f} {stat['avg_time']:<8.2f} {stat['max_time']:<8.2f} {stat['question_count']:<6} {stat['calls']:<5}")
    
    if len(category_stats) >= 3:
        tqdm_safe_print(f"\nTop 3 slowest categories:")
        for i, stat in enumerate(category_stats[:3], 1):
            tqdm_safe_print(f"  {i}. {stat['category']}: {stat['total_time']:.1f}s (avg: {stat['avg_time']:.2f}s)")
    
    tqdm_safe_print("="*80)

def force_memory_cleanup():
    """Force memory cleanup (enhanced)"""
    import matplotlib.pyplot as plt
    plt.close('all')
    
    try:
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
    except ImportError:
        memory_before = 0
    
    gc.collect()
    gc.collect()
    
    try:
        if memory_before > 0:
            memory_after = process.memory_info().rss / 1024 / 1024
            if memory_before - memory_after > 10:
                tqdm_safe_print(f" Memory cleanup: {memory_before:.1f}MB  {memory_after:.1f}MB")
    except:
        pass
    
    try:
        import numpy
        if hasattr(numpy, '_core') and hasattr(numpy._core, '_internal'):
            numpy._core._internal.clear_cache()
        elif hasattr(numpy, 'core') and hasattr(numpy.core, '_internal'):
            numpy.core._internal.clear_cache()
    except:
        pass
    
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
    except:
        pass
    
    if batch_buffer['buffer_size'] > 0:
        batch_buffer['answers'].clear()
        batch_buffer['questions'].clear() 
        batch_buffer['detailed'].clear()
        batch_buffer['conversations'].clear()
        batch_buffer['buffer_size'] = 0

import threading
from concurrent.futures import ThreadPoolExecutor
import queue

write_queue = queue.Queue(maxsize=100)
write_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="AsyncWriter")

def async_write_worker(file_path, data_list):
    """Async write worker"""
    try:
        with open(file_path, 'a', encoding='utf-8', buffering=8192*4) as f:
            for data in data_list:
                f.write(json.dumps(data, ensure_ascii=False, default=lambda x: float(x) if isinstance(x, np.float32) else x) + '\n')
    except Exception as e:
        tqdm_safe_print(f"Async write error: {str(e)}")

def append_detailed_data_to_file(detailed_file, detailed_data_list):
    """Append detailed data to existing JSON file"""
    if not detailed_data_list or not detailed_file:
        return
    
    try:
        existing_data = []
        if os.path.exists(detailed_file):
            try:
                with open(detailed_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = []
            except (json.JSONDecodeError, Exception):
                existing_data = []
        
        existing_data.extend(detailed_data_list)
        
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        
        tqdm_safe_print(f"Appended {len(detailed_data_list)} detailed data items: {detailed_file}")
        
    except Exception as e:
        tqdm_safe_print(f"Error appending detailed data: {str(e)}")

def flush_batch_buffer(answer_file, question_file, detailed_file, conversation_file=None):
    """Flush batch buffer contents to files asynchronously"""
    if batch_buffer['buffer_size'] == 0:
        return
    
    try:
        futures = []
        
        if batch_buffer['answers']:
            future = write_executor.submit(async_write_worker, answer_file, batch_buffer['answers'].copy())
            futures.append(future)
        
        if batch_buffer['questions']:
            future = write_executor.submit(async_write_worker, question_file, batch_buffer['questions'].copy())
            futures.append(future)
        
        if batch_buffer['detailed'] and detailed_file:
            append_detailed_data_to_file(detailed_file, batch_buffer['detailed'].copy())
        
        batch_buffer['answers'].clear()
        batch_buffer['questions'].clear()
        batch_buffer['detailed'].clear()
        batch_buffer['buffer_size'] = 0
        
    except Exception as e:
        tqdm_safe_print(f"Async batch write error: {str(e)}")

def add_to_batch_buffer(data_ans, data_ques, data_detailed, conv_data=None):
    """Add data to batch buffer"""
    batch_buffer['answers'].append(data_ans)
    batch_buffer['questions'].append(data_ques)
    batch_buffer['detailed'].append(data_detailed)
    
    if conv_data:
        batch_buffer['conversations'].append(conv_data)
    
    batch_buffer['buffer_size'] += 1
    
    return batch_buffer['buffer_size'] >= batch_buffer['max_buffer_size']

def write_json_line(file_path, data):
    """Append one line of JSON data to JSONL file"""
    def numpy_encoder(obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def convert_to_json_serializable(data):
        """Convert data to JSON-serializable format"""
        if isinstance(data, dict):
            return {key: convert_to_json_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [convert_to_json_serializable(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(convert_to_json_serializable(item) for item in data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return round(float(data), 2)
        elif isinstance(data, (np.bool_)):
            return bool(data)
        else:
            return data

    serializable_data = convert_to_json_serializable(data)

    with open(file_path, 'a', encoding='utf-8') as f:
        json.dump(serializable_data, f, ensure_ascii=False)
        f.write('\n')

def process_single_file(process_args, svf_plots, qa_multiplier=1):
    """Process single file with memory usage considerations"""
    if len(process_args) >= 5 and isinstance(process_args, tuple):
        svf_file_path, geonrw_path, area, main_args, skip_plots = process_args[:5]
        debug_mode = main_args.debug
        mode = main_args.mode
    else:
        tqdm_safe_print(f"Error: process_single_file expects at least 5 arguments in a tuple, got {len(process_args)}")
        return {
            'success': False,
            'error': "Internal error: Invalid arguments to process_single_file",
            'file_path': "unknown"
        }

    # Use centralized seed management for reproducibility
    seed_manager = get_seed_manager()
    file_seed = seed_manager.create_file_seed(svf_file_path)
    seed_manager.reset(file_seed)
    plot_limit_reached = False
    
    if mode in ["train", "both"]:
        if not hasattr(process_single_file, 'train_file_counter'):
            process_single_file.train_file_counter = 0
        
        if process_single_file.train_file_counter >= 10:
            plot_limit_reached = True
        else:
            process_single_file.train_file_counter += 1
    
    if skip_plots:
        plot_limit_reached = True
        if debug_mode:
            print(f"skip_plots option is set, skipping plot generation: {svf_file_path}")
    
    if len(svf_plots) >= 50:
        log_file = os.path.join(os.path.dirname(svf_file_path), "plot_errors.log")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"Warning: Image plot count reached 50. No more plots will be created.\n")
        return {
            'success': False,
            'error': "Image plot count limit reached",
            'file_path': svf_file_path
        }
    
    try:
        if not os.path.exists(svf_file_path):
            tqdm_safe_print(f"Error: File does not exist: {svf_file_path}")
            return {
                'success': False,
                'error': "File does not exist",
                'file_path': svf_file_path
            }
            
        try:
            svf_img = Image.open(svf_file_path)
            svf_map = np.array(svf_img)
            svf_img.close()
            if debug_mode:
                tqdm_safe_print(f"Loaded SVF data: {svf_map.shape}")
        except Exception as e:
            tqdm_safe_print(f"SVF file reading error: {svf_file_path} - {str(e)}")
            return {
                'success': False,
                'error': f"SVF file reading error: {str(e)}",
                'file_path': svf_file_path
            }
        
        base_filename = os.path.basename(svf_file_path).replace('_svf_umep.tif', '')
        
        height_file = os.path.join(geonrw_path, area, f"{base_filename}.tif")
        seg_file = os.path.join(geonrw_path, area, f"{base_filename.replace('dem', 'seg')}.tif")
        rgb_file = os.path.join(geonrw_path, area, f"{base_filename.replace('dem', 'rgb')}.jp2")
        
        target_size = None
        if os.path.exists(height_file):
            height_img = Image.open(height_file)
            height_map = np.array(height_img)
            height_img.close()
            target_size = height_map.shape
        else:
            height_map = None
            
        if target_size is not None and svf_map.shape != target_size:
            tqdm_safe_print(f"Resizing SVF map: {svf_map.shape} -> {target_size}")
            try:
                svf_img = Image.fromarray(svf_map.astype(np.float32), mode='F')
                svf_map = np.array(svf_img.resize((target_size[1], target_size[0]), BICUBIC))
                tqdm_safe_print(f"Resized SVF map: {svf_map.shape}")
            except Exception as resize_error:
                tqdm_safe_print(f"Failed to resize SVF map: {str(resize_error)}")
        
        valid_pixels = np.sum((svf_map > 0) & np.isfinite(svf_map))
        if valid_pixels < 100:
            tqdm_safe_print(f"Warning: {svf_file_path} - Insufficient valid pixels ({valid_pixels})")
            return {
                'success': False,
                'error': "Insufficient valid pixels",
                'file_path': svf_file_path
            }
        
        valid_svf = svf_map[(svf_map > 0) & np.isfinite(svf_map)]
        if len(valid_svf) > 0:
            svf_range = np.max(valid_svf) - np.min(valid_svf)
            if svf_range < 0.001:
                tqdm_safe_print(f"Warning: {svf_file_path} - SVF value variation too small (range: {svf_range})")
                return {
                    'success': False,
                    'error': "SVF value variation insufficient",
                    'file_path': svf_file_path
                }
        else:
            tqdm_safe_print(f"Warning: {svf_file_path} - No valid SVF values")
            return {
                'success': False,
                'error': "No valid SVF values",
                'file_path': svf_file_path
            }
        
        if os.path.exists(seg_file):
            seg_img = Image.open(seg_file)
            segmentation_map = np.array(seg_img)
            seg_img.close()
            
            if segmentation_map.shape != svf_map.shape:
                tqdm_safe_print(f"Warning: {seg_file} - Size mismatch with SVF map ({svf_map.shape}) vs ({segmentation_map.shape})")
                try:
                    if segmentation_map.ndim == 2:
                        seg_img = Image.fromarray(segmentation_map.astype(np.uint8), mode='L')
                        segmentation_map = np.array(seg_img.resize((svf_map.shape[1], svf_map.shape[0]), NEAREST))
                    else:
                        tqdm_safe_print(f"Unexpected segmentation map dimensions: {segmentation_map.ndim}")
                        segmentation_map = None
                    
                    if segmentation_map is not None:
                        if segmentation_map.dtype != np.uint8 and segmentation_map.dtype != np.int:
                            segmentation_map = segmentation_map.astype(np.uint8)
                        tqdm_safe_print(f"Resized segmentation map: {segmentation_map.shape}, dtype: {segmentation_map.dtype}")
                except Exception as resize_error:
                    tqdm_safe_print(f"Failed to resize segmentation map: {str(resize_error)}")
                    segmentation_map = None
        else:
            tqdm_safe_print(f"Warning: Segmentation map not found: {seg_file}")
            segmentation_map = None
        
        if os.path.exists(rgb_file):
            rgb_img = Image.open(rgb_file)
            rgb_image = np.array(rgb_img)
            rgb_img.close()
            
            if rgb_image.shape[:2] != svf_map.shape:
                tqdm_safe_print(f"Warning: {rgb_file} - Size mismatch with SVF map ({svf_map.shape}) vs ({rgb_image.shape[:2]})")
                try:
                    if rgb_image.ndim == 3:
                        rgb_img = Image.fromarray(rgb_image)
                        rgb_resized = rgb_img.resize((svf_map.shape[1], svf_map.shape[0]), BICUBIC)
                        rgb_image = np.array(rgb_resized)
                    else:
                        rgb_img = Image.fromarray(rgb_image, mode='L')
                        rgb_image = np.array(rgb_img.resize((svf_map.shape[1], svf_map.shape[0]), BICUBIC))
                    tqdm_safe_print(f"Resized RGB image: {rgb_image.shape}")
                except Exception as resize_error:
                    tqdm_safe_print(f"Failed to resize RGB image: {str(resize_error)}")
                    rgb_image = None
        else:
            rgb_image = None
            
        if np.all(svf_map == svf_map.flatten()[0]) or np.isnan(svf_map).any():
            tqdm_safe_print(f"Warning: Invalid SVF data: {svf_file_path}")
            return {
                'success': False,
                'error': "Invalid SVF data (all same values or contains NaN)",
                'file_path': svf_file_path
            }
        
        image_path = rgb_file if os.path.exists(rgb_file) else svf_file_path
        openai_api_key = os.environ.get('OPENAI_API_KEY', None)
        
        if main_args.reconstruct_svf: # Use main_args here
            print("estimation file")
            question_generator = ConstructSVFQuestionRGB(
                estimated_svf_map=svf_map,
                estimated_height_map=height_map,
                estimated_segmentation_map=segmentation_map,
                rgb_image=rgb_image,
                file_path=svf_file_path,
                debug=debug_mode,
                use_gpt_rephrase=True,
                openai_api_key=openai_api_key,
                hard_ratio=main_args.hard_ratio,
                mode=mode,
                coordinate_answers=main_args.FT_mode,
                balanced_categories=main_args.balanced_categories
            )
        else:
            question_generator = ConstructSVFQuestionRegionBased(
                estimated_svf_map=svf_map,
                estimated_height_map=height_map,
                estimated_segmentation_map=segmentation_map,
                rgb_image=rgb_image,
                file_path=svf_file_path,
                cnt=0,
                debug=debug_mode,
                use_gpt_rephrase=True,
                openai_api_key=openai_api_key,
                hard_ratio=main_args.hard_ratio,
                mode=mode,
                coordinate_answers=main_args.FT_mode,
                balanced_categories=main_args.balanced_categories
            )
        enable_timing = debug_mode or getattr(main_args, 'quick_test', False)
        
        if enable_timing:
            file_processing_start = time.time()
            question_generation_start = time.time()
        
        question_generator.chooseQuestionsToAsk(int(5 * qa_multiplier))
        
        if enable_timing:
            question_generation_end = time.time()
            question_generation_time = question_generation_end - question_generation_start
            
            if hasattr(question_generator, 'category_timing') and question_generator.category_timing:
                for category, timing in question_generator.category_timing.items():
                    record_category_performance(category, timing)
            
            tqdm_safe_print(f"File {os.path.basename(svf_file_path)}: Question generation time={question_generation_time:.2f}s")
        
        bbox_images = []
        
        if not hasattr(process_single_file, 'plot_counter'):
            process_single_file.plot_counter = 0
            
        if not hasattr(process_single_file, 'question_id_estimate'):
            process_single_file.question_id_estimate = 0
            
        MAX_QUESTION_ID = 30
        
        is_within_id_limit = process_single_file.question_id_estimate < MAX_QUESTION_ID
            
        if not plot_limit_reached and process_single_file.plot_counter < 10 and question_generator.questions and question_generator.answers and is_within_id_limit:
            plot_dir = svf_plots
            os.makedirs(plot_dir, exist_ok=True)
            
            if mode in ["train", "both"] and process_single_file.train_file_counter >= 10:
                tqdm_safe_print(f"Plot generation completed for first 10 files in train mode. No more plots will be generated.")
            
            for i, (question_info, answer, canonical_q) in enumerate(zip(question_generator.questions, 
                                                                      question_generator.answers, 
                                                                      question_generator.canonical_questions)):
                estimated_id = process_single_file.question_id_estimate + i + 1
                
                if estimated_id > MAX_QUESTION_ID:
                    tqdm_safe_print(f"Question ID limit ({MAX_QUESTION_ID}) exceeded, skipping remaining plots")
                    break
                
                question_order = i + 1
                
                original_output_path = os.path.join(plot_dir, f"file_{process_single_file.plot_counter+1}_q{question_order}.png")
                
                if isinstance(question_info, dict) and "id" in question_info:
                    actual_question_id = question_info["id"]
                    linked_output_path = os.path.join(plot_dir, f"id_{actual_question_id}.png")
                else:
                    linked_output_path = None
                
                short_file = os.path.basename(svf_file_path)
                title_suffix = f", Question {question_order}"
                if isinstance(question_info, dict) and "id" in question_info:
                    title_suffix += f" (ID: {question_info['id']})"
                else:
                    title_suffix += f" (Est. ID: ~{estimated_id})"
                
                category = None
                if canonical_q and isinstance(canonical_q, list) and len(canonical_q) > 0:
                    category = canonical_q[0]
                    title_suffix += f", Category: {category}"
                
                title = f"File: {short_file}{title_suffix}"
                
                try:
                    plot_path = plot_correct_bbox(svf_map, question_info, answer, original_output_path, title, rgb_image=question_generator.rgb_image, category=category)
                    bbox_images.append(plot_path)
                    
                    if linked_output_path and plot_path:
                        try:
                            if os.path.exists(linked_output_path):
                                os.remove(linked_output_path)
                            
                            try:
                                os.symlink(plot_path, linked_output_path)
                            except (OSError, NotImplementedError):
                                import shutil
                                shutil.copy2(plot_path, linked_output_path)
                        except Exception as e:
                            tqdm_safe_print(f"Link/copy creation error: {str(e)}")
                except Exception as e:
                    tqdm_safe_print(f"Plot creation error: {str(e)}")
                    
            process_single_file.plot_counter += 1
        elif plot_limit_reached and debug_mode:
            tqdm_safe_print(f"Mode '{mode}' only generates plots for first 10 files. Skipping plot generation for this file.")
        
        process_single_file.question_id_estimate += len(question_generator.questions)
        
        return {
            'success': True,
            'file_path': svf_file_path,
            'area': area,
            'image_path': image_path,
            'questions': question_generator.questions,
            'answers': question_generator.answers,
            'canonical_questions': question_generator.canonical_questions,
            'bbox_images': bbox_images
        }
    except Exception as e:
        tqdm_safe_print(f"Unexpected error occurred: {str(e)}")
        import traceback
        tqdm_safe_print(traceback.format_exc())
        return {
            'success': False,
            'error': f"Unexpected error: {str(e)}",
            'file_path': svf_file_path
        }
    finally:
        force_memory_cleanup()
        if 'svf_map' in locals():
            del svf_map
        if 'height_map' in locals():
            del height_map
        if 'segmentation_map' in locals():
            del segmentation_map
        if 'rgb_image' in locals():
            del rgb_image
        if 'question_generator' in locals():
            del question_generator

def generate_point_choices(heightMap, true_point, num_choices=4, perturb_range=20, max_attempts=200):
    """
    Improved choice generation function - duplicate avoidance and boundary check enhancement
    
    Parameters:
    -----------
    heightMap : numpy.ndarray
        Height map data (used to get array size)
    true_point : tuple
        Correct answer coordinates (row, col)
    num_choices : int
        Number of choices to generate (including correct answer)
    perturb_range : int
        Perturbation range (±perturb_range)
    max_attempts : int
        Maximum number of attempts
    
    Returns:
    --------
    list
        List of choices (shuffled)
    """
    h, w = heightMap.shape
    choices = [true_point]
    attempts = 0
    
    while len(choices) < num_choices and attempts < max_attempts:
        attempts += 1
        dr = random.randint(-perturb_range, perturb_range)
        dc = random.randint(-perturb_range, perturb_range)
        new_r = min(max(0, true_point[0] + dr), h - 1)
        new_c = min(max(0, true_point[1] + dc), w - 1)
        
        new_point = (new_r, new_c)
        if new_point in choices:
            continue
        
        choices.append(new_point)
    
    if len(choices) < num_choices:
        tqdm_safe_print(f"Warning: Could not generate {num_choices} choices using normal method. Using alternative approach.")
        
        grid_points = []
        step = max(1, min(h, w) // 10)
        
        for r in range(0, h, step):
            for c in range(0, w, step):
                if (r, c) not in choices and (r, c) != true_point:
                    grid_points.append((r, c))
        
        if grid_points:
            random.shuffle(grid_points)
            choices.extend(grid_points[:num_choices - len(choices)])
    
    while len(choices) < num_choices:
        offset = len(choices)
        new_r = min(true_point[0] + offset, h - 1)
        new_c = min(true_point[1] + offset, w - 1)
        choices.append((new_r, new_c))
    
    random.shuffle(choices)
    return choices

def add_coordinate_system_explanation(question_text):
    """Add coordinate system and region specification explanation to question text"""
    coordinate_explanation = "\nNote: Coordinates are given as (Y, X) where the origin (0, 0) is at the top-left of the map. Y increases downward and X increases to the right."
    
    region_explanation = " When a region is specified as 'Region at (Y, X) with size WxH', it refers to a rectangular area with its top-left corner at coordinates (Y, X) and width W extending to the right and height H extending downward."
    
    full_explanation = coordinate_explanation + region_explanation
    
    if "Please choose from:" in question_text:
        parts = question_text.split("Please choose from:")
        return parts[0] + full_explanation + "\nPlease choose from:" + parts[1]
    else:
        return question_text + full_explanation

def save_temp_data(output_dir, temp_data, mode_name, timestamp):
    """Save temporary data"""
    temp_dir = os.path.join(output_dir, "temp_saves")
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_filename = f"temp_data_{mode_name}_{timestamp}.json"
    temp_file_path = os.path.join(temp_dir, temp_filename)
    
    with open(temp_file_path, 'w', encoding='utf-8') as f:
        json.dump(temp_data, f, ensure_ascii=False, indent=2)
    
    tqdm_safe_print(f"Saved temporary data: {temp_file_path}")
    return temp_file_path

def restore_temp_data(temp_file_path):
    """Restore temporary data"""
    if os.path.exists(temp_file_path):
        with open(temp_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def cleanup_temp_files(temp_files):
    """Clean up temporary files"""
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                tqdm_safe_print(f"Deleted: {temp_file}")
            except Exception as e:
                tqdm_safe_print(f"Deletion error: {temp_file}, {e}")

def get_last_question_id_from_jsonl(file_path):
    """Get last question ID from JSONL file"""
    if not os.path.exists(file_path):
        return 0
    
    last_id = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'question_id' in data:
                        last_id = max(last_id, data['question_id'])
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        tqdm_safe_print(f"File reading error: {file_path}, {e}")
    
    return last_id

def copy_existing_jsonl_to_new_file(source_file, dest_file):
    """Copy existing JSONL file to new file"""
    if not os.path.exists(source_file):
        tqdm_safe_print(f"Warning: Source file not found: {source_file}")
        return 0
    
    copied_count = 0
    try:
        with open(source_file, 'r', encoding='utf-8') as src, \
             open(dest_file, 'w', encoding='utf-8') as dst:
            for line in src:
                if line.strip():
                    dst.write(line)
                    copied_count += 1
        tqdm_safe_print(f"Copied {copied_count} lines from existing data: {source_file} -> {dest_file}")
    except Exception as e:
        tqdm_safe_print(f"File copy error: {source_file} -> {dest_file}, {e}")
        return 0
    
    return copied_count

def plot_correct_bbox(svf_map, question_info, answer, output_path, title, rgb_image=None, category=None):
    """Plot bounding box"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        if rgb_image is not None:
            ax.imshow(rgb_image)
        else:
            ax.imshow(svf_map, cmap='viridis')
        
        if isinstance(question_info, dict) and 'bbox' in question_info:
            bbox = question_info['bbox']
            if len(bbox) == 4:  # (x, y, width, height)
                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                       linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
        
        ax.set_title(title)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    except Exception as e:
        tqdm_safe_print(f"Plot creation error: {str(e)}")
        return None

def update_plot_files_with_ids(plot_id_mapping):
    """Apply IDs to plot files"""
    try:
        for plot_path, question_id in plot_id_mapping.items():
            if os.path.exists(plot_path):
                dir_name = os.path.dirname(plot_path)
                new_name = f"id_{question_id}.png"
                new_path = os.path.join(dir_name, new_name)
                
                if not os.path.exists(new_path):
                    import shutil
                    shutil.copy2(plot_path, new_path)
                    tqdm_safe_print(f"Applied ID to plot file: {new_path}")
    except Exception as e:
        tqdm_safe_print(f"Error applying IDs to plot files: {str(e)}")

def smart_area_sampling(all_files, sampling_ratio=0.3, min_files_per_area=2):
    """Smart area-based sampling - considering similarity of adjacent regions"""
    tqdm_safe_print(f"Starting area-based sampling: {len(all_files)} files -> approx. {int(len(all_files)*sampling_ratio)} files")
    
    area_files = {}
    for file_info in all_files:
        file_path, geonrw_path, area, phase_args, skip_plots = file_info
        if area not in area_files:
            area_files[area] = []
        area_files[area].append(file_info)
    
    sampled_files = []
    
    for area, files in area_files.items():
        area_sample_count = max(min_files_per_area, int(len(files) * sampling_ratio))
        
        if len(files) <= area_sample_count:
            sampled_files.extend(files)
        else:
            files.sort(key=lambda x: x[0])
            
            indices = np.linspace(0, len(files)-1, area_sample_count, dtype=int)
            for idx in indices:
                sampled_files.append(files[idx])
        
        tqdm_safe_print(f"Area {area}: {len(files)} → {len([f for f in sampled_files if f[2] == area])} files")
    
    tqdm_safe_print(f"Sampling complete: {len(all_files)} -> {len(sampled_files)} files ({len(sampled_files)/len(all_files)*100:.1f}%)")
    return sampled_files

category_cache = {}

def get_cached_categories(area, svf_stats=None):
    """Get/create category cache by area"""
    if area in category_cache:
        return category_cache[area]
    
    if svf_stats:
        mean_svf = np.mean(svf_stats)
        std_svf = np.std(svf_stats)
        
        suitable_categories = []
        
        if mean_svf > 0.7:
            suitable_categories.extend(['max_value', 'high_low_comparison', 'quartile_comparison'])
        elif mean_svf < 0.3:
            suitable_categories.extend(['min_value', 'low_high_comparison', 'specific_threshold'])
        else:
            suitable_categories.extend(['average_calculation', 'specific_coordinate', 'range_identification'])
            
        if std_svf > 0.2:
            suitable_categories.extend(['high_low_comparison', 'range_identification'])
        
        category_cache[area] = suitable_categories
        tqdm_safe_print(f"Created category cache for area {area}: {len(suitable_categories)} types")
        return suitable_categories
    
    return None

_bias_free_random = None

def setup_bias_free_shuffling():
    """Set up bias prevention for choice shuffling"""
    global _bias_free_random
    import time
    choice_random_seed = int(time.time() * 1000) % 2147483647
    _bias_free_random = random.Random(choice_random_seed)
    tqdm_safe_print(f"Choice bias prevention: initialized with independent seed {choice_random_seed}")

def export_category_text_files_from_qa_data(qa_data, output_dir="category_text_output", limit_per_category=100):
    """
    Export category-wise text files from QA data
    
    Args:
        qa_data: List of QA data
        output_dir: Output directory
        limit_per_category: Maximum output items per category
    
    Returns:
        List of exported files
    """
    import os
    import re
    from collections import defaultdict, Counter
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    
    category_data = defaultdict(list)
    
    for item in qa_data:
        category = item.get('category', 'unknown')
        category_data[category].append(item)
    
    print(f"Exporting category-wise text files... (max {limit_per_category} items per category)")
    
    exported_files = []
    for category, items in category_data.items():
        limited_items = items[:limit_per_category] if limit_per_category else items
        
        safe_category = re.sub(r'[<>:"/\\|?*]', '_', category)
        file_path = os.path.join(output_dir, f"{safe_category}.txt")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"\n" + "=" * 80 + "\n")
            f.write(f"Category: {category}\n")
            f.write(f"Exported items: {len(limited_items)} / Total {len(items)} items\n")
            f.write(f"Export time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, item in enumerate(limited_items, 1):
                question_id = item.get('question_id', i)
                question_text = item.get('text', '')
                answer = item.get('answer', '')
                image = item.get('image', '')
                choices = item.get('choices', [])
                
                f.write(f"Q{i:03d} (ID: {question_id})\n")
                f.write("-" * 40 + "\n")
                
                f.write("Question:\n")
                f.write(f"{question_text}\n\n")
                
                if choices:
                    f.write("Choices:\n")
                    for j, choice in enumerate(choices, 1):
                        f.write(f"  {j}. {choice}\n")
                    f.write("\n")
                
                f.write(f"Answer: {answer}\n")
                
                if image:
                    f.write(f"Image: {image}\n")
                
                f.write("\n" + "=" * 60 + "\n\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Summary for category '{category}'\n")
            f.write("=" * 80 + "\n")
            f.write(f"Exported items: {len(limited_items)} / Total {len(items)} items\n")
            f.write(f"Number of images used: {len(set(item.get('image', '') for item in limited_items if item.get('image')))}\n")
            
            answer_types = Counter()
            for item in limited_items:
                answer = item.get('answer', '')
                if answer in ['Yes', 'No']:
                    answer_types['Yes/No'] += 1
                elif answer.startswith('Region'):
                    answer_types['Region'] += 1
                elif answer.startswith('grid_'):
                    answer_types['Grid'] += 1
                elif re.match(r'^\d+\.\d+$', answer):
                    answer_types['Numeric'] += 1
                elif ',' in answer:
                    answer_types['Multi-choice'] += 1
                else:
                    answer_types['Other'] += 1
            
            f.write(f"Answer type distribution: {dict(answer_types)}\n")
            f.write(f"Export completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        exported_files.append(file_path)
        print(f"{category}: Exported {len(limited_items)} items to {file_path}")
    
    summary_file = os.path.join(output_dir, "_category_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("Category-wise Text File Export Summary (via run.py)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Export time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total QA data: {len(qa_data)}\n")
        f.write(f"Number of categories: {len(category_data)}\n")
        f.write(f"Limit per category: {limit_per_category} items\n")
        f.write(f"Number of exported files: {len(exported_files)}\n\n")
        
        f.write("Questions per category (exported / total):\n")
        for category, items in sorted(category_data.items(), key=lambda x: len(x[1]), reverse=True):
            limited_count = min(len(items), limit_per_category) if limit_per_category else len(items)
            f.write(f"  {category}: {limited_count} / {len(items)} items\n")
        
        f.write("\nExported file list:\n")
        for file_path in sorted(exported_files):
            file_name = os.path.basename(file_path)
            f.write(f"  {file_name}\n")
    
    exported_files.append(summary_file)
    
    print(f"\nCategory-wise text file export completed!")
    print(f"Output directory: {output_dir}")
    print(f"Summary file: {summary_file}")
    print(f"Total files: {len(exported_files)}")
    
    return exported_files
    
def bias_free_shuffle(choices_list):
    """Independent random shuffle for bias prevention"""
    global _bias_free_random
    if _bias_free_random is None:
        setup_bias_free_shuffling()
    _bias_free_random.shuffle(choices_list)
    return choices_list

def clear_category_cache():
    """Clear category cache"""
    global category_cache
    category_cache.clear()

def main():
    # Initialize global seed manager for reproducibility
    # Seed can be set via --seed argument or RANDOM_SEED environment variable
    setup_bias_free_shuffling()
    
    parser = argparse.ArgumentParser(description="Script to generate questions and answers from SVF data")
    parser.add_argument("--svf_path", type=str,
                        help="Path to root directory containing SVF files")
    parser.add_argument("--geonrw_path", type=str,
                        help="Path to GeoNRW data root (height, segmentation, RGB)")
    parser.add_argument("--mode", type=str, choices=["train", "test", "both"], default="both",
                      help="Processing mode: 'train', 'test', or 'both'")
    parser.add_argument("--question_types", type=str, default="all",
                      help="Question types to generate (comma-separated, or 'all')")
    parser.add_argument("--visual_hints", action="store_true",
                      help="Add visual hints to questions")
    parser.add_argument("--simplify", action="store_true",
                      help="Simplify questions and answers")
    parser.add_argument("--output_dir", type=str, default="cs_qa_0718",
                      help="Output directory path (default: 'svf_qa_output')")
    parser.add_argument("--num_processes", type=int, default=0,
                      help="Number of processes to use (default: 0=auto)")
    parser.add_argument("--batch_size", type=int, default=8,
                      help="Batch size (default: 10)")
    parser.add_argument("--max_files", type=int, default=3000,
                      help="Maximum number of files to process (default: 3000, 0=all)")
    parser.add_argument("--update_plot_ids", action="store_true",
                      help="Apply question IDs to generated plot files")
    parser.add_argument("--conversation", action="store_true", default=True,
                      help="Output as conversation format JSON")
    parser.add_argument("--FT-mode", action="store_true", default=False,
                      help="Fine-tuning mode: Generate coordinate-based answers (e.g., 'Region A: [24%, 8%, 48%, 32%]') instead of choice-based answers")
    parser.add_argument("--skip_plots", action="store_true", default=True,
                      help="Skip plot generation")
    parser.add_argument("--debug", action="store_true", default=False,
                      help="Enable debug mode")
    parser.add_argument("--skip_detail_files", action="store_true", default=False,
                      help="Skip detailed file generation (recommended for train mode)")
    parser.add_argument("--with_hints", action="store_true",
                      help="Generate output files with hints")
    parser.add_argument("--with_svf_array", action="store_true",
                      help="Generate output files with SVF array data")
    parser.add_argument("--with_svf_values", action="store_true",
                      help="Generate output files with SVF values for choices")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed value (default: 42)")
    parser.add_argument("--reconstruct_svf", default=False, action="store_true",
                        help="Reconstruct SVF values")
    parser.add_argument('--qa_multiplier', type=float, default=15, 
                       help='QA count multiplier (default 15=75 QAs, increased for speed)')
    parser.add_argument('--hard_ratio', type=float, default=0.3,
                       help='Ratio of hard questions (0.0-1.0, default: 0.0)')
    parser.add_argument("--versions", nargs='+', default=["standard"],
                      help="Dataset versions to generate ('standard', 'medium', 'large', 'all', 'small')")
    parser.add_argument("--balanced_categories", action="store_true", default=False,
                      help="Generate questions evenly from each category (1 question per category per image)")
    parser.add_argument("--resume_from", type=str, default=None,
                      help="File path to resume processing from temporary save file")
    parser.add_argument("--continue_from", type=str, default=None,
                      help="File path to continue from existing JSONL file (e.g., svf_qa_output_simple/svf_answers_train_0606.jsonl)")
    parser.add_argument("--force_single_process", action="store_true", default=False,
                      help="Run in single process mode (for memory issues)")
    parser.add_argument("--run_bias_evaluation", action="store_true", default=False,
                        help="Automatically run bias evaluation after QA generation")
    parser.add_argument("--enable_smart_sampling", action="store_true", default=True,
                        help="Enable area-based smart sampling (for speed)")
    parser.add_argument("--sampling_ratio", type=float, default=0.4,
                        help="Area sampling ratio (0.1-1.0, default 0.4=40%)")
    parser.add_argument("--quick_test", action="store_true", default=False,
                        help="Quick test: fast test with few files (max_files=5, qa_multiplier=1)")
    parser.add_argument("--test_hard_ranking", action="store_true", default=False,
                        help="Test hard_ranking: generate few with hard_ratio=0.5")
    parser.add_argument("--micro_test", action="store_true", default=False,
                        help="Micro test: fastest test with 1 file 1 question")
    parser.add_argument("--append_existing", action="store_true", default=False,
                        help="Append to existing files")
    args = parser.parse_args()
    
    if args.quick_test:
        args.max_files = 5
        args.qa_multiplier = 1
        # args.debug = True
        args.skip_detail_files = False
        args.enable_smart_sampling = False
        tqdm_safe_print("Quick test mode: max_files=5, qa_multiplier=1")
    
    if args.test_hard_ranking:
        args.max_files = 5
        args.qa_multiplier = 2
        args.hard_ratio = 0.5
        args.debug = True
        args.skip_detail_files = False
        args.enable_smart_sampling = False
        tqdm_safe_print("hard_ranking test mode: max_files=5, qa_multiplier=2, hard_ratio=0.5, debug=True")
    
    if args.micro_test:
        args.max_files = 1
        args.qa_multiplier = 0.2
        args.hard_ratio = 1.0
        args.debug = True
        args.skip_detail_files = False
        args.enable_smart_sampling = False
        tqdm_safe_print("Micro test mode: max_files=1, qa_multiplier=0.2, hard_ratio=1.0, debug=True")
    
    is_standard_version_only = len(args.versions) == 1 and args.versions[0] == 'standard'
    if not is_standard_version_only and '--max_files' not in sys.argv and not args.quick_test and not args.test_hard_ranking and not args.micro_test:
        tqdm_safe_print(f"Info: Version {args.versions} specified, setting max_files=0 (all files) to generate target question count.")
        args.max_files = 0
    
    if args.mode == "train" and not args.skip_detail_files:
        if '--skip_detail_files' not in sys.argv:
            args.skip_detail_files = True
            tqdm_safe_print("Skipping detailed file generation in train mode (for efficiency)")
    
    if args.quick_test or args.test_hard_ranking or args.micro_test:
        args.skip_detail_files = False
    else:
        if '--max_files' in sys.argv and 0 < args.max_files < 3000:
            if args.skip_detail_files:
                args.skip_detail_files = False
                tqdm_safe_print(f"max_files={args.max_files} specified, generating detailed files (for debugging/verification)")
        else:
            tqdm_safe_print("max_files not specified, skipping detailed file generation.")
            args.skip_detail_files = True
    # Set global seed using centralized seed manager
    set_global_seed(args.seed)
    global_seed = get_global_seed()
    tqdm_safe_print(f"Random seed value: {global_seed} (set via --seed argument or RANDOM_SEED env var)")
    
    start_time = time.time()
    
    # Use command-line arguments instead of hardcoded paths
    # This ensures reproducibility and flexibility
    if not args.svf_path:
        raise ValueError("--svf_path must be provided. Hardcoded paths have been removed for reproducibility.")
    if not args.geonrw_path:
        raise ValueError("--geonrw_path must be provided. Hardcoded paths have been removed for reproducibility.")
    
    svf_base_path = args.svf_path
    geonrw_path = args.geonrw_path
    
    # Determine train/test paths based on mode and provided paths
    # If svf_path points to a specific train/test directory, use it directly
    # Otherwise, construct paths based on the base path
    if args.mode == "train":
        svf_train = args.svf_path if os.path.isdir(args.svf_path) else os.path.join(args.svf_path, "train")
        svf_test = None
    elif args.mode == "test":
        svf_test = args.svf_path if os.path.isdir(args.svf_path) else os.path.join(args.svf_path, "test")
        svf_train = None
    else:  # both
        # Try to detect train/test subdirectories
        base_dir = args.svf_path
        train_candidate = os.path.join(base_dir, "train")
        test_candidate = os.path.join(base_dir, "test")
        if os.path.isdir(train_candidate) and os.path.isdir(test_candidate):
            svf_train = train_candidate
            svf_test = test_candidate
        else:
            # Fallback: use the same directory for both (user should specify separate paths)
            svf_train = args.svf_path
            svf_test = args.svf_path
            tqdm_safe_print("Warning: Could not detect train/test subdirectories. Using the same path for both.")
    
    tqdm_safe_print("Starting processing...")
    tqdm_safe_print(f"SVF base path: {svf_base_path}")
    tqdm_safe_print(f"GeoNRW path: {geonrw_path}")
    
    qa_data_list = []
    
    category_text_buffer = {}
    category_text_dir = None
    realtime_text_counter = 0
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    tqdm_safe_print(f"Creating/verifying output directory: {output_dir}")

    version_config = {
        "standard": {"train": None, "test": None},
        "medium": {"train": 12000, "test": 5000},
        "large": {"train": 150000, "test": 15000},
        "xl": {"train": 200000, "test": 20000},
        "small": {"train": 7000, "test": 1000}
    }
    
    if "all" in args.versions:
        versions_to_process = ["standard", "medium", "large"]
    else:
        versions_to_process = args.versions
    
    NUM_CATEGORIES = 13
    tqdm_safe_print(f"\n=== Version-wise Calculation Results ===")
    for ver, config in version_config.items():
        if ver in versions_to_process:
            tqdm_safe_print(f"\n{ver.upper()}")
            for mode_name, target_count in config.items():
                if target_count is None:
                    tqdm_safe_print(f"  {mode_name}: No limit (existing behavior)")
                else:
                    required_images = math.ceil(target_count / NUM_CATEGORIES)
                    questions_per_image = NUM_CATEGORIES
                    tqdm_safe_print(f"  {mode_name}: Target {target_count} questions ÷ {NUM_CATEGORIES} categories = {required_images} images needed")
                    tqdm_safe_print(f"    → {questions_per_image} questions per image planned")
    tqdm_safe_print("=" * 35)
    
    modes_to_process = []
    if args.mode == "train":
        if svf_train is None:
            raise ValueError("Train path is not set. Please check --svf_path argument.")
        modes_to_process.append(("train", svf_train))
    elif args.mode == "test":
        if svf_test is None:
            raise ValueError("Test path is not set. Please check --svf_path argument.")
        modes_to_process.append(("test", svf_test))
    elif args.mode == "both":
        if svf_train is None or svf_test is None:
            raise ValueError("Both train and test paths must be set for 'both' mode. Please check --svf_path argument.")
        modes_to_process.append(("train", svf_train))
        modes_to_process.append(("test", svf_test))
    
    for version in versions_to_process:
        if version not in version_config:
            tqdm_safe_print(f"Warning: Configuration for version '{version}' not found. Skipping.")
            continue
        
        tqdm_safe_print(f"\n===== Starting {version.upper()} VERSION generation =====")
        
        total_question_count = 0
        total_files_processed = 0
        total_files_count = 0
        
        plot_id_mapping = {}
        
        for mode_name, svf_path in modes_to_process:
            tqdm_safe_print(f"\nStarting {mode_name.upper()} data processing...")
            
            processing_phases = []
            
            if args.hard_ratio > 0:
                phase_name = f"mixed_hr{args.hard_ratio}"
                phase_label = f'Mixed questions (Hard {args.hard_ratio*100:.0f}%)'
            else:
                phase_name = 'standard'
                phase_label = 'Standard questions'
                
            processing_phases.append({
                'phase_name': phase_name,
                'hard_ratio': args.hard_ratio,
                'phase_label': phase_label
            })
            
            total_question_count_mode = 0
            total_files_processed_mode = 0
            
            for phase in processing_phases:
                phase_name = phase['phase_name']
                phase_hard_ratio = phase['hard_ratio']
                phase_label = phase['phase_label']
                
                tqdm_safe_print(f"\n--- Starting {phase_label} phase ---")
                
                date = datetime.now().strftime("%m%d")
                
                if args.qa_multiplier > 1:
                    output_prefix = f"svf_{args.qa_multiplier}x"
                else:
                    output_prefix = "svf"
    
                if version != "standard":
                    output_prefix = f"{output_prefix}_{version}"
                
                max_files_suffix = ""
                if '--max_files' in sys.argv and args.max_files > 0:
                    max_files_suffix = f"_max{args.max_files}"
    
                if args.hard_ratio > 0:
                    hard_suffix = f"_hr{args.hard_ratio}"
                else:
                    hard_suffix = ""
                
                base_question_file = os.path.join(output_dir, f"{output_prefix}_questions_{mode_name}_{phase_name}_{date}{hard_suffix}{max_files_suffix}.jsonl")
                base_answer_file = os.path.join(output_dir, f"{output_prefix}_answers_{mode_name}_{phase_name}_{date}{hard_suffix}{max_files_suffix}.jsonl")
                base_detailed_file = os.path.join(output_dir, f"{output_prefix}_detailed_{mode_name}_{phase_name}_{date}{hard_suffix}{max_files_suffix}.json") if not args.skip_detail_files else None
                
                def handle_existing_file(file_path):
                    if file_path is None:
                        return None
                    if not os.path.exists(file_path):
                        return file_path
                    if args.append_existing:
                        return file_path
                    else:
                        counter = 1
                        base_name, ext = os.path.splitext(file_path)
                        while os.path.exists(f"{base_name}_{counter}{ext}"):
                            counter += 1
                        return f"{base_name}_{counter}{ext}"
                
                question_file = handle_existing_file(base_question_file)
                answer_file = handle_existing_file(base_answer_file)
                detailed_file = handle_existing_file(base_detailed_file)
                
                conversation_file = os.path.join(output_dir, f"{output_prefix}_conversation_{mode_name}_{phase_name}_{date}{hard_suffix}{max_files_suffix}.json") if args.conversation else None
                
                hints_file = os.path.join(output_dir, f"{output_prefix}_with_hints_{mode_name}_{phase_name}_{date}{hard_suffix}{max_files_suffix}.json") if args.with_hints else None
                
                array_file = os.path.join(output_dir, f"{output_prefix}_with_svf_array_{mode_name}_{phase_name}_{date}{hard_suffix}{max_files_suffix}.json") if args.with_svf_array else None
                
                values_file = os.path.join(output_dir, f"{output_prefix}_with_svf_values_{mode_name}_{phase_name}_{date}{hard_suffix}{max_files_suffix}.json") if args.with_svf_values else None
                
                all_files = []
                for area in os.listdir(svf_path):
                    area_path = os.path.join(svf_path, area)
                    if not os.path.isdir(area_path):
                        continue
                        
                    svf_files = [f for f in os.listdir(area_path) if f.endswith('_svf_umep.tif')]
                    
                    phase_args = argparse.Namespace(**vars(args))
                    phase_args.hard_ratio = phase_hard_ratio
                    
                    all_files.extend([(os.path.join(area_path, f), geonrw_path, area, phase_args, args.skip_plots) for f in svf_files])
    
                if args.enable_smart_sampling and len(all_files) > 50:
                    all_files = smart_area_sampling(all_files, 
                                                   sampling_ratio=args.sampling_ratio, 
                                                   min_files_per_area=2)
                    
                if args.max_files > 0:
                    random.shuffle(all_files)
                    all_files = all_files[:args.max_files]
    
                target_question_count = version_config[version][mode_name]
                if target_question_count is not None:
                    if phase_name == 'hard':
                        target_question_count = int(target_question_count * args.hard_ratio)
                    elif args.hard_ratio > 0:
                        target_question_count = int(target_question_count * (1.0 - args.hard_ratio))
                    
                    if len(all_files) > 0:
                        if args.balanced_categories:
                            ideal_images_needed = math.ceil(target_question_count / NUM_CATEGORIES)
                            if len(all_files) >= ideal_images_needed:
                                questions_per_file = NUM_CATEGORIES
                                images_to_use = ideal_images_needed
                                tqdm_safe_print(f"Sufficient images ({len(all_files)}) → Ideal distribution")
                            else:
                                questions_per_file = math.ceil(target_question_count / len(all_files))
                                images_to_use = len(all_files)
                                tqdm_safe_print(f"Warning: Insufficient images ({len(all_files)}) → Dynamic adjustment")
                            
                            phase_qa_multiplier = questions_per_file / 5.0
                            tqdm_safe_print(f"Info: {version.upper()} {mode_name.upper()} mode (balanced)")
                            tqdm_safe_print(f"  Target questions: {target_question_count}, Categories: {NUM_CATEGORIES}")
                            tqdm_safe_print(f"  Available images: {len(all_files)}, Images to use: {images_to_use}")
                            tqdm_safe_print(f"  Questions per image: {questions_per_file} (qa_multiplier: {phase_qa_multiplier:.2f})")
                        else:
                            questions_per_file = max(1, (target_question_count + len(all_files) - 1) // len(all_files))
                            phase_qa_multiplier = questions_per_file / 5.0
                            tqdm_safe_print(f"Info: {version.upper()} {mode_name.upper()} mode question limit is {target_question_count}.")
                            tqdm_safe_print(f"Available files: {len(all_files)}, Questions per file: {questions_per_file}")
                            tqdm_safe_print(f"=> Adjusting qa_multiplier to {phase_qa_multiplier:.2f}.")
                    else:
                        phase_qa_multiplier = args.qa_multiplier
                else:
                    phase_qa_multiplier = args.qa_multiplier
                
                tqdm_safe_print(f"{phase_label} - Total files to process: {len(all_files)}")
                
                for file in [question_file, answer_file, detailed_file, conversation_file, hints_file, array_file, values_file]:
                    if file and os.path.exists(file):
                        os.remove(file)
                
                if args.continue_from and os.path.exists(args.continue_from) and phase_name == 'standard':
                    tqdm_safe_print(f"Continuing from existing file: {args.continue_from}")
                    
                    last_id = get_last_question_id_from_jsonl(args.continue_from)
                    question_id_counter = last_id
                    tqdm_safe_print(f"Resume start ID: {question_id_counter + 1}")
                    
                    copy_existing_jsonl_to_new_file(args.continue_from, answer_file)
                    
                    continue_question_file = args.continue_from.replace('_answers_', '_questions_')
                    if os.path.exists(continue_question_file):
                        copy_existing_jsonl_to_new_file(continue_question_file, question_file)
                    
                    continue_detailed_file = args.continue_from.replace('_answers_', '_detailed_').replace('.jsonl', '.json')
                    if os.path.exists(continue_detailed_file):
                        try:
                            with open(continue_detailed_file, 'r', encoding='utf-8') as src:
                                existing_detailed_data = json.load(src)
                            tqdm_safe_print(f"Copied {len(existing_detailed_data) if isinstance(existing_detailed_data, list) else 1} existing detailed data items: {continue_detailed_file}")
                        except Exception as e:
                            tqdm_safe_print(f"Detailed file copy error: {continue_detailed_file}, {e}")
                    
                    continue_conversation_file = args.continue_from.replace('_answers_', '_conversation_').replace('.jsonl', '.json')
                    if os.path.exists(continue_conversation_file) and args.conversation:
                        try:
                            with open(continue_conversation_file, 'r', encoding='utf-8') as src:
                                existing_conversation_data = json.load(src)
                            if isinstance(existing_conversation_data, list):
                                conversation_data.extend(existing_conversation_data)
                            else:
                                conversation_data.append(existing_conversation_data)
                            tqdm_safe_print(f"Copied {len(existing_conversation_data) if isinstance(existing_conversation_data, list) else 1} existing conversation data items: {continue_conversation_file}")
                        except Exception as e:
                            tqdm_safe_print(f"Conversation file copy error: {continue_conversation_file}, {e}")
                    
                else:
                    if phase_name == 'standard':
                        question_id_counter = 0
                        tqdm_safe_print(f"[DEBUG] First phase '{phase_name}': question_id_counter = 0")
                    else:
                        question_id_counter = total_question_count_mode
                        tqdm_safe_print(f"[DEBUG] Continuing phase '{phase_name}': question_id_counter = {total_question_count_mode}")
                    
                    tqdm_safe_print(f"[DEBUG] Phase: {phase_name}, total_question_count_mode: {total_question_count_mode}, question_id_counter: {question_id_counter}")
                
                tqdm_safe_print(f"Question ID start value: {question_id_counter}")
                
                num_processes = args.num_processes if args.num_processes > 0 else max(1, cpu_count() - 1)
                num_processes = min(num_processes, cpu_count())
                tqdm_safe_print(f"Number of processes to use: {num_processes}")
                
                if args.force_single_process:
                    num_processes = 1
                    tqdm_safe_print("force_single_process option specified, running in single process mode")
                else:
                    import psutil
                    memory = psutil.virtual_memory()
                    memory_usage_percent = memory.percent
                    
                    if memory_usage_percent > 90:
                        num_processes = 1
                        tqdm_safe_print(f"High memory usage ({memory_usage_percent:.1f}%), limiting to 1 process")
                    elif memory_usage_percent > 80:
                        num_processes = min(num_processes, 2)
                        tqdm_safe_print(f"High memory usage ({memory_usage_percent:.1f}%), limiting to {num_processes} processes")
                    elif memory_usage_percent > 70:
                        num_processes = min(num_processes, 4)
                        tqdm_safe_print(f"Memory usage ({memory_usage_percent:.1f}%), limiting to {num_processes} processes")
                
                tqdm_safe_print(f"Final number of processes: {num_processes}")
                
                files_processed = 0
                conversation_data = []
                file_plot_mapping = []
                temp_save_counter = 0
                temp_file_list = []
                SAVE_INTERVAL = 500
                
                try:
                    from functools import partial
                    date_time = datetime.now().strftime("%m%d%H%M")
                    svf_plots = f"svf_plots_{phase_name}_{date_time}"
                    
                    process_file_with_plots = partial(process_single_file, 
                                                      svf_plots=svf_plots, 
                                                      qa_multiplier=phase_qa_multiplier)
                    
                    try:
                        import psutil
                        available_memory_gb = psutil.virtual_memory().available / (1024**3)
                        if available_memory_gb > 8:
                            chunk_size = min(1000, max(200, len(all_files) // 3))
                        elif available_memory_gb > 4:
                            chunk_size = min(500, max(100, len(all_files) // 5))
                        else:
                            chunk_size = min(200, max(50, len(all_files) // 10))
                    except ImportError:
                        chunk_size = min(500, max(100, len(all_files) // 5))
                    
                    tqdm_safe_print(f"Speed optimization: Processing file list in large chunks of {chunk_size} files")
                    
                    with Pool(processes=num_processes) as pool:
                        for chunk_start in range(0, len(all_files), chunk_size):
                            chunk_end = min(chunk_start + chunk_size, len(all_files))
                            file_chunk = all_files[chunk_start:chunk_end]
                            
                            chunk_num = chunk_start // chunk_size + 1
                            total_chunks = (len(all_files) + chunk_size - 1) // chunk_size
                            tqdm_safe_print(f"=== Starting chunk {chunk_num}/{total_chunks} ===")
                            
                            try:
                                import psutil
                                memory = psutil.virtual_memory()
                                if memory.percent > 90:
                                    tqdm_safe_print(f"Warning: High memory usage ({memory.percent:.1f}%), executing memory cleanup")
                                    force_memory_cleanup()
                                    time.sleep(2)
                                elif memory.percent > 95:
                                    tqdm_safe_print(f"CRITICAL: Memory usage reached critical level ({memory.percent:.1f}%), pausing for memory cleanup")
                                    force_memory_cleanup()
                                    time.sleep(5)
                            except ImportError:
                                pass
                            
                            for result in tqdm(pool.imap_unordered(process_file_with_plots, file_chunk), 
                                              total=len(file_chunk), 
                                                  desc=f"{mode_name} ({version}) {phase_label} chunk{chunk_num}"):
                                if result['success']:
                                    if 'questions' not in result or 'answers' not in result or 'canonical_questions' not in result:
                                        tqdm_safe_print(f"Warning: Missing required keys in result: {result['file_path']}")
                                        continue
                                        
                                    if 'bbox_images' in result and result['bbox_images']:
                                        file_basename = os.path.basename(result['file_path'])
                                        local_mapping = []
                                        
                                        for i, (plot_path, q, a, cat) in enumerate(zip(
                                                    result['bbox_images'],
                                                    result['questions'][:len(result['bbox_images'])],
                                                    result['answers'][:len(result['bbox_images'])],
                                                    result['canonical_questions'][:len(result['bbox_images'])]
                                                )):
                                            if plot_path:
                                                local_mapping.append({
                                                    'plot_path': plot_path,
                                                    'file_basename': file_basename,
                                                    'question_index': i,
                                                        'next_question_id': question_id_counter + 1 + i
                                                })
                                        
                                        if local_mapping:
                                            file_plot_mapping.extend(local_mapping)
                                            
                                    for q, a, cat in zip(result['questions'], result['answers'], result['canonical_questions']):
                                        question_id_counter += 1
                                        temp_save_counter += 1
                                        
                                        if 'area' in result and 'image_path' in result:
                                            image_path = f"{result['area']}/{os.path.basename(result['image_path'])}"
                                        else:
                                            image_path = result['file_path']
                                        
                                        data_ans = {
                                            "question_id": question_id_counter,
                                            "image": image_path,
                                            "answer": a,
                                            "text": q["question"] if isinstance(q, dict) else q,
                                            "category": cat[0] if isinstance(cat, list) and len(cat) > 0 else "unknown",
                                            "phase": phase_name
                                        }
                                        
                                        data_ques = {
                                            "question_id": question_id_counter,
                                            "image": image_path,
                                            "text": q["question"] if isinstance(q, dict) else q,
                                            "category": cat[0] if isinstance(cat, list) and len(cat) > 0 else "unknown",
                                            "phase": phase_name
                                        }
                                        
                                        data_detailed = {
                                            "question_id": question_id_counter,
                                            "image": image_path,
                                            "answer": a,
                                            "text": q["question"] if isinstance(q, dict) else q,
                                            "category": cat[0] if isinstance(cat, list) and len(cat) > 0 else "unknown",
                                            "debug_info": q.get("debug_info", []) if isinstance(q, dict) else [],
                                            "phase": phase_name
                                        }
                                        
                                        if isinstance(q, dict) and "scores" in q:
                                            data_detailed["scores"] = q["scores"]
                                            
                                        if isinstance(q, dict) and "question" in q:
                                            q_text = q["question"]
                                            choices_section = q_text.split("Please choose from:")
                                            if len(choices_section) > 1:
                                                choices_text = choices_section[1].strip()
                                                choices_lines = [line.strip() for line in choices_text.split("\n") if line.strip()]
                                                data_detailed["choices"] = choices_lines
                                        
                                        conv_data = None
                                        if args.conversation:
                                            conv_data = {
                                                    "id": f"{version}-{phase_name}-{str(question_id_counter).zfill(12)}",
                                                "image": image_path,
                                                "conversations": [
                                                    {
                                                        "from": "human",
                                                        "value": f"<image>\n{q['question'] if isinstance(q, dict) else q}"
                                                    },
                                                    {
                                                        "from": "gpt",
                                                        "value": str(a)
                                                    }
                                                    ],
                                                    "metadata": {
                                            "category": cat[0] if isinstance(cat, list) and len(cat) > 0 else "unknown",
                                                    "mode": mode_name,
                                                    "version": version,
                                                    "phase": phase_name
                                        }
                                            }
                                            conversation_data.append(conv_data)
                                        
                                        for mapping in file_plot_mapping:
                                            if mapping.get('next_question_id') == question_id_counter:
                                                mapping['question_id'] = question_id_counter
                                                plot_id_mapping[mapping['plot_path']] = question_id_counter
                                        
                                        if isinstance(q, dict) and "choices" in q:
                                            data_ans["choices"] = q["choices"]
                                            data_ques["choices"] = q["choices"]
                                            data_detailed["choices"] = q["choices"]
                                        
                                        should_flush = add_to_batch_buffer(data_ans, data_ques, data_detailed, conv_data)
                                        
                                        if len(qa_data_list) < 100:
                                            qa_data_list.append({
                                                'question': data_detailed.get('text', ''),
                                                'answer': data_detailed.get('answer', ''),
                                                'category': data_detailed.get('category', 'unknown'),
                                                'question_id': data_detailed.get('question_id', question_id_counter),
                                                'choices': data_detailed.get('choices', []),
                                                'image': data_detailed.get('image', '')
                                            })
                                            
                                            category = data_detailed.get('category', 'unknown')
                                            if category not in category_text_buffer:
                                                category_text_buffer[category] = []
                                            
                                            text_entry = f"Q{question_id_counter:03d}: {data_detailed.get('text', '')[:50]}...{data_detailed.get('choices', '')}  {data_detailed.get('answer', '')}\n"
                                            category_text_buffer[category].append(text_entry)
                                            
                                            realtime_text_counter += 1
                                        
                                        if should_flush:
                                            flush_batch_buffer(answer_file, question_file, detailed_file, conversation_file)
                                        
                                        if realtime_text_counter >= 50 and category_text_buffer:
                                            if category_text_dir is None:
                                                category_text_dir = Path(output_dir) / "category_text_realtime"
                                                category_text_dir.mkdir(exist_ok=True)
                                            
                                            for cat, entries in category_text_buffer.items():
                                                if entries:
                                                    cat_file = category_text_dir / f"{cat}.txt"
                                                    with open(cat_file, 'a', encoding='utf-8') as f:
                                                        f.writelines(entries)
                                            tqdm_safe_print(f"Real-time category text write: {category_text_dir}")
                                            category_text_buffer.clear()
                                            realtime_text_counter = 0
                                        
                                        if temp_save_counter >= 500:
                                            temp_save_counter = 0
                                            tqdm_safe_print(f"{phase_label} progress: {question_id_counter} questions completed - Data already saved to {answer_file}")
                                        
                                    files_processed += 1
                                else:
                                    tqdm_safe_print(f"\nError occurred: {result['file_path']}")
                                    tqdm_safe_print(f"Error details: {result['error']}")
                        
                        if batch_buffer['buffer_size'] > 0:
                            flush_batch_buffer(answer_file, question_file, detailed_file, conversation_file)
                        
                        if category_text_buffer and category_text_dir:
                            for cat, entries in category_text_buffer.items():
                                if entries:
                                    cat_file = category_text_dir / f"{cat}.txt"
                                    with open(cat_file, 'a', encoding='utf-8') as f:
                                        f.writelines(entries)
                            category_text_buffer.clear()
                            tqdm_safe_print(f"Real-time category text completed: {category_text_dir}")
                        
                        if args.conversation and conversation_data:
                            with open(conversation_file, 'w', encoding='utf-8') as f:
                                json.dump(conversation_data, f, ensure_ascii=False, indent=2)
                            tqdm_safe_print(f"Saved conversation format data: {conversation_file}")
                        
                        tqdm_safe_print(f"All data saved. Detailed data is appended every 500 QAs.")
    
                except KeyboardInterrupt:
                    tqdm.write(f"\n\n=== Interrupting {phase_label} phase processing... ===")
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    interrupt_data = {
                        'mode': mode_name,
                        'version': version,
                        'phase': phase_name,
                        'question_count': question_id_counter,
                        'files_processed': files_processed,
                        'conversation_data': conversation_data if args.conversation else [],
                        'timestamp': timestamp,
                        'interrupt_reason': 'KeyboardInterrupt'
                    }
                    
                    interrupt_file = save_temp_data(output_dir, interrupt_data, f"{mode_name}_{phase_name}_interrupted", timestamp)
                    tqdm_safe_print(f"Saved interrupt data: {interrupt_file}")
                    
                    pool.terminate()
                    pool.join()
                    
                    sys.exit(1)
                
                if temp_file_list:
                    cleanup_temp_files(temp_file_list)
                
                phase_question_count = question_id_counter - total_question_count_mode
                total_question_count_mode = question_id_counter
                total_files_processed_mode += files_processed
                
                tqdm_safe_print(f"\n--- {phase_label} phase completed ---")
                tqdm_safe_print(f"Files processed: {files_processed}/{len(all_files)}")
                tqdm_safe_print(f"Questions generated: {phase_question_count}")
                tqdm_safe_print(f"Output files:")
                tqdm_safe_print(f"- Questions: {question_file}")
                tqdm_safe_print(f"- Answers: {answer_file}")
                if not args.skip_detail_files and detailed_file:
                    tqdm_safe_print(f"- Detailed: {detailed_file}")
                if args.conversation and conversation_file:
                    tqdm_safe_print(f"- Conversation: {conversation_file}")
                
                total_question_count += total_question_count_mode
                total_files_processed += total_files_processed_mode
                
                if args.hard_ratio > 0 and len(processing_phases) > 1:
                    tqdm_safe_print(f"\n--- Starting phase integration file generation ---")
                    
                    date = datetime.now().strftime("%m%d")
                    
                    if args.qa_multiplier > 1:
                        output_prefix = f"svf_{args.qa_multiplier}x"
                    else:
                        output_prefix = "svf"
                    
                    if version != "standard":
                        output_prefix = f"{output_prefix}_{version}"
                    
                    combined_suffix = f"_combined"
                    
                    combined_question_file = os.path.join(output_dir, f"{output_prefix}_questions_{mode_name}_{date}{combined_suffix}_hr{args.hard_ratio}{max_files_suffix}.jsonl")
                    combined_answer_file = os.path.join(output_dir, f"{output_prefix}_answers_{mode_name}_{date}{combined_suffix}_hr{args.hard_ratio}{max_files_suffix}.jsonl")
                    combined_detailed_file = os.path.join(output_dir, f"{output_prefix}_detailed_{mode_name}_{date}{combined_suffix}_hr{args.hard_ratio}{max_files_suffix}.json") if not args.skip_detail_files else None
                    combined_conversation_file = os.path.join(output_dir, f"{output_prefix}_conversation_{mode_name}_{date}{combined_suffix}_hr{args.hard_ratio}{max_files_suffix}.json") if args.conversation else None
                    
                    try:
                        standard_files = {}
                        hard_files = {}
                        
                        for phase in processing_phases:
                            phase_name = phase['phase_name']
                            
                            if phase_name == 'standard':
                                hard_suffix_phase = f"_standard_only"
                            elif phase_name == 'hard':
                                hard_suffix_phase = f"_hard_only"
                            else:
                                continue
                            
                            phase_files = {
                                'question': os.path.join(output_dir, f"{output_prefix}_questions_{mode_name}_{phase_name}_{date}{hard_suffix_phase}{max_files_suffix}.jsonl"),
                                'answer': os.path.join(output_dir, f"{output_prefix}_answers_{mode_name}_{phase_name}_{date}{hard_suffix_phase}{max_files_suffix}.jsonl"),
                                'detailed': os.path.join(output_dir, f"{output_prefix}_detailed_{mode_name}_{phase_name}_{date}{hard_suffix_phase}{max_files_suffix}.json") if not args.skip_detail_files else None,
                                'conversation': os.path.join(output_dir, f"{output_prefix}_conversation_{mode_name}_{phase_name}_{date}{hard_suffix_phase}{max_files_suffix}.json") if args.conversation else None
                            }
                            
                            if phase_name == 'standard':
                                standard_files = phase_files
                            elif phase_name == 'hard':
                                hard_files = phase_files
                        
                        def combine_jsonl_files(output_file, *input_files):
                            """Combine multiple JSONL files"""
                            with open(output_file, 'w', encoding='utf-8') as outf:
                                for input_file in input_files:
                                    if input_file and os.path.exists(input_file):
                                        with open(input_file, 'r', encoding='utf-8') as inf:
                                            for line in inf:
                                                outf.write(line)
                        
                        def combine_json_files(output_file, *input_files):
                            """Combine multiple JSON files"""
                            combined_data = []
                            for input_file in input_files:
                                if input_file and os.path.exists(input_file):
                                    with open(input_file, 'r', encoding='utf-8') as inf:
                                        try:
                                            data = json.load(inf)
                                            if isinstance(data, list):
                                                combined_data.extend(data)
                                            else:
                                                combined_data.append(data)
                                        except json.JSONDecodeError as e:
                                            tqdm_safe_print(f"JSON reading error {input_file}: {e}")
                            
                            with open(output_file, 'w', encoding='utf-8') as outf:
                                json.dump(combined_data, outf, ensure_ascii=False, indent=2)
                        
                        tqdm_safe_print("Merging question files...")
                        combine_jsonl_files(combined_question_file, standard_files.get('question'), hard_files.get('question'))
                        
                        tqdm_safe_print("Merging answer files...")
                        combine_jsonl_files(combined_answer_file, standard_files.get('answer'), hard_files.get('answer'))
                        
                        if not args.skip_detail_files and combined_detailed_file:
                            tqdm_safe_print("Merging detailed files...")
                            combine_json_files(combined_detailed_file, standard_files.get('detailed'), hard_files.get('detailed'))
                        
                        if args.conversation and combined_conversation_file:
                            tqdm_safe_print("Merging conversation files...")
                            combine_json_files(combined_conversation_file, standard_files.get('conversation'), hard_files.get('conversation'))
                        
                        tqdm_safe_print(f"\n--- Phase integration file generation completed ---")
                        tqdm_safe_print(f"Integrated files:")
                        tqdm_safe_print(f"- Questions: {combined_question_file}")
                        tqdm_safe_print(f"- Answers: {combined_answer_file}")
                        if not args.skip_detail_files and combined_detailed_file:
                            tqdm_safe_print(f"- Detailed: {combined_detailed_file}")
                        if args.conversation and combined_conversation_file:
                            tqdm_safe_print(f"- Conversation: {combined_conversation_file}")
                        
                        def count_lines_in_jsonl(file_path):
                            """Count lines in JSONL file"""
                            if os.path.exists(file_path):
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    return sum(1 for _ in f)
                            return 0
                        
                        combined_question_count = count_lines_in_jsonl(combined_question_file)
                        combined_answer_count = count_lines_in_jsonl(combined_answer_file)
                        
                        tqdm_safe_print(f"Integration result: Questions={combined_question_count}, Answers={combined_answer_count}")
                        
                    except Exception as e:
                        tqdm_safe_print(f"Phase integration error: {str(e)}")
                        import traceback
                        tqdm_safe_print(traceback.format_exc())
            
            if args.update_plot_ids and plot_id_mapping:
                update_plot_files_with_ids(plot_id_mapping)
            
            tqdm_safe_print(f"\nAll phases of {mode_name.upper()} data processing completed")
            tqdm_safe_print(f"Total files processed: {total_files_processed_mode}")
            tqdm_safe_print(f"Total questions generated: {total_question_count_mode}")
    
        end_time = time.time()
        processing_time = end_time - start_time
        
        tqdm_safe_print("\n=== All processing completed ===")
        tqdm_safe_print(f"Processing time: {processing_time:.2f}s ({processing_time/60:.2f} minutes)")
        if processing_time > 0:
            tqdm_safe_print(f"Average generation speed: {total_question_count/processing_time:.2f} questions/sec")
            tqdm_safe_print(f"Average file processing speed: {total_files_processed/processing_time:.2f} files/sec")
            tqdm_safe_print(f"Please refer to the above for output files for each version")
        
        if args.debug or getattr(args, 'quick_test', False):
            print_performance_summary()
    
        try:
            if qa_data_list:
                all_qa_data = qa_data_list
                
                if all_qa_data:
                    print("\n" + "="*60)
                    print(f"           Category-wise text file export ({len(all_qa_data)} items)")
                    print("="*60)
                    
                    text_output_dir = Path(output_dir) / "category_text_files_first100"
                    exported_text_files = export_category_text_files_from_qa_data(
                        all_qa_data, 
                        str(text_output_dir), 
                        limit_per_category=50
                    )
                    
                    print(f"\nCategory-wise text file export completed!")
                    print(f"Output directory: {text_output_dir}")
                    print(f"Number of files: {len(exported_text_files)}")
                else:
                    print("Warning: No QA data available for category-wise text export")
        except Exception as e:
            print(f"Warning: Error occurred during category-wise text file export: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
        
        if args.run_bias_evaluation:
            print("\n" + "="*60)
            print("           Starting bias evaluation")
            print("="*60)
            
            try:
                from automated_qa_bias_evaluation import AutomatedQABiasEvaluator
                
                qa_files_for_evaluation = []
                output_path = Path(output_dir)
                
                file_patterns = [
                    "*_detailed_*.json",
                    "*_answers_*.jsonl",
                    "*_questions_*.jsonl",
                    "*_conversation_*.json",
                ]
                
                print(f"Searching for QA files... (output directory: {output_dir})")
                
                for pattern in file_patterns:
                    matching_files = list(output_path.glob(pattern))
                    if matching_files:
                        print(f"  Pattern '{pattern}': Found {len(matching_files)} files")
                        qa_files_for_evaluation.extend([str(f) for f in matching_files])
                
                qa_files_for_evaluation = list(set(qa_files_for_evaluation))
                
                if qa_files_for_evaluation:
                    bias_output_dir = output_path / "bias_evaluation"
                    evaluator = AutomatedQABiasEvaluator(
                        output_dir=str(bias_output_dir),
                        verbose=args.debug
                    )
                    
                    print(f"\nNumber of files to evaluate: {len(qa_files_for_evaluation)}")
                    for i, file_path in enumerate(qa_files_for_evaluation):
                        if i < 10:
                            print(f"  - {os.path.basename(file_path)}")
                        elif i == 10:
                            print(f"  ... {len(qa_files_for_evaluation) - 10} more files")
                            break
                    
                    print("\nRunning bias evaluation...")
                    evaluator.run_full_evaluation(qa_files_for_evaluation)
                    
                    print(f"\nBias evaluation results saved to: {bias_output_dir}")
                    
                else:
                    print("Warning: No QA files found for evaluation")
                    print(f"   Searched directory: {output_dir}")
                    print(f"   Available files:")
                    all_files = list(output_path.glob("*"))
                    for file_path in all_files[:10]:
                        print(f"     - {file_path.name}")
                    if len(all_files) > 10:
                        print(f"     ... {len(all_files) - 10} more files")
                    
            except ImportError as e:
                print(" Could not import bias evaluation module")
                print(f"   Error details: {e}")
                print("   Please ensure automated_qa_bias_evaluation.py exists and all required dependencies are installed")
            except Exception as e:
                print(f" Error occurred during bias evaluation: {e}")
                import traceback
                if args.debug:
                    print("Error details:")
                    traceback.print_exc()
                else:
                    print("   Use --debug option to see detailed error information")
    
if __name__ == "__main__":
    main()

