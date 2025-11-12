import argparse
import os
from datetime import datetime
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import openai
from openai import OpenAI
import json
import time
from PIL import Image, ImageFile
import base64

from qwen3_svf_qa import dsm_to_rgb, svf_to_rgb, seg_to_rgb, get_related_image_paths

ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_json_data(file_path):
    """Load question data from JSON file"""
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

def encode_image_to_base64(image_path):
    """Encode image to Base64"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Image encoding error: {e}, path: {image_path}")
        return None

def convert_jp2_to_jpeg(image_path, force_convert=False):
    """Convert JP2 image to JPEG"""
    input_path = Path(image_path)
    if not input_path.suffix.lower() in ['.jp2', '.jpx', '.j2k', '.jpc']:
        return str(input_path)
    output_filename = input_path.stem + '.jpg'
    output_path = os.path.join(input_path.parent, output_filename)
    if os.path.exists(output_path) and not force_convert:
        return output_path
    try:
        with Image.open(input_path) as img:
            if img.mode in ['RGBA', 'LA', 'P']:
                rgb_img = img.convert('RGB')
                rgb_img.save(output_path, 'JPEG', quality=95)
            else:
                img.save(output_path, 'JPEG', quality=95)
        return output_path
    except Exception as e:
        print(f"JP2->JPEG conversion error: {e}, path: {image_path}")
        return str(input_path)

def prepare_multimodal_images(image_paths, modalities, dsm_colormap='terrain', svf_colormap='plasma', temp_dir=None):
    """Prepare images for each modality and save as temporary files"""
    processed_images = {}
    temp_files = []
    
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    
    for modality in modalities:
        if modality not in image_paths or not image_paths[modality]:
            continue
            
        image_path = image_paths[modality]
        processed_image = None
        
        try:
            if modality == "rgb":
                converted_path = convert_jp2_to_jpeg(image_path)
                if os.path.exists(converted_path):
                    processed_images[modality] = converted_path
                    continue
                    
            elif modality == "dsm":
                processed_image = dsm_to_rgb(image_path, colormap=dsm_colormap)
                
            elif modality == "svf":
                processed_image = svf_to_rgb(image_path, colormap=svf_colormap)
                
            elif modality == "seg":
                processed_image = seg_to_rgb(image_path)
            
            if processed_image is not None:
                base_name = Path(image_path).stem
                temp_filename = f"{modality}_{base_name}_{int(time.time())}.png"
                temp_path = os.path.join(temp_dir, temp_filename)
                processed_image.save(temp_path, 'PNG')
                processed_images[modality] = temp_path
                temp_files.append(temp_path)
                print(f"  Temporarily saved {modality} image: {temp_path}")
                
        except Exception as e:
            print(f"  Error processing {modality} image: {e}, path: {image_path}")
            continue
    
    return processed_images, temp_files

def get_geonrw_class_names():
    """Get GeoNRW segmentation class names"""
    return {
        0: "Background", 1: "Forest", 2: "Water", 3: "Agricultural",
        4: "Residential/Commercial/Industrial", 5: "Grassland/Swamp/Shrubbery",
        6: "Railway/Train station", 7: "Highway/Squares", 8: "Airport/Shipyard",
        9: "Roads", 10: "Buildings"
    }

def get_geonrw_class_colors():
    """Get GeoNRW segmentation class color mapping"""
    return {
        0: [0, 0, 0], 1: [0, 100, 0], 2: [0, 0, 255], 3: [255, 255, 0],
        4: [128, 128, 128], 5: [144, 238, 144], 6: [165, 42, 42],
        7: [192, 192, 192], 8: [255, 165, 0], 9: [128, 128, 0], 10: [255, 0, 0]
    }

def generate_modality_description(modalities, dsm_colormap='terrain', svf_colormap='plasma'):
    """Generate detailed description text based on used modalities"""
    descriptions = []
    
    if "rgb" in modalities:
        descriptions.append("RGB: Standard aerial/satellite RGB image showing natural colors.")
    
    if "dsm" in modalities:
        descriptions.append(
            f"DSM: Digital Surface Model (elevation data) converted to RGB using '{dsm_colormap}' colormap. "
            f"Blue/green colors = low elevation, yellow/brown colors = high elevation. "
            f"This visualization helps identify terrain features and building heights."
        )
    
    if "svf" in modalities:
        descriptions.append(
            f"SVF: Sky View Factor (openness measure 0-1) converted to RGB using '{svf_colormap}' colormap. "
            f"Dark blue/purple colors (0.0-0.3) = very low SVF (heavily enclosed areas with minimal sky visibility), "
            f"light blue colors (0.3-0.5) = low SVF (partially enclosed areas), "
            f"green/teal colors (0.5-0.7) = moderate SVF (semi-open areas), "
            f"yellow colors (0.7-0.9) = high SVF (mostly open areas), "
            f"red colors (0.9-1.0) = very high SVF (completely open areas with maximum sky visibility). "
            f"Higher values indicate more open spaces with fewer obstructions to the sky."
        )
    if "seg" in modalities:
        class_names = get_geonrw_class_names()
        colors = get_geonrw_class_colors()
        
        seg_desc = ("SEG: Land use segmentation map with fixed colors representing different land cover classes. "
                   "Color coding: ")
        color_mappings = []
        for cls in range(1, 11):
            rgb = colors[cls]
            name = class_names[cls]
            color_mappings.append(f"{name}=RGB{tuple(rgb)}")
        
        seg_desc += "; ".join(color_mappings)
        descriptions.append(seg_desc)
    
    if len(descriptions) > 1:
        modality_desc = "**Image Analysis Guide:**\nThe following images are provided for analysis:\n" + \
                       "\n".join([f"• {desc}" for desc in descriptions])
    elif len(descriptions) == 1:
        modality_desc = f"**Image Analysis Guide:**\n• {descriptions[0]}"
    else:
        modality_desc = ""
    
    return modality_desc

def prepare_messages(image_paths, text, modalities, dsm_colormap='terrain', svf_colormap='plasma', temp_dir=None):
    """Prepare messages for GPT-4 (with multimodal image processing)"""
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
                print(f"Warning: Processed image file does not exist: {image_path}")
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
    only_rgb = len(actual_modalities) == 1 and actual_modalities[0] == "rgb"
    if modality_desc and not only_rgb:
        enhanced_text = (
            f"{modality_desc}\n\n**Question:** {text}\n\n"
            "Please provide only the final short answer to the question without any other text."
        )
    else:
        enhanced_text = text + " Please provide only the final short answer to the question without any other text."
    content.append({
        "type": "text", 
        "text": enhanced_text
    })
    
    message = {
        "role": "user",
        "content": content
    }
    
    return message, temp_files
def is_free_input(questions_file):
    if "free" in questions_file:
        return True
    return False
def process_batch(client, batch_messages, args):
    """Send questions to GPT-4 in batch processing"""
    output_texts = []
    
    for i, (messages, temp_files) in enumerate(batch_messages):
        print(f"  Processing question {i+1}/{len(batch_messages)} in batch...")
        
        try:
            system_message = {
                "role": "system",
                "content": (
                    "You are an expert in landscape analysis. "
                    "Analyze the provided images which may include RGB, DSM (Digital Surface Model), "
                    "SVF (Sky View Factor), and segmentation maps. "
                    "Answer the question concisely and provide brief reasoning if possible."
                )
            }

            full_messages = [system_message] + messages

            # For GPT-5 models, use Responses API to avoid empty outputs with chat.completions
            if 'gpt-5' in args.model:
                def to_responses_input(chat_messages):
                    """Convert chat-style messages into Responses API input blocks."""
                    responses_msgs = []
                    for msg in chat_messages:
                        role = msg.get("role", "user")
                        content_blocks = []
                        content = msg.get("content")
                        # system content may be a plain string
                        if isinstance(content, str):
                            content_blocks.append({"type": "input_text", "text": content})
                        else:
                            for block in content:
                                btype = block.get("type")
                                if btype == "text":
                                    content_blocks.append({"type": "input_text", "text": block.get("text", "")})
                                elif btype == "image_url":
                                    # Accept both {"image_url": {"url": "..."}} and {"image_url": "..."}
                                    image_field = block.get("image_url")
                                    if isinstance(image_field, dict):
                                        image_url = image_field.get("url", "")
                                    else:
                                        image_url = image_field or ""
                                    if image_url:
                                        content_blocks.append({"type": "input_image", "image_url": image_url})
                        responses_msgs.append({"role": role, "content": content_blocks})
                    return responses_msgs

                responses_input = to_responses_input(full_messages)
                resp = client.responses.create(
                    model=args.model,
                    input=responses_input,
                    max_output_tokens=args.max_new_tokens
                )
                # Prefer output_text if available
                content = getattr(resp, 'output_text', '') or ''
                if not content:
                    # Fallback: try to join first output content blocks
                    try:
                        first_output = resp.output[0]
                        parts = first_output.content if hasattr(first_output, 'content') else []
                        texts = [p.text for p in parts if getattr(p, 'type', '') == 'output_text' and hasattr(p, 'text')]
                        content = "\n".join(texts).strip()
                    except Exception:
                        content = ""
                content = content.strip()
            else:
                chat_kwargs = dict(
                    model=args.model,
                    messages=full_messages,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                # o4-mini models use max_completion_tokens
                if 'o4-mini' in args.model:
                    if is_free_input(args.questions_file):
                        pass
                    else:
                        chat_kwargs['max_completion_tokens'] = args.max_new_tokens
                else:
                    chat_kwargs['max_tokens'] = args.max_new_tokens
                    chat_kwargs['temperature'] = args.temperature if args.temperature > 0 else 0
                if args.temperature > 0:
                    chat_kwargs["top_p"] = 0.95

                response = client.chat.completions.create(**chat_kwargs)
                content = response.choices[0].message.content.strip()
                print("content: ", content)
        except Exception as e:
            print(f"    Error occurred during generation: {e}")
            content = f"Error: Model generation failed - {str(e)}"
        
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Error deleting temporary file: {e}, file: {temp_file}")
        
        output_texts.append(content)
    
    return output_texts

def save_results(results, output_file):
    """Save results to JSONL file"""
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'a', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def run_single_test(args):
    """Run a single test execution"""
    # Use secure API key management
    from utils.api_key_manager import get_openai_api_key, validate_input_path, validate_output_path
    
    try:
        api_key = get_openai_api_key()
    except ValueError as e:
        # Sanitize error message to prevent key leakage
        raise ValueError(str(e))
    
    openai.api_key = api_key
    client = OpenAI(api_key=api_key)
    
    # Validate input paths
    if args.questions_file:
        validate_input_path(args.questions_file, "file")
    if args.image_dir:
        validate_input_path(args.image_dir, "directory")
    if args.output_file:
        validate_output_path(args.output_file)
    
    print(f"Loading question data: {args.questions_file}")
    questions = load_json_data(args.questions_file)
    print(f"Loaded {len(questions)} questions")
    
    dsm_dir = args.dsm_dir or args.image_dir
    svf_dir = args.svf_dir or args.image_dir
    seg_dir = args.seg_dir or args.image_dir
    
    if args.use_all_modalities:
        modalities = ["rgb", "dsm", "svf", "seg"]
    else:
        modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    print(f"Using modalities: {modalities}")
    
    batch_size = args.batch_size
    question_slice = questions[:args.limit] if args.limit > 0 else questions
    results = []
    
    temp_dir = tempfile.mkdtemp()
    print(f"Temporary directory: {temp_dir}")

    output_file = args.output_file
    if os.path.exists(output_file):
        base, ext = os.path.splitext(output_file)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{base}_new_{timestamp}{ext}"
        print(f"Existing output_file found, renamed to: {output_file}")
        args.output_file = output_file

    try:
        for i in range(0, len(question_slice), batch_size):
            batch_questions = question_slice[i:i+batch_size]
            batch_messages = []
            batch_question_info = []
            
            print(f"Processing batch {(i//batch_size)+1}/{(len(question_slice)-1)//batch_size+1}...")
            
            for question in batch_questions:
                question_id = question.get("question_id", 0)
                image_rel_path = question.get("image", "")
                text = question.get("text", "")
                category = question.get("category", "unknown")
                
                rgb_path = os.path.join(args.image_dir, image_rel_path)
                image_paths = get_related_image_paths(rgb_path, dsm_dir, svf_dir, seg_dir)
                
                print(f"  Question ID {question_id}: Preparing image paths...")
                message, temp_files = prepare_messages(
                    image_paths, text, modalities, 
                    dsm_colormap=args.dsm_colormap,
                    svf_colormap=args.svf_colormap,
                    temp_dir=temp_dir
                )
                
                batch_messages.append(([message], temp_files))
                batch_question_info.append({
                    "question_id": question_id,
                    "image_rel_path": image_rel_path,
                    "text": text,
                    "category": category,
                    "used_modalities": modalities
                })
            
            batch_results = process_batch(client, batch_messages, args)
            
            for idx, output_text in enumerate(batch_results):
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

            if len(results) >= 500:
                print(f"Saving intermediate progress to {args.output_file} ({len(results)} items)")
                save_results(results, args.output_file)
                results.clear()
        
        if results:
            print(f"Saving final results to {args.output_file} ({len(results)} items)")
            save_results(results, args.output_file)
            print(f"Processing complete. Generated {len(results)} answers.")
        else:
            print("Warning: No questions were processed. Results were not saved.")
    
    finally:
        try:
            import shutil
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Error cleaning up temporary directory: {e}")

def main(args):
    if args.single_modality_test:
        all_modalities = ["rgb", "dsm", "svf", "seg"]
        
        if args.modality_combinations:
            combinations = args.modality_combinations.split(',')
        else:
            combinations = all_modalities + ["all"]
        
        original_output_file = args.output_file
        original_questions_file = args.questions_file
        
        print(f"Single modality test mode: Testing {len(combinations)} combinations")
        print(f"Test combinations: {combinations}")
        
        for i, combo in enumerate(combinations):
            print(f"\n=== Test {i+1}/{len(combinations)}: {combo} ===")
            
            if combo == "all":
                args.use_all_modalities = True
                args.modalities = ",".join(all_modalities)
                modality_info = "all_"
            else:
                args.use_all_modalities = False
                args.modalities = combo.replace('+', ',')
                modality_info = f"{args.modalities.replace(',', '_')}_"
            
            if original_output_file:
                base_path = Path(original_output_file)
                args.output_file = str(base_path.parent / f"{base_path.stem}_{modality_info.rstrip('_')}{base_path.suffix}")
            else:
                today = datetime.now().strftime("%Y%m%d")
                input_filename = os.path.basename(args.questions_file)
                input_basename = os.path.splitext(input_filename)[0]
                model_name = args.model.replace("-", "_").replace(".", "_")
                args.output_file = f"{input_basename}_{modality_info}{today}_{model_name}_output.json"
                if not os.path.isabs(args.output_file):
                    args.output_file = os.path.join(os.path.dirname(args.questions_file), args.output_file)
            
            print(f"Using modalities: {args.modalities}")
            print(f"Output file: {args.output_file}")
            
            run_single_test(args)
        
        print(f"\n=== Single modality test complete: Tested {len(combinations)} combinations ===")
        return
    
    run_single_test(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal landscape QA batch inference using GPT-4o/4.1")
    parser.add_argument('--questions_file', type=str, default="svf_qa/svf_combined_3000q_free.jsonl", help="Path to JSON file containing question data")
    parser.add_argument("--image_dir", type=str, default="GeoNRW/", help="Base directory for RGB image files")
    parser.add_argument("--dsm_dir", type=str, default="GeoNRW/", help="Directory for DSM data")
    parser.add_argument("--svf_dir", type=str, default="svf/skyview_umep_test/", help="Directory for SVF data")
    parser.add_argument("--seg_dir", type=str, default="GeoNRW/", help="Directory for segmentation data")
    parser.add_argument("--use_all_modalities", action="store_true", help="Use all modalities")
    parser.add_argument("--modalities", type=str, default="rgb", help="Comma-separated list of modalities to use (e.g., rgb,dsm,svf)")
    parser.add_argument("--dsm_colormap", type=str, default="terrain", help="Colormap for DSM (e.g., terrain, jet, viridis)")
    parser.add_argument("--svf_colormap", type=str, default="plasma", help="Colormap for SVF (e.g., plasma, viridis, inferno)")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature parameter for generation")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--output_file", type=str, default=None, help="Output JSON file path (auto-generated if not specified)")
    parser.add_argument("--limit", type=int, default=-1, help="Maximum number of questions to process (-1 for all)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (number of questions to process simultaneously)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use (e.g., gpt-4o, o4-mini, gpt-4o-mini, gpt-4.1-nano)")
    parser.add_argument("--force_convert", action="store_true", help="Force reconversion even if JPEG images already exist")
    parser.add_argument("--single_modality_test", action="store_true", help="Test each modality individually")
    parser.add_argument("--modality_combinations", type=str, default=None, help="Comma-separated list of modality combinations to test (e.g., rgb,dsm,rgb+dsm,all)")
    
    args = parser.parse_args()
    
    if not args.single_modality_test and not args.output_file:
        today = datetime.now().strftime("%Y%m%d")
        input_filename = os.path.basename(args.questions_file)
        input_basename = os.path.splitext(input_filename)[0]
        modality_info = ""
        if args.modalities and not args.use_all_modalities:
            modality_info = f"{args.modalities.replace(',', '_')}_"
        model_name = args.model.replace("-", "_").replace(".", "_")
        args.output_file = f"{input_basename}_{modality_info}{today}_{model_name}_output.json"
        if not os.path.isabs(args.output_file):
            args.output_file = os.path.join(os.path.dirname(args.questions_file), args.output_file)
    
    print(f"Input file: {args.questions_file}")
    print(f"Using model: {args.model}")
    if not args.single_modality_test:
        print(f"Output file will be: {args.output_file}")
    
    main(args) 