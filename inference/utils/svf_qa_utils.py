
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
