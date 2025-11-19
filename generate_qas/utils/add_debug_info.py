import json
import argparse
from tqdm import tqdm
import os

def add_debug_info_to_answer(answer, debug_info, level='simple'):
    """
    Add debug information to the answer text with different levels of detail.
    
    Args:
        answer (str): Original answer text
        debug_info (dict): Debug information dictionary
        level (str): Level of detail ('simple', 'medium', 'full')
    
    Returns:
        str: Answer text with added debug information
    """
    if not debug_info:
        return f"<final short answer>{answer}</final short answer>"
        
    if level == 'simple':
        # Add only scores
        if 'scores' in debug_info:
            explanation = "Based on the scores for each choice:\n"
            for idx, score in debug_info['scores'].items():
                if 'choices' in debug_info and int(idx) < len(debug_info['choices']):
                    choice = debug_info['choices'][int(idx)]
                    explanation += f"{choice}: {score:.3f}\n"
            return f"{explanation}\n<final short answer>{answer}</final short answer>"
            
    elif level == 'medium':
        # Add scores and main metrics
        if 'scores' in debug_info:
            explanation = "Based on the scores and main metrics for each choice:\n"
            for idx, score in debug_info['scores'].items():
                if 'choices' in debug_info and int(idx) < len(debug_info['choices']):
                    choice = debug_info['choices'][int(idx)]
                    explanation += f"{choice}:\n"
                    explanation += f"  Total Score: {score:.3f}\n"
                    
                    # Add main metrics if available
                    if 'debug_info' in debug_info and int(idx) < len(debug_info['debug_info']):
                        details = debug_info['debug_info'][int(idx)]
                        if isinstance(details, str) and 'Details:' in details:
                            details_dict = eval(details.split('Details:')[1].strip())
                            for key, value in details_dict.items():
                                if key in ['svf_raw', 'bcr_raw', 'far_est_raw']:
                                    explanation += f"  {key}: {value:.3f}\n"
            
            return f"{explanation}\n<final short answer>{answer}</final short answer>"
            
    else:  # full
        # Add complete debug information
        if 'debug_info' in debug_info:
            explanation = "Detailed analysis of each choice:\n"
            for idx, details in enumerate(debug_info['debug_info']):
                if 'choices' in debug_info and idx < len(debug_info['choices']):
                    choice = debug_info['choices'][idx]
                    explanation += f"\n{choice}:\n{details}\n"
            return f"{explanation}\n<final short answer>{answer}</final short answer>"
    
    return f"<final short answer>{answer}</final short answer>"

def process_file(input_file, output_files, levels):
    """
    Process a JSONL file and add debug information to answers.
    
    Args:
        input_file (str): Path to input JSONL file
        output_files (dict): Dictionary of output file paths for each level
        levels (list): List of detail levels to process
    """
    # Open all output files
    file_handles = {}
    for level in levels:
        file_handles[level] = open(output_files[level], 'w', encoding='utf-8')
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            for line in tqdm(f_in, desc=f"Processing {input_file}"):
                data = json.loads(line)
                
                # Process each level
                for level in levels:
                    data_copy = data.copy()
                    if 'debug_info' in data:
                        data_copy['answer'] = add_debug_info_to_answer(
                            data['answer'],
                            data,
                            level
                        )
                    
                    # Write modified data to output file
                    file_handles[level].write(json.dumps(data_copy, ensure_ascii=False) + '\n')
    
    finally:
        # Close all output files
        for handle in file_handles.values():
            handle.close()

def main():
    parser = argparse.ArgumentParser(description='Add debug information to answers in JSONL files')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Input directory containing JSONL files')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for modified JSONL files')
    parser.add_argument('--levels', type=str, default='simple,medium,full',
                      help='Comma-separated list of detail levels to process')
    
    args = parser.parse_args()
    
    # Parse levels
    levels = [level.strip() for level in args.levels.split(',')]
    valid_levels = ['simple', 'medium', 'full']
    levels = [level for level in levels if level in valid_levels]
    
    if not levels:
        print("No valid levels specified. Using default: simple")
        levels = ['simple']
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process all JSONL files in input directory
    for filename in os.listdir(args.input_dir):
        if filename.endswith('.jsonl'):
            input_path = os.path.join(args.input_dir, filename)
            
            # Create output paths for each level
            output_files = {}
            for level in levels:
                output_path = os.path.join(
                    args.output_dir,
                    f"{os.path.splitext(filename)[0]}_{level}_debug.jsonl"
                )
                output_files[level] = output_path
            
            process_file(input_path, output_files, levels)
            print(f"Processed {filename} -> {[os.path.basename(path) for path in output_files.values()]}")

if __name__ == "__main__":
    main() 