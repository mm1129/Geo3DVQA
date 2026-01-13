#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一時保存ファイルからデータを復元するためのヘルパースクリプト
run.pyがKeyboardInterruptで中断された場合に使用します。
"""

import os
import json
import argparse
from datetime import datetime

def restore_temp_data(temp_file_path, output_dir):
    """一時ファイルからデータを復元"""
    if not os.path.exists(temp_file_path):
        print(f"Error: Temporary file not found: {temp_file_path}")
        return False
    
    try:
        with open(temp_file_path, 'r', encoding='utf-8') as f:
            temp_data = json.load(f)
        
        print(f"Loaded temporary file: {temp_file_path}")
        print(f"Mode: {temp_data.get('mode', 'unknown')}")
        print(f"Version: {temp_data.get('version', 'unknown')}")
        print(f"Question count: {temp_data.get('question_count', 0)}")
        print(f"Files processed: {temp_data.get('files_processed', 0)}")
        print(f"Timestamp: {temp_data.get('timestamp', 'unknown')}")
        
        # 復元ファイルの作成
        mode_name = temp_data.get('mode', 'unknown')  
        version = temp_data.get('version', 'standard')
        timestamp = temp_data.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        # 会話形式データの復元
        if temp_data.get('conversation_data'):
            conversation_file = os.path.join(output_dir, f"restored_conversation_{mode_name}_{timestamp}.json")
            with open(conversation_file, 'w', encoding='utf-8') as f:
                json.dump(temp_data['conversation_data'], f, ensure_ascii=False, indent=2)
            print(f"Restored conversation data: {conversation_file}")
        
        # 統計情報ファイルの作成
        stats_file = os.path.join(output_dir, f"restored_stats_{mode_name}_{timestamp}.json")
        stats_data = {
            'restoration_info': {
                'restored_from': temp_file_path,
                'restoration_time': datetime.now().isoformat(),
                'original_timestamp': temp_data.get('timestamp'),
                'interrupt_reason': temp_data.get('interrupt_reason', 'unknown')
            },
            'processing_stats': {
                'mode': temp_data.get('mode'),
                'version': temp_data.get('version'),  
                'question_count': temp_data.get('question_count', 0),
                'files_processed': temp_data.get('files_processed', 0)
            }
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2)
        print(f"Saved statistics: {stats_file}")
        
        return True
        
    except Exception as e:
        print(f"Restoration error: {e}")
        return False

def list_temp_files(output_dir):
    """利用可能な一時ファイルをリスト表示"""
    temp_dir = os.path.join(output_dir, "temp_saves")
    
    if not os.path.exists(temp_dir):
        print(f"Temporary save directory not found: {temp_dir}")
        return []
    
    temp_files = []
    for filename in os.listdir(temp_dir):
        if filename.startswith('temp_data_') and filename.endswith('.json'):
            file_path = os.path.join(temp_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                temp_files.append({
                    'path': file_path,
                    'filename': filename,
                    'mode': data.get('mode', 'unknown'),
                    'version': data.get('version', 'unknown'),
                    'question_count': data.get('question_count', 0),
                    'timestamp': data.get('timestamp', 'unknown'),
                    'interrupt_reason': data.get('interrupt_reason', 'normal_save')
                })
            except Exception as e:
                print(f"File read error {filename}: {e}")
    
    return temp_files

def main():
    parser = argparse.ArgumentParser(description="一時保存ファイルからデータを復元")
    parser.add_argument("--output_dir", type=str, default="svf_qo_re0625",
                       help="出力ディレクトリのパス")
    parser.add_argument("--temp_file", type=str,
                       help="復元する一時ファイルのパス")
    parser.add_argument("--list", action="store_true",
                       help="利用可能な一時ファイルをリスト表示")
    
    args = parser.parse_args()
    
    if args.list:
        print("\n=== Available Temporary Save Files ===")
        temp_files = list_temp_files(args.output_dir)
        
        if not temp_files:
            print("No temporary save files found.")
            return
        
        for i, temp_file in enumerate(temp_files, 1):
            print(f"\n{i}. {temp_file['filename']}")
            print(f"   Path: {temp_file['path']}")
            print(f"   Mode: {temp_file['mode']}")
            print(f"   Version: {temp_file['version']}")
            print(f"   Question count: {temp_file['question_count']}")
            print(f"   Timestamp: {temp_file['timestamp']}")
            print(f"   Interrupt reason: {temp_file['interrupt_reason']}")
        
        print(f"\nTo restore, run:")
        print(f"python restore_from_temp.py --output_dir {args.output_dir} --temp_file <path>")
        
    elif args.temp_file:
        print(f"Restoring data from temporary file: {args.temp_file}")
        success = restore_temp_data(args.temp_file, args.output_dir)
        
        if success:
            print("Restoration completed successfully")
        else:
            print("Failed to restore data")
    else:
        print("Please specify --list or --temp_file option")
        print("For help: python restore_from_temp.py --help")

if __name__ == "__main__":
    main() 