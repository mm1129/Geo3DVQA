"""
Freeform Analysis QA Generator

This module provides a QA generator that uses FreeformAnalysisCategories
to generate questions and answers based on comprehensive scene analysis.
"""

import numpy as np
import random
import json
from typing import List, Dict, Any, Optional, Tuple
from .freeform_analysis_categories import FreeformAnalysisCategories
from .bias_free_utils import bias_free_shuffle


class FreeformQAGenerator:
    """
    QA Generator using FreeformAnalysisCategories for comprehensive scene analysis
    """
    
    def __init__(self, debug: bool = False, use_gpt4: bool = True, 
                 api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize the Freeform QA Generator
        
        Args:
            debug: Enable debug output
            use_gpt4: Enable GPT-4 powered answer generation
            api_key: OpenAI API key (optional, uses environment variable if None)
            model: GPT model to use
        """
        self.debug = debug
        self.use_gpt4 = use_gpt4
        self.api_key = api_key
        self.model = model
        
        # Categories available for analysis
        self.analysis_categories = [
            'urban_development_application',
            'renewable_energy_installation', 
            'landscape_analysis',
            'water_accumulation'
        ]
        
        if self.debug:
            print(f" FreeformQAGenerator initialized with GPT-4: {self.use_gpt4}")
            print(f" Available categories: {self.analysis_categories}")
    
    def generate_qa_from_scene(self, scene_data: Dict[str, Any], 
                              target_categories: Optional[List[str]] = None, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate QA pairs from scene data using freeform analysis
        
        Args:
            scene_data: Scene data dictionary with SVF, DSM, SEG, RGB data
            target_categories: Specific categories to focus on (None for all)
            output_path: Output path for the main JSONL file (used to determine metadata directory)
        
        Returns:
            List of QA dictionaries
        """
        qa_pairs = []
        
        # Create FreeformAnalysisCategories instance
        try:
            analyzer = FreeformAnalysisCategories(
                svf_map=scene_data.get('svf'),
                height_map=scene_data.get('dsm'),
                segmentation_map=scene_data.get('seg'),
                rgb_image=scene_data.get('rgb'),
                file_path=scene_data.get('file_path', 'unknown'),
                debug=self.debug,
                use_gpt4=self.use_gpt4,
                api_key=self.api_key,
                model=self.model
            )
        except Exception as e:
            if self.debug:
                print(f"  Failed to create analyzer: {e}")
            return []
        
        # Determine which categories to process
        categories_to_process = target_categories or self.analysis_categories
        
        # Process each category
        for category in categories_to_process:
            try:
                if category == 'urban_development_application':
                    qa_data = analyzer.urban_development_application()
                elif category == 'renewable_energy_installation':
                    qa_data = analyzer.renewable_energy_installation()
                elif category == 'landscape_analysis':
                    qa_data = analyzer.landscape_analysis()
                elif category == 'water_accumulation':
                    qa_data = analyzer.water_accumulation()
                else:
                    if self.debug:
                        print(f"  Unknown category: {category}")
                    continue
                
                # Extract answer and metadata from qa_data
                answer = qa_data.get('answer', '')
                metadata = qa_data.get('context', [])
                
                # Create clean QA pair without heavy metadata
                qa_pair = {
                    'scene_id': scene_data.get('scene_id', 'unknown'),
                    'question': qa_data['question'],
                    'answer': answer,
                    'category': category,
                    'question_type': 'free_form',
                    'analysis_type': qa_data['analysis_type'],
                    'generation_method': qa_data.get('generation_method', 'freeform_analysis')
                }
                
                # Save metadata and metrics to separate YAML files if they exist and output_path is provided
                if output_path and (qa_data.get('metrics') or metadata):
                    import yaml
                    import os
                    
                    # Create metadata directory structure
                    scene_id = scene_data.get('scene_id', 'unknown')
                    base_output_dir = os.path.dirname(output_path)
                    metadata_dir = os.path.join(base_output_dir, 'qa_metadata', scene_id)
                    os.makedirs(metadata_dir, exist_ok=True)
                    
                    # Generate unique filename for this QA pair
                    qa_filename = f"{category}_{len(qa_pairs):04d}"
                    
                    # Save metrics if available
                    if qa_data.get('metrics'):
                        metrics_path = os.path.join(metadata_dir, f"{qa_filename}_metrics.yaml")
                        with open(metrics_path, 'w', encoding='utf-8') as f:
                            yaml.dump(qa_data['metrics'], f, default_flow_style=False, allow_unicode=True)
                        
                        if self.debug:
                            print(f" Metrics saved to: {metrics_path}")
                    
                    # Save metadata if available
                    if metadata:
                        metadata_path = os.path.join(metadata_dir, f"{qa_filename}_metadata.yaml")
                        with open(metadata_path, 'w', encoding='utf-8') as f:
                            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
                        
                        if self.debug:
                            print(f" Metadata saved to: {metadata_path}")
                    
                    # Add reference to metadata files in QA pair (optional)
                    qa_pair['metadata_files'] = {
                        'metrics': f"{qa_filename}_metrics.yaml" if qa_data.get('metrics') else None,
                        'metadata': f"{qa_filename}_metadata.yaml" if metadata else None,
                        'metadata_dir': metadata_dir
                    }
                qa_pairs.append(qa_pair)
                
                if self.debug:
                    print(f" Generated QA for {category}")
                    
            except Exception as e:
                if self.debug:
                    print(f"  Failed to generate QA for {category}: {e}")
                continue
        
        return qa_pairs
    
    def generate_qa_dataset_realtime(self, scenes: List[Dict[str, Any]], 
                                   target_questions: int = 200, 
                                   output_path: Optional[str] = None,
                                   dataset_type: str = "train") -> List[Dict[str, Any]]:
        """
        Generate QA dataset with real-time saving
        
        Args:
            scenes: List of scene data dictionaries
            target_questions: Target number of questions
            output_path: Path to save the dataset (optional)
            dataset_type: Type of dataset (train/test)
        
        Returns:
            List of QA pairs
        """
        all_qa_pairs = []
        questions_per_scene = max(1, target_questions // len(scenes))
        
        if self.debug:
            print(f" Generating {target_questions} questions from {len(scenes)} scenes")
            print(f" Target questions per scene: {questions_per_scene}")
        
        # Open output file if specified
        output_file = None
        if output_path:
            output_file = open(output_path, 'w', encoding='utf-8')
        metadata_dir = os.path.join(os.path.dirname(output_path) if output_path else 'metadata', 'qa_metadata')
        try:
            scene_count = 0
            for scene in scenes:
                scene_count += 1
                
                if self.debug:
                    print(f" Processing scene {scene_count}/{len(scenes)}: {scene.get('scene_id', 'unknown')}")
                
                # Generate QA pairs for this scene
                scene_qa_pairs = self.generate_qa_from_scene(scene, output_path=output_path)
                
                if not scene_qa_pairs:
                    if self.debug:
                        print(f"  No QA pairs generated for scene {scene.get('scene_id', 'unknown')}")
                    continue
                
                # Add to collection
                all_qa_pairs.extend(scene_qa_pairs)
                
                # Save to file if specified
                if output_file:
                    for qa_pair in scene_qa_pairs:
                        json.dump(qa_pair, output_file, ensure_ascii=False)
                        output_file.write('\n')
                        output_file.flush()
                
                if self.debug:
                    print(f" Generated {len(scene_qa_pairs)} QA pairs for scene {scene.get('scene_id', 'unknown')}")
                
                # Check if we've reached the target
                if len(all_qa_pairs) >= target_questions:
                    break
        
        finally:
            if output_file:
                output_file.close()
        
        # Shuffle the results to remove any order bias
        all_qa_pairs = bias_free_shuffle(all_qa_pairs)
        
        # Truncate to target number if needed
        if len(all_qa_pairs) > target_questions:
            all_qa_pairs = all_qa_pairs[:target_questions]
        
        if self.debug:
            print(f" Generated {len(all_qa_pairs)} QA pairs total")
        
        return all_qa_pairs
    
    def visualize_and_save_statistics(self, scenes: List[Dict[str, Any]], 
                                    output_dir: str) -> Dict[str, Any]:
        """
        Visualize and save statistics for scenes (compatibility with existing interface)
        
        Args:
            scenes: List of scene data dictionaries
            output_dir: Output directory for visualizations
        
        Returns:
            Summary statistics
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        successful_visualizations = 0
        successful_statistics = 0
        
        for i, scene in enumerate(scenes):
            try:
                # Create analyzer for this scene
                analyzer = FreeformAnalysisCategories(
                    svf_map=scene.get('svf'),
                    height_map=scene.get('dsm'),
                    segmentation_map=scene.get('seg'),
                    rgb_image=scene.get('rgb'),
                    file_path=scene.get('file_path', f'scene_{i}'),
                    debug=self.debug,
                    use_gpt4=self.use_gpt4,
                    api_key=self.api_key,
                    model=self.model
                )
                
                # Generate statistics for all categories
                scene_stats = {}
                for category in self.analysis_categories:
                    try:
                        if category == 'urban_development_application':
                            qa_data, answer, metadata = analyzer.urban_development_application()
                        elif category == 'renewable_energy_installation':
                            qa_data, answer, metadata = analyzer.renewable_energy_installation()
                        elif category == 'landscape_analysis':
                            qa_data, answer, metadata = analyzer.landscape_analysis()
                        elif category == 'water_accumulation':
                            qa_data, answer, metadata = analyzer.water_accumulation()
                        else:
                            continue
                        
                        scene_stats[category] = {
                            'metrics': qa_data.get('metrics', {}),
                            'question': qa_data.get('question', ''),
                            'answer': answer,
                            'generation_method': qa_data.get('generation_method', 'freeform_analysis')
                        }
                    except Exception as e:
                        if self.debug:
                            print(f"  Failed to process {category} for scene {i}: {e}")
                        continue
                
                # Save statistics to JSON
                stats_file = os.path.join(output_dir, f'scene_{i:03d}_freeform_stats.json')
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump(scene_stats, f, ensure_ascii=False, indent=2)
                
                successful_statistics += 1
                successful_visualizations += 1  # Count as successful for compatibility
                
                if self.debug:
                    print(f" Saved statistics for scene {i} to {stats_file}")
                    
            except Exception as e:
                if self.debug:
                    print(f"  Failed to process scene {i}: {e}")
                continue
        
        return {
            'successful_visualizations': successful_visualizations,
            'successful_statistics': successful_statistics,
            'total_scenes': len(scenes)
        }