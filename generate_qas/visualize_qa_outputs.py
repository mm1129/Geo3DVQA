#!/usr/bin/env python3
"""
QA Output Visualization Tool
Tool for analyzing and visualizing question-answer pairs in JSONL files.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
import os
import re
from PIL import Image, ImageDraw, ImageFont
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
from visualize_qa_results import QAVisualizer
from tqdm import tqdm
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap
import random

class QAOutputAnalyzer:
    def __init__(self, jsonl_path, image_dir=None, svf_dir=None):
        """
        Class for analyzing and visualizing question-answer pairs from JSON or JSONL files.
        
        Args:
            jsonl_path (str): Path to JSON or JSONL file
            image_dir (str): Path to image directory (for visualization)
            svf_dir (str): Path to SVF directory (for visualization)
        """
        self.jsonl_path = jsonl_path
        self.image_dir = image_dir
        self.svf_dir = svf_dir
        self.data = []
        self.analysis_results = {}
        
        # Use existing visualizer if image path is provided
        self.visualizer = None
        if image_dir:
            self.visualizer = QAVisualizer(jsonl_path, image_dir, svf_dir)
            # Get SVF colormap from visualizer
            self.svf_cmap = self.visualizer.svf_cmap
        else:
            # Create SVF colormap only if visualizer is not available
            from matplotlib.colors import LinearSegmentedColormap
            self.svf_cmap = LinearSegmentedColormap.from_list(
                'svf', ['darkblue', 'blue', 'cyan', 'yellow', 'orange', 'red'], N=256)
        
        self.load_data()
        
    def load_data(self):
        """Load question-answer data from JSON or JSONL file"""
        try:
            with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Try loading as JSON array first
            try:
                data = json.loads(content)
                # random.shuffle(data)
                if isinstance(data, list):
                    self.data = self._normalize_data_format(data)
                    print(f" Loaded as JSON array: {len(self.data)} QA pairs")
                    return
                else:
                    # If single object, convert to list
                    self.data = self._normalize_data_format([data])
                    print(f" Loaded as single JSON object: 1 QA pair")
                    return
            except json.JSONDecodeError:
                # If failed, process as JSONL
                pass
            
            # Load as JSONL file, line by line
            raw_data = []
            for line in content.split('\n'):
                line = line.strip()
                if line:
                    try:
                        raw_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f" Line parse error (skipped): {e}")
                        continue
            
            self.data = self._normalize_data_format(raw_data)
            print(f" Loaded as JSONL: {len(self.data)} QA pairs")
            
        except Exception as e:
            print(f" Data load error: {e}")
            
    def _normalize_data_format(self, raw_data):
        """
        Normalize different data formats to unified format
        Supports both old format and new conversation format
        """
        normalized_data = []
        
        for item in raw_data:
            # Check if it's new conversation format
            if 'conversations' in item and 'metadata' in item:
                # New conversation format
                question_text = ""
                answer_text = ""
                
                # Extract conversation content
                for conv in item.get('conversations', []):
                    if conv.get('from') == 'human':
                        question_text = conv.get('value', '').replace('<image>\n', '').strip()
                    elif conv.get('from') == 'gpt':
                        answer_text = conv.get('value', '').strip()
                
                # Create normalized item
                normalized_item = {
                    'question_id': item.get('id', ''),
                    'category': item.get('metadata', {}).get('category', ''),
                    'text': question_text,
                    'answer': answer_text,
                    'image': item.get('image', ''),
                    'mode': item.get('metadata', {}).get('mode', ''),
                    'version': item.get('metadata', {}).get('version', ''),
                    'phase': item.get('metadata', {}).get('phase', '')
                }
                
                # Try to extract choices if available
                if 'choices' in item:
                    normalized_item['choices'] = item['choices']
                
                normalized_data.append(normalized_item)
                
            else:
                # Old format - keep as is but ensure required fields exist
                normalized_item = {
                    'question_id': item.get('question_id', item.get('id', '')),
                    'category': item.get('category', ''),
                    'text': item.get('text', ''),
                    'answer': item.get('answer', ''),
                    'image': item.get('image', '')
                }
                
                # Copy additional fields
                for key, value in item.items():
                    if key not in normalized_item:
                        normalized_item[key] = value
                
                normalized_data.append(normalized_item)
        
        return normalized_data
            
    def analyze_categories(self):
        """Analyze by category"""
        category_counts = Counter(item['category'] for item in self.data)
        image_counts = Counter(item['image'] for item in self.data)
        
        self.analysis_results['categories'] = dict(category_counts)
        self.analysis_results['images'] = dict(image_counts)
        self.analysis_results['total_questions'] = len(self.data)
        self.analysis_results['total_images'] = len(image_counts)
        
        print("\n Number of questions per category:")
        for category, count in category_counts.most_common():
            print(f"  {category}: {count}")
            
        print(f"\n Number of images used: {len(image_counts)}")
        print(f" Total number of questions: {len(self.data)}")
        
        return self.analysis_results

    def analyze_prompt_variations(self):
        """Analyze prompt variations (using extract_prompt_variation)"""
        prompt_variations = defaultdict(list)
        category_prompts = defaultdict(set)
        
        for item in self.data:
            category = item['category']
            text = item['text']
            
            # Extract prompt variation
            prompt_base = extract_prompt_variation(text)
            if prompt_base:
                prompt_variations[prompt_base].append(item)
                category_prompts[category].add(prompt_base)
        
        print("\n Prompt variation analysis:")
        print(f"  Unique prompt bases: {len(prompt_variations)}")
        
        # Prompt diversity per category
        print("\n  Prompt diversity per category:")
        for category, prompts in category_prompts.items():
            print(f"    {category}: {len(prompts)} prompt types")
        
        # Most frequently used prompt bases
        print("\n  Top prompt bases by frequency:")
        sorted_prompts = sorted(prompt_variations.items(), key=lambda x: len(x[1]), reverse=True)
        for prompt_base, items in sorted_prompts[:10]:
            print(f"    '{prompt_base[:50]}...': {len(items)} uses")
        
        return {
            'prompt_variations': dict(prompt_variations),
            'category_prompts': dict(category_prompts),
            'total_unique_prompts': len(prompt_variations)
        }
        
    def analyze_question_complexity(self):
        """Analyze question complexity (including hard categories and freeform categories)"""
        complexity_analysis = {
            'avg_text_length': 0,
            'coordinate_types': defaultdict(int),
            'question_patterns': defaultdict(int),
            'answer_types': defaultdict(int),
            'hard_categories': defaultdict(int),
            'freeform_categories': defaultdict(int),
            'difficulty_levels': defaultdict(int)
        }
        
        text_lengths = []
        coordinate_patterns = []
        
        # Definition of hard categories
        hard_categories = [
            'hard_sun_exposure', 'hard_scenic_quality', 'hard_pixel', 'hard_grid_5×5',
            'hard_metric', 'hard_ranking', 'hard_urban_analysis', 'hard_scenic_analysis',
            'hard_openness_analysis'
        ]
        
        # Definition of freeform categories
        freeform_categories = [
            'urban_development_application', 'renewable_energy_installation',
            'landscape_analysis', 'water_accumulation'
        ]
        
        # Definition of SVF RGB estimated categories
        svf_rgb_categories = [
            'sunExposure', 'urbanDensity', 'skyVisibility', 'opennessAssessment',
            'visibilityRange', 'svfComparison', 'naturalArtificialRatio',
            'scenicQuality', 'landcoverType', 'landUse', 'heightInference',
            'highestRegion'
        ]
        
        for item in self.data:
            text = item['text']
            answer = item['answer']
            category = item['category']
            
            # Text length
            text_lengths.append(len(text))
            
            # Classify as hard category
            if category in hard_categories:
                complexity_analysis['hard_categories'][category] += 1
                complexity_analysis['difficulty_levels']['hard'] += 1
            elif category in freeform_categories:
                complexity_analysis['freeform_categories'][category] += 1
                complexity_analysis['difficulty_levels']['freeform'] += 1
            elif category in svf_rgb_categories:
                complexity_analysis['difficulty_levels']['svf_rgb'] += 1
                if 'svf_rgb_categories' not in complexity_analysis:
                    complexity_analysis['svf_rgb_categories'] = defaultdict(int)
                complexity_analysis['svf_rgb_categories'][category] += 1
            else:
                complexity_analysis['difficulty_levels']['standard'] += 1
            
            # Detect coordinate types
            if '(' in text and '%)' in text:
                complexity_analysis['coordinate_types']['percentage_points'] += 1
            if '[xmin=' in text and 'ymin=' in text:
                complexity_analysis['coordinate_types']['percentage_regions'] += 1
            if 'Region' in text and '[' in text:
                complexity_analysis['coordinate_types']['labeled_regions'] += 1
            if 'grid' in text.lower() or 'cell' in text.lower():
                complexity_analysis['coordinate_types']['grid_based'] += 1
            if '5×5' in text or '5×5' in text:
                complexity_analysis['coordinate_types']['5×5_grid'] += 1
                
            # Analyze answer types
            if answer in ['Yes', 'No']:
                complexity_analysis['answer_types']['yes_no'] += 1
            elif answer.startswith('Region'):
                complexity_analysis['answer_types']['region_choice'] += 1
            elif answer.startswith('Point'):
                complexity_analysis['answer_types']['point_choice'] += 1
            elif answer.startswith('grid_'):
                complexity_analysis['answer_types']['grid_cell'] += 1
            elif ',' in answer and not '(' in answer:  # Ranking format
                complexity_analysis['answer_types']['ranking'] += 1
            elif category in freeform_categories:
                complexity_analysis['answer_types']['freeform_descriptive'] += 1
            elif category in svf_rgb_categories:
                # Analyze SVF RGB answer types
                if 'A)' in answer or 'B)' in answer or 'C)' in answer or 'D)' in answer:
                    complexity_analysis['answer_types']['svf_multiple_choice'] += 1
                elif answer.startswith('Point'):
                    complexity_analysis['answer_types']['svf_point_selection'] += 1
                elif answer.startswith('Region'):
                    complexity_analysis['answer_types']['svf_region_selection'] += 1
                elif answer in ['Yes', 'No']:
                    complexity_analysis['answer_types']['svf_yes_no'] += 1
                else:
                    complexity_analysis['answer_types']['svf_other'] += 1
            elif re.match(r'^\d+\.\d+$', answer):  # Numeric format
                complexity_analysis['answer_types']['numerical'] += 1
            elif '(' in answer and '%)' in answer:  # Coordinate format
                complexity_analysis['answer_types']['coordinate'] += 1
            else:
                complexity_analysis['answer_types']['other'] += 1
                
            # Analyze question patterns
            if 'IMPORTANT:' in text:
                complexity_analysis['question_patterns']['explicit_instructions'] += 1
            if 'format:' in text.lower():
                complexity_analysis['question_patterns']['format_specified'] += 1
            if 'analysis' in text.lower():
                complexity_analysis['question_patterns']['analysis_required'] += 1
            if 'calculate' in text.lower():
                complexity_analysis['question_patterns']['calculation_required'] += 1
                
        complexity_analysis['avg_text_length'] = np.mean(text_lengths)
        
        print("\n Question complexity analysis (including hard categories):")
        print(f"  Average text length: {complexity_analysis['avg_text_length']:.1f} chars")
        
        print("\n  Difficulty levels:")
        for level, count in complexity_analysis['difficulty_levels'].items():
            print(f"    {level}: {count}")
            
        print("\n  Hard category distribution:")
        for hard_cat, count in complexity_analysis['hard_categories'].items():
            print(f"    {hard_cat}: {count}")
            
        print("\n  Coordinate types:")
        for coord_type, count in complexity_analysis['coordinate_types'].items():
            print(f"    {coord_type}: {count}")
            
        print("\n  Answer types:")
        for answer_type, count in complexity_analysis['answer_types'].items():
            print(f"    {answer_type}: {count}")
            
        print("\n  Question patterns:")
        for pattern, count in complexity_analysis['question_patterns'].items():
            print(f"    {pattern}: {count}")
            
        return complexity_analysis
        
    def check_question_quality(self):
        """Check question quality (VLM benchmark perspective + hard categories + freeform categories)"""
        quality_issues = []
        
        # Definition of hard categories
        hard_categories = [
            'hard_sun_exposure', 'hard_scenic_quality', 'hard_pixel', 'hard_grid_5×5',
            'hard_metric', 'hard_ranking', 'hard_urban_analysis', 'hard_scenic_analysis',
            'hard_openness_analysis'
        ]
        
        # Definition of freeform categories
        freeform_categories = [
            'urban_development_application', 'renewable_energy_installation',
            'landscape_analysis', 'water_accumulation'
        ]
        
        # Definition of SVF RGB estimated categories
        svf_rgb_categories = [
            'sunExposure', 'urbanDensity', 'skyVisibility', 'opennessAssessment',
            'visibilityRange', 'svfComparison', 'naturalArtificialRatio',
            'scenicQuality', 'landcoverType', 'landUse', 'heightInference',
            'highestRegion', 'grid_analysis', 'solar_analysis', 'multimodal_landcover_analysis',
            'multimodal_urban_analysis', 'energy_planning', 'height_analysis'
        ]
        
        for i, item in enumerate(self.data):
            question_id = item['question_id']
            text = item['text']
            category = item['category']
            answer = item['answer']
            
            # Check for ambiguous expressions
            ambiguous_phrases = [
                'most variable SVF characteristics',
                'optimal',
                'best',
                'feels',
                'looks',
                'seems',
                'balanced mix',
                'harmoniously',
                'beautiful',
                'scenic'
            ]
            
            for phrase in ambiguous_phrases:
                if phrase in text:
                    quality_issues.append({
                        'question_id': question_id,
                        'category': category,
                        'issue': f"Ambiguous expression: '{phrase}'",
                        'text_snippet': text[:100] + "..."
                    })
                    
            # Check for redundancy
            if 'SVF' in text and 'Sky View Factor' in text:
                quality_issues.append({
                    'question_id': question_id,
                    'category': category,
                    'issue': "Redundant: both SVF and Sky View Factor present",
                    'text_snippet': text[:100] + "..."
                })
                
            # Check for coordinate format consistency
            coordinate_formats = []
            if re.search(r'\(\d+%, \d+%\)', text):
                coordinate_formats.append('point_percentage')
            if re.search(r'\[xmin=\d+%, ymin=\d+%', text):
                coordinate_formats.append('region_percentage')
                
            if len(coordinate_formats) > 1:
                quality_issues.append({
                    'question_id': question_id,
                    'category': category,
                    'issue': "Mixed coordinate formats",
                    'text_snippet': text[:100] + "..."
                })
                
            # Hard category specific checks
            if category in hard_categories:
                # hard_pixel checks
                if category == 'hard_pixel':
                    if 'IMPORTANT:' not in text:
                        quality_issues.append({
                            'question_id': question_id,
                            'category': category,
                            'issue': "No IMPORTANT instruction in hard_pixel question",
                            'text_snippet': text[:100] + "..."
                        })
                    if not re.match(r'^\d+\.\d{4}$', answer):
                        quality_issues.append({
                            'question_id': question_id,
                            'category': category,
                            'issue': f"Invalid pixel answer format: '{answer}' (expected X.XXXX)",
                            'text_snippet': text[:100] + "..."
                        })
                
                # hard_grid_5×5 checks
                elif category == 'hard_grid_5×5':
                    if not answer.startswith('grid_'):
                        quality_issues.append({
                            'question_id': question_id,
                            'category': category,
                            'issue': f"Invalid 5×5 grid answer format: '{answer}' (expected grid_X_Y)",
                            'text_snippet': text[:100] + "..."
                        })
                    if '5×5' not in text and '5×5' not in text:
                        quality_issues.append({
                            'question_id': question_id,
                            'category': category,
                            'issue': "No grid explanation in 5×5 grid question",
                            'text_snippet': text[:100] + "..."
                        })
                
                # hard_ranking checks
                elif category == 'hard_ranking':
                    if ',' not in answer:
                        quality_issues.append({
                            'question_id': question_id,
                            'category': category,
                            'issue': f"Invalid ranking answer format: '{answer}' (expected A,B,C)",
                            'text_snippet': text[:100] + "..."
                        })
                
                # Numeric answer categories
                elif category in ['hard_metric', 'hard_urban_analysis', 'hard_scenic_analysis', 'hard_openness_analysis']:
                    if not re.match(r'^\d+\.\d{4}$', answer):
                        quality_issues.append({
                            'question_id': question_id,
                            'category': category,
                            'issue': f"Invalid numeric answer format: '{answer}' (expected X.XXXX)",
                            'text_snippet': text[:100] + "..."
                        })
            
            # Freeform category specific checks
            elif category in freeform_categories:
                # Check for overly short answers (should be detailed)
                if len(answer.split()) < 15:
                    quality_issues.append({
                        'question_id': question_id,
                        'category': category,
                        'issue': f"Answer too short for freeform category: {len(answer.split())} words (expected 15+ words)",
                        'text_snippet': f"Answer: '{answer}'"
                    })
                
                # Check for statistical integration
                if 'metrics' in item and not any(metric in answer for metric in ['SVF', 'elevation', 'terrain', 'height', 'slope', 'ratio', '%']):
                    quality_issues.append({
                        'question_id': question_id,
                        'category': category,
                        'issue': "Answer lacks statistical integration despite available metrics",
                        'text_snippet': f"Answer: '{answer}'"
                    })
                
                # Check for generic repetitive answers
                generic_phrases = [
                    "Sky visibility could be improved",
                    "Excellent solar potential with high sky visibility",
                    "High sky openness creates a sense of spatial freedom",
                    "The open-space square faces the most risks"
                ]
                
                for phrase in generic_phrases:
                    if phrase in answer:
                        quality_issues.append({
                            'question_id': question_id,
                            'category': category,
                            'issue': f"Generic repetitive answer detected: '{phrase}'",
                            'text_snippet': f"Answer: '{answer}'"
                        })
            
            # SVF RGB category specific checks
            elif category in svf_rgb_categories:
                # Check for coordinate format consistency
                if 'Point' in answer and not answer.startswith('Point'):
                    quality_issues.append({
                        'question_id': question_id,
                        'category': category,
                        'issue': f"Inconsistent point format: '{answer}' (should start with 'Point')",
                        'text_snippet': f"Answer: '{answer}'"
                    })
                
                # Check for region format consistency
                if 'Region' in answer and not answer.startswith('Region'):
                    quality_issues.append({
                        'question_id': question_id,
                        'category': category,
                        'issue': f"Inconsistent region format: '{answer}' (should start with 'Region')",
                        'text_snippet': f"Answer: '{answer}'"
                    })
                
                # Check for multiple choice format
                if any(choice in answer for choice in ['A)', 'B)', 'C)', 'D)']) and not any(answer.startswith(choice) for choice in ['A)', 'B)', 'C)', 'D)']):
                    quality_issues.append({
                        'question_id': question_id,
                        'category': category,
                        'issue': f"Inconsistent multiple choice format: '{answer}'",
                        'text_snippet': f"Answer: '{answer}'"
                    })
                
                # Check for coordinate information in questions
                if 'coordinates' in text.lower() and not any(coord in text for coord in ['(', ')', '%', 'Point', 'Region']):
                    quality_issues.append({
                        'question_id': question_id,
                        'category': category,
                        'issue': "Question mentions coordinates but lacks coordinate information",
                        'text_snippet': text[:100] + "..."
                    })
                        
        print(f"\n  Number of quality issues detected: {len(quality_issues)}")
        if quality_issues:
            print("\nIssue details:")
            for issue in quality_issues[:15]:  # Show first 15
                print(f"  Q{issue['question_id']} ({issue['category']}): {issue['issue']}")
                
        return quality_issues
        
    def create_category_distribution_plot(self, save_path=None):
        """Plot category distribution"""
        if not self.analysis_results.get('categories'):
            self.analyze_categories()
            
        categories = list(self.analysis_results['categories'].keys())
        counts = list(self.analysis_results['categories'].values())
        
        plt.figure(figsize=(12, 8))
        
        # Horizontal bar plot (for long category names)
        bars = plt.barh(categories, counts, color='skyblue', alpha=0.7)
        
        # Show value on each bar
        for i, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    str(count), ha='left', va='center', fontweight='bold')
        
        plt.xlabel('Number of Questions', fontsize=12)
        plt.ylabel('Category', fontsize=12)
        plt.title('Number of Questions per Category', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Category distribution plot saved: {save_path}")
        else:
            plt.show()

    def create_prompt_variation_report(self, save_path=None):
        """Create prompt variation report"""
        prompt_analysis = self.analyze_prompt_variations()
        
        report = []
        report.append("=" * 80)
        report.append("Prompt Variation Analysis Report")
        report.append("=" * 80)
        report.append(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total unique prompts: {prompt_analysis['total_unique_prompts']}")
        report.append("")
        
        # Prompt diversity per category
        report.append(" Prompt diversity per category:")
        for category, prompts in prompt_analysis['category_prompts'].items():
            report.append(f"  {category}: {len(prompts)} types")
        report.append("")
        
        # Top prompt bases by frequency
        report.append(" Top prompt bases by frequency:")
        sorted_prompts = sorted(prompt_analysis['prompt_variations'].items(), 
                              key=lambda x: len(x[1]), reverse=True)
        for i, (prompt_base, items) in enumerate(sorted_prompts[:20]):
            report.append(f"  {i+1:2d}. '{prompt_base[:60]}...' ({len(items)} uses)")
        report.append("")
        
        # Prompt diversity score
        total_questions = len(self.data)
        if total_questions > 0:
            diversity_ratio = prompt_analysis['total_unique_prompts'] / total_questions
            diversity_score = min(diversity_ratio * 100, 100)
        else:
            diversity_ratio = 0.0
            diversity_score = 0.0
        report.append(f" Prompt diversity score: {diversity_score:.1f}/100")
        report.append(f"   (Unique ratio: {diversity_ratio:.1%})")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f" Prompt variation report saved: {save_path}")
        else:
            print(report_text)
            
        return report_text
            
    def create_quality_report(self, save_path=None):
        """Create quality report (including hard categories)"""
        quality_issues = self.check_question_quality()
        complexity_analysis = self.analyze_question_complexity()
        prompt_analysis = self.analyze_prompt_variations()
        
        report = []
        report.append("=" * 80)
        report.append("VLM Benchmark Quality Report (Hard, Freeform & SVF RGB Categories)")
        report.append("=" * 80)
        report.append(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total questions: {len(self.data)}")
        report.append(f"Total images: {len(set(item['image'] for item in self.data))}")
        report.append(f"Unique prompt count: {prompt_analysis['total_unique_prompts']}")
        report.append("")
        
        # Difficulty level distribution
        report.append(" Difficulty level distribution:")
        for level, count in complexity_analysis['difficulty_levels'].items():
            percentage = (count / len(self.data)) * 100
            report.append(f"  {level}: {count} ({percentage:.1f}%)")
        report.append("")
        
        # Freeform category distribution
        if complexity_analysis['freeform_categories']:
            report.append(" Freeform category distribution:")
            for category, count in complexity_analysis['freeform_categories'].items():
                percentage = (count / len(self.data)) * 100
                report.append(f"  {category}: {count} ({percentage:.1f}%)")
            report.append("")
        
        # SVF RGB category distribution
        if 'svf_rgb_categories' in complexity_analysis and complexity_analysis['svf_rgb_categories']:
            report.append(" SVF RGB category distribution:")
            for category, count in complexity_analysis['svf_rgb_categories'].items():
                percentage = (count / len(self.data)) * 100
                report.append(f"  {category}: {count} ({percentage:.1f}%)")
            report.append("")
        
        # Category distribution
        report.append(" Category distribution:")
        for category, count in self.analysis_results['categories'].items():
            percentage = (count / len(self.data)) * 100
            category_type = "Hard" if category.startswith('hard_') else "Standard"
            report.append(f"  {category} ({category_type}): {count} ({percentage:.1f}%)")
        report.append("")
        
        # Hard category details
        if complexity_analysis['hard_categories']:
            report.append(" Hard category details:")
            for hard_cat, count in complexity_analysis['hard_categories'].items():
                report.append(f"  {hard_cat}: {count}")
            report.append("")
        
        # Complexity analysis
        report.append(" Question complexity:")
        report.append(f"  Average text length: {complexity_analysis['avg_text_length']:.1f} chars")
        report.append("  Coordinate type distribution:")
        for coord_type, count in complexity_analysis['coordinate_types'].items():
            report.append(f"    {coord_type}: {count}")
        report.append("  Answer type distribution:")
        for answer_type, count in complexity_analysis['answer_types'].items():
            report.append(f"    {answer_type}: {count}")
        report.append("  Question patterns:")
        for pattern, count in complexity_analysis['question_patterns'].items():
            report.append(f"    {pattern}: {count}")
        report.append("")
        
        # Prompt diversity
        report.append(" Prompt diversity:")
        if len(self.data) > 0:
            diversity_ratio = prompt_analysis['total_unique_prompts'] / len(self.data)
        else:
            diversity_ratio = 0.0
        report.append(f"  Unique prompt ratio: {diversity_ratio:.1%}")
        report.append("  Number of prompt types per category:")
        for category, prompts in prompt_analysis['category_prompts'].items():
            report.append(f"    {category}: {len(prompts)} types")
        report.append("")
        
        # Quality issues
        report.append(f"  Quality issues: {len(quality_issues)}")
        if quality_issues:
            issue_categories = defaultdict(int)
            for issue in quality_issues:
                issue_categories[issue['issue'].split(':')[0]] += 1
            
            for issue_type, count in issue_categories.items():
                report.append(f"  {issue_type}: {count}")
            
            report.append("\nDetailed issue list:")
            for issue in quality_issues:
                report.append(f"  Q{issue['question_id']}: {issue['issue']}")
        report.append("")
        
        # Benchmark quality metrics
        report.append(" Benchmark Quality Metrics:")
        
        # 1. Diversity metric
        unique_categories = len(self.analysis_results['categories'])
        diversity_score = min(unique_categories / 10.0, 1.0) * 100  # 10 categories = full score
        report.append(f"  Category diversity score: {diversity_score:.1f}/100")
        
        # 2. Difficulty balance
        hard_ratio = complexity_analysis['difficulty_levels'].get('hard', 0) / len(self.data)
        balance_score = (1 - abs(hard_ratio - 0.4)) * 100  # 40% hard categories is ideal
        report.append(f"  Difficulty balance score: {balance_score:.1f}/100 (Hard ratio: {hard_ratio:.1%})")
        
        # 3. Quality score
        quality_score = max(0, (1 - len(quality_issues) / len(self.data))) * 100
        report.append(f"  Quality score: {quality_score:.1f}/100")
        
        # 4. Answer format diversity
        unique_answer_types = len(complexity_analysis['answer_types'])
        answer_diversity_score = min(unique_answer_types / 8.0, 1.0) * 100  # 8 types = full score
        report.append(f"  Answer format diversity score: {answer_diversity_score:.1f}/100")
        
        # 5. Prompt diversity score
        if len(self.data) > 0:
            prompt_diversity_score = min(diversity_ratio * 100, 100)
        else:
            prompt_diversity_score = 0.0
        report.append(f"  Prompt diversity score: {prompt_diversity_score:.1f}/100")
        
        # 6. Overall score
        overall_score = (diversity_score + balance_score + quality_score + answer_diversity_score + prompt_diversity_score) / 5
        report.append(f"  [Overall Benchmark Quality Score: {overall_score:.1f}/100]")
        report.append("")
        
        # Recommendations
        report.append(" Recommendations:")
        if diversity_score < 70:
            report.append("  1. Increase category diversity (add more question types)")
        if balance_score < 70:
            report.append("  2. Adjust difficulty balance (optimize hard category ratio)")
        if quality_score < 90:
            report.append("  3. Resolve quality issues (unify ambiguous expressions and formats)")
        if answer_diversity_score < 70:
            report.append("  4. Diversify answer formats (numeric, coordinates, grid, ranking, etc.)")
        if prompt_diversity_score < 70:
            report.append("  5. Increase prompt variation (diversify templates)")
        if overall_score >= 85:
            report.append("   High-quality benchmark!")
        elif overall_score >= 70:
            report.append("   Good quality. Further improvement possible with fine-tuning.")
        else:
            report.append("   Room for quality improvement. Please consider the above recommendations.")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f" Quality report saved: {save_path}")
        else:
            print(report_text)
            
        return report_text

    def _generate_visualization_choices(self, qa_item):
        """
        動的に選択肢を生成（ビジュアライゼーション専用）
        質問の内容や回答には一切影響を与えない
        """
        text = qa_item.get('text', '')
        category = qa_item.get('category', '')
        answer = qa_item.get('answer', '')
        
        choices = []
        
        if category in ['hard_ranking', 'regional_svf_variability', 'sun_exposure', 'highest_region', 
                       'urban_density', 'openness_assessment', 'natural_artificial_ratio', 'scenic_quality',
                       'hard_pixel', 'height_inference', 'height_average']:
            # regionカテゴリの場合、テキストから領域を抽出
            
            if category == 'hard_ranking':
                # hard_rankingの場合、全順列を生成
                # Pattern 1: - A: [xmin=...] (single letter format)
                single_letters = re.findall(r'- ([ABC]):', text)
                regions = [f"Region {letter}" for letter in single_letters]
                
                # Pattern 2: - Region A: [xmin=...] (full name format)
                if not regions:
                    regions = re.findall(r'- (Region [ABC]):', text)
                
                # Pattern 3: Region A: Region (...) (old format)
                if not regions:
                    regions = re.findall(r'(Region [ABC]):', text)
                
                print(f" [_generate_visualization_choices] Found regions: {regions}")
                
                if len(regions) >= 3:
                    import itertools
                    # 3つの領域の全順列を生成
                    perms = list(itertools.permutations(regions[:3]))
                    choices = [', '.join(perm) for perm in perms]
                    print(f" [_generate_visualization_choices] Generated {len(choices)} choices")
                    
            else:
                # 他のregion-basedカテゴリの場合、単一の地域選択または括弧形式
                # Region A: [xmin=...] 形式から抽出
                regions = re.findall(r'(Region [ABCDEF]):\s*\[', text)
                
                # 括弧形式の場合: Region (x%, y%, z%, w%)
                if not regions:
                    bracket_regions = re.findall(r'Region \(\d+\.?\d*%, \d+\.?\d*%, \d+\.?\d*%, \d+\.?\d*%\)', text)
                    if bracket_regions:
                        choices = bracket_regions
                        print(f" [_generate_visualization_choices] Found bracket format regions for {category}: {len(choices)} choices")
                
                if regions:
                    choices = regions  # 単一選択なので順列不要
                    print(f" [_generate_visualization_choices] Found regions for {category}: {regions}")
                    print(f" [_generate_visualization_choices] Generated {len(choices)} region choices")
                
        elif category == 'hard_pixel':
            # 数値回答の場合、正解±変動の選択肢を生成
            try:
                correct_val = float(answer)
                import random
                random.seed(42)  # 再現性のため
                choices = [answer]  # 正解を含める
                # 3つの不正解を生成
                for _ in range(3):
                    variation = random.uniform(-0.2, 0.2)
                    wrong_val = max(0.0, min(1.0, correct_val + variation))
                    choices.append(f"{wrong_val:.4f}")
            except ValueError:
                choices = [answer]
                
        elif category == 'hard_grid_5×5':
            # グリッドセルの選択肢を生成
            if answer.startswith('grid_'):
                parts = answer.split('_')
                if len(parts) == 3:
                    choices = [answer]  # 正解
                    # 近隣セルを追加
                    try:
                        row, col = int(parts[1]), int(parts[2])
                        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                            new_row, new_col = row + dr, col + dc
                            if 1 <= new_row <= 5 and 1 <= new_col <= 5:
                                choices.append(f"grid_{new_row}_{new_col}")
                                if len(choices) >= 4:
                                    break
                    except ValueError:
                        pass
                        
        elif category in ['hard_metric', 'hard_urban_analysis', 'hard_scenic_analysis', 'hard_openness_analysis']:
            # 数値回答の場合
            try:
                correct_val = float(answer)
                import random
                random.seed(42)
                choices = [answer]
                for _ in range(3):
                    variation = random.uniform(-0.15, 0.15)
                    wrong_val = max(0.0, min(10.0, correct_val + variation))
                    choices.append(f"{wrong_val:.4f}")
            except ValueError:
                choices = [answer]
        
        return choices

    def parse_hard_question_info(self, qa_item):
        """
        Extract choices and coordinate information from hard category question text.
        
        Args:
            qa_item: dict (one line from jsonl)
            
        Returns:
            dict: Extracted information
        """
        text = qa_item.get('text', '')
        category = qa_item.get('category', '')
        answer = qa_item.get('answer', '')
        
        parsed_info = {
            'choices': [],
            'regions': [],
            'points': [],
            'coordinates': None
        }
        
        # hard_pixel: extract coordinate info and region info if available
        if category == 'hard_pixel':
            # Extract coordinates like "at point (56.5%, 33.2%)" or "pixel coordinates (67.8%, 13.9%)" - 小数点対応
            coord_match = re.search(r'(?:point|coordinates) \((\d+\.?\d*)%, (\d+\.?\d*)%\)', text)
            if coord_match:
                x_pct, y_pct = map(float, coord_match.groups())
                parsed_info['coordinates'] = [x_pct, y_pct]
                parsed_info['points'] = [(x_pct, y_pct)]
                print(f" [parse_hard_question_info] Found hard_pixel coordinates: ({x_pct:.1f}%, {y_pct:.1f}%)")
            
            # Extract area coordinates like "within the area [15%, 58%, 22%, 64%]" - bbox format
            area_match = re.search(r'area \[(\d+\.?\d*)%, (\d+\.?\d*)%, (\d+\.?\d*)%, (\d+\.?\d*)%\]', text)
            if area_match:
                xmin, ymin, xmax, ymax = map(float, area_match.groups())
                # Calculate center point for visualization
                x_pct = (xmin + xmax) / 2
                y_pct = (ymin + ymax) / 2
                parsed_info['coordinates'] = [x_pct, y_pct]
                parsed_info['bbox'] = [xmin, ymin, xmax, ymax]
                parsed_info['points'] = [(x_pct, y_pct)]
                print(f" [parse_hard_question_info] Found hard_pixel area: [{xmin}%, {ymin}%, {xmax}%, {ymax}%], center: ({x_pct:.1f}%, {y_pct:.1f}%)")
            
            # Extract region coordinates like "region [47%, 26%, 57%, 36%]" - new bbox format
            region_match = re.search(r'region \[(\d+\.?\d*)%, (\d+\.?\d*)%, (\d+\.?\d*)%, (\d+\.?\d*)%\]', text)
            if region_match:
                xmin, ymin, xmax, ymax = map(float, region_match.groups())
                # Calculate center point for visualization
                x_pct = (xmin + xmax) / 2
                y_pct = (ymin + ymax) / 2
                parsed_info['coordinates'] = [x_pct, y_pct]
                parsed_info['bbox'] = [xmin, ymin, xmax, ymax]
                parsed_info['points'] = [(x_pct, y_pct)]
                print(f" [parse_hard_question_info] Found hard_pixel region: [{xmin}%, {ymin}%, {xmax}%, {ymax}%], center: ({x_pct:.1f}%, {y_pct:.1f}%)")
            
            # Also check for region patterns in hard_pixel (if question includes region analysis)
            region_pattern = r'(Region [ABCDEF]):\s*\[xmin=(\d+\.?\d*)%, ymin=(\d+\.?\d*)%, xmax=(\d+\.?\d*)%, ymax=(\d+\.?\d*)%\]'
            region_matches = re.findall(region_pattern, text)
            if region_matches:
                for region_name, xmin, ymin, xmax, ymax in region_matches:
                    parsed_info['regions'].append({
                        'label': region_name,
                        'actual_label': region_name,
                        'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)]
                    })
                print(f" [parse_hard_question_info] Found {len(region_matches)} regions in hard_pixel")
        
        # sky_visibility / visibility_range: extract coordinate choices
        elif category in ['sky_visibility', 'visibility_range']:
            # Extract coordinate choices from "Please choose from:" section only
            choose_from_match = re.search(r'Please choose from:\s*(.*)', text, re.DOTALL)
            if choose_from_match:
                choose_from_text = choose_from_match.group(1)
                coord_matches = re.findall(r'\((\d+\.?\d*)%, (\d+\.?\d*)%\)', choose_from_text)
                if coord_matches:
                    for x_pct, y_pct in coord_matches:
                        parsed_info['points'].append((float(x_pct), float(y_pct)))
                    # Also store as choice strings
                    parsed_info['choices'] = [f"({float(x):.1f}%, {float(y):.1f}%)" for x, y in coord_matches]
                    print(f" [parse_hard_question_info] Found {len(coord_matches)} coordinate choices in {category}")
            else:
                # Fallback: extract all coordinates but filter out examples
                coord_matches = re.findall(r'\((\d+\.?\d*)%, (\d+\.?\d*)%\)', text)
                # Filter out common example coordinates like (25%, 50%)
                filtered_matches = [(x, y) for x, y in coord_matches if not (float(x) == 25.0 and float(y) == 50.0)]
                if filtered_matches:
                    for x_pct, y_pct in filtered_matches:
                        parsed_info['points'].append((float(x_pct), float(y_pct)))
                    parsed_info['choices'] = [f"({float(x):.1f}%, {float(y):.1f}%)" for x, y in filtered_matches]
                    print(f" [parse_hard_question_info] Found {len(filtered_matches)} coordinate points in {category} (filtered)")
            
            # Parse answer coordinate
            if answer and '(' in answer and '%' in answer:
                answer_match = re.search(r'\((\d+\.?\d*)%, (\d+\.?\d*)%\)', answer)
                if answer_match:
                    x_pct, y_pct = map(float, answer_match.groups())
                    parsed_info['coordinates'] = [x_pct, y_pct]
                    print(f" [parse_hard_question_info] Found {category} answer coordinates: ({x_pct:.1f}%, {y_pct:.1f}%)")
        
        # hard_ranking: extract region info
        elif category == 'hard_ranking':
            # Pattern 1: - A: [xmin=83%, ymin=45%, xmax=99%, ymax=60%] (single letter format)
            region_pattern_single = r'- ([ABC]): \[xmin=(\d+\.?\d*)%, ymin=(\d+\.?\d*)%, xmax=(\d+\.?\d*)%, ymax=(\d+\.?\d*)%\]'
            region_matches_single = re.findall(region_pattern_single, text)
            
            if region_matches_single:
                # Map coordinates to regions directly (single letter format)
                for letter, xmin, ymin, xmax, ymax in region_matches_single:
                    region_name = f"Region {letter}"
                    parsed_info['regions'].append({
                        'label': region_name,
                        'actual_label': region_name,
                        'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)]
                    })
            else:
                # Pattern 2: - Region A: [xmin=10.5%, ymin=5.2%, xmax=25.8%, ymax=19.1%] (full name format)
                region_pattern_full = r'- (Region [ABC]): \[xmin=(\d+\.?\d*)%, ymin=(\d+\.?\d*)%, xmax=(\d+\.?\d*)%, ymax=(\d+\.?\d*)%\]'
                region_matches_full = re.findall(region_pattern_full, text)
                
                for region_name, xmin, ymin, xmax, ymax in region_matches_full:
                    parsed_info['regions'].append({
                        'label': region_name,
                        'actual_label': region_name,
                        'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)]
                    })
                
                if not region_matches_full:
                    # Pattern 3: Old format: Region A: Region (34.5%, 67.2%, 55.8%, 89.1%)
                    region_pattern_old = r'(Region [ABC]): Region \((\d+\.?\d*)%, (\d+\.?\d*)%, (\d+\.?\d*)%, (\d+\.?\d*)%\)'
                    region_matches_old = re.findall(region_pattern_old, text)
                    
                    for region_name, xmin, ymin, xmax, ymax in region_matches_old:
                        parsed_info['regions'].append({
                            'label': region_name,
                            'actual_label': region_name,
                            'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)]
                        })
                    
            print(f" [parse_hard_question_info] Found {len(parsed_info['regions'])} regions for hard_ranking")
        
        # height_average: extract region coordinates
        elif category == 'height_average':
            # Extract coordinates like "region [71%, 10%, 82%, 21%]"
            region_match = re.search(r'region \[(\d+\.?\d*)%, (\d+\.?\d*)%, (\d+\.?\d*)%, (\d+\.?\d*)%\]', text)
            if region_match:
                xmin, ymin, xmax, ymax = map(float, region_match.groups())
                # Calculate center point for visualization
                x_pct = (xmin + xmax) / 2
                y_pct = (ymin + ymax) / 2
                parsed_info['coordinates'] = [x_pct, y_pct]
                parsed_info['bbox'] = [xmin, ymin, xmax, ymax]
                parsed_info['points'] = [(x_pct, y_pct)]
                print(f" [parse_hard_question_info] Found height_average region: [{xmin}%, {ymin}%, {xmax}%, {ymax}%], center: ({x_pct:.1f}%, {y_pct:.1f}%)")
        
        # height_inference: extract region info if available
        elif category == 'height_inference':
            # Check for region patterns in height_inference questions
            region_pattern = r'(Region [ABCDEF]):\s*\[xmin=(\d+\.?\d*)%, ymin=(\d+\.?\d*)%, xmax=(\d+\.?\d*)%, ymax=(\d+\.?\d*)%\]'
            region_matches = re.findall(region_pattern, text)
            if region_matches:
                for region_name, xmin, ymin, xmax, ymax in region_matches:
                    parsed_info['regions'].append({
                        'label': region_name,
                        'actual_label': region_name,
                        'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)]
                    })
                print(f" [parse_hard_question_info] Found {len(region_matches)} regions in height_inference")
        
        # region-based categories: extract region info (Region A/B/C/D + bbox coordinates)
        elif category in ['regional_svf_variability', 'sun_exposure', 'highest_region', 'urban_density', 
                         'openness_assessment', 'natural_artificial_ratio', 'scenic_quality']:
            # Pattern 1: Region A: [xmin=47%, ymin=53%, xmax=58%, ymax=77%]
            region_pattern = r'(Region [ABCDEF]):\s*\[xmin=(\d+\.?\d*)%, ymin=(\d+\.?\d*)%, xmax=(\d+\.?\d*)%, ymax=(\d+\.?\d*)%\]'
            region_matches = re.findall(region_pattern, text)
            
            # Pattern 2: A: [xmin=47%, ymin=53%, xmax=58%, ymax=77%] (single letter format)
            single_pattern = r'([ABCDEF]):\s*\[xmin=(\d+\.?\d*)%, ymin=(\d+\.?\d*)%, xmax=(\d+\.?\d*)%, ymax=(\d+\.?\d*)%\]'
            single_matches = re.findall(single_pattern, text)
            
            # Pattern 3: Region A: [44%, 19%, 63%, 38%] (simple coordinate format)
            simple_region_pattern = r'(Region [ABCDEF]):\s*\[(\d+\.?\d*)%, (\d+\.?\d*)%, (\d+\.?\d*)%, (\d+\.?\d*)%\]'
            simple_region_matches = re.findall(simple_region_pattern, text)
            
            # Pattern 4: A: [44%, 19%, 63%, 38%] (single letter simple format)
            simple_single_pattern = r'([ABCDEF]):\s*\[(\d+\.?\d*)%, (\d+\.?\d*)%, (\d+\.?\d*)%, (\d+\.?\d*)%\]'
            simple_single_matches = re.findall(simple_single_pattern, text)
            
            # Process Region A: format
            for region_name, xmin, ymin, xmax, ymax in region_matches:
                try:
                    parsed_info['regions'].append({
                        'label': region_name,
                        'actual_label': region_name,
                        'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)]
                    })
                except ValueError as e:
                    print(f" [parse_hard_question_info] Float conversion error for {region_name}: {e}")
                    
            # Process A: format 
            for letter, xmin, ymin, xmax, ymax in single_matches:
                try:
                    region_name = f"Region {letter}" if letter.isalpha() else letter
                    parsed_info['regions'].append({
                        'label': region_name,
                        'actual_label': letter,  # Store original single letter
                        'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)]
                    })
                except ValueError as e:
                    print(f" [parse_hard_question_info] Float conversion error for {letter}: {e}")
                    
            # Process Region A: [44%, 19%, 63%, 38%] format (simple coordinates)
            for region_name, xmin, ymin, xmax, ymax in simple_region_matches:
                try:
                    parsed_info['regions'].append({
                        'label': region_name,
                        'actual_label': region_name,
                        'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)]
                    })
                except ValueError as e:
                    print(f" [parse_hard_question_info] Float conversion error for {region_name}: {e}")
                    
            # Process A: [44%, 19%, 63%, 38%] format (single letter simple)
            for letter, xmin, ymin, xmax, ymax in simple_single_matches:
                try:
                    region_name = f"Region {letter}" if letter.isalpha() else letter
                    parsed_info['regions'].append({
                        'label': region_name,
                        'actual_label': letter,  # Store original single letter
                        'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)]
                    })
                except ValueError as e:
                    print(f" [parse_hard_question_info] Float conversion error for {letter}: {e}")
                    
            # Alternative pattern: Region (x%, y%, z%, w%) format (fallback)
            if not region_matches and not single_matches and not simple_region_matches and not simple_single_matches:
                alt_pattern = r'Region \((\d+\.?\d*)%, (\d+\.?\d*)%, (\d+\.?\d*)%, (\d+\.?\d*)%\)'
                alt_matches = re.findall(alt_pattern, text)
                for i, (xmin, ymin, xmax, ymax) in enumerate(alt_matches):
                    try:
                        region_name = f"Region {chr(65+i)}"  # A, B, C, ...
                        parsed_info['regions'].append({
                            'label': region_name,
                            'actual_label': region_name,
                            'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)]
                        })
                    except ValueError as e:
                        print(f" [parse_hard_question_info] Float conversion error for Region {chr(65+i)}: {e}")
                    
            print(f" [parse_hard_question_info] Found {len(parsed_info['regions'])} regions for {category}")
            
            # 既存のchoicesを使用、またはtextから生成
            if 'choices' in qa_item and qa_item['choices']:
                parsed_info['choices'] = qa_item['choices']
            else:
                # テキストから選択肢を抽出
                choices_in_text = []
                
                # Extract from Region A: format
                for region_name, _, _, _, _ in region_matches:
                    choices_in_text.append(region_name)
                    
                # Extract from A: format
                for letter, _, _, _, _ in single_matches:
                    # Use original format for choices (single letter if available)
                    choices_in_text.append(letter)
                    
                # Extract from Region A: [44%, 19%, 63%, 38%] format
                for region_name, _, _, _, _ in simple_region_matches:
                    choices_in_text.append(region_name)
                    
                # Extract from A: [44%, 19%, 63%, 38%] format
                for letter, _, _, _, _ in simple_single_matches:
                    # Use original format for choices (single letter if available)
                    choices_in_text.append(letter)
                
                # Fallback to alternative pattern if no matches found
                if not choices_in_text:
                    alt_pattern = r'Region \((\d+\.?\d*)%, (\d+\.?\d*)%, (\d+\.?\d*)%, (\d+\.?\d*)%\)'
                    alt_matches = re.findall(alt_pattern, text)
                    for i in range(len(alt_matches)):
                        choices_in_text.append(f"Region {chr(65+i)}")
                        
                parsed_info['choices'] = choices_in_text
                
        # hard_grid_5×5: 選択肢は不要（グリッド全体を表示）
        elif category == 'hard_grid_5×5':
            # 回答からグリッド位置を抽出（複数形式に対応）
            if answer.startswith('grid_'):
                # grid_X_Y 形式
                parts = answer.split('_')
                if len(parts) == 3:
                    try:
                        row = int(parts[1])
                        col = int(parts[2])
                        parsed_info['grid_answer'] = (row, col)
                    except ValueError:
                        pass
            elif '(' in answer and ')' in answer:
                # (X,Y) 形式
                coord_match = re.search(r'\((\d+),(\d+)\)', answer)
                if coord_match:
                    row = int(coord_match.group(1))
                    col = int(coord_match.group(2))
                    parsed_info['grid_answer'] = (row, col)
                    
            print(f" [parse_hard_question_info] Grid answer: {parsed_info.get('grid_answer')}")
        
        # First check for choices in separate 'choices' field (used in some JSON formats)
        if 'choices' in qa_item and qa_item['choices']:
            parsed_info['choices'] = qa_item['choices']
        
        # If no choices found in separate field, try extracting from text
        elif 'Please choose from:' in text:
            choices_section = text.split('Please choose from:')[1]
            # 次のセクション（Note:など）で区切る
            for delimiter in ['\nNote:', '\nIMPORTANT:', '\nHint:', '\nFor short answer']:
                if delimiter in choices_section:
                    choices_section = choices_section.split(delimiter)[0]
                    break
            
            # 選択肢を抽出（番号付きまたは単純なリスト）
            lines = choices_section.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    # 番号を除去 (1. や (23%, 87%) など) but preserve decimal numbers
                    # Only remove numbered list markers like "1. " or "1) " at the start
                    cleaned_choice = re.sub(r'^\d+\.\s+', '', line)
                    cleaned_choice = re.sub(r'^\d+\)\s+', '', cleaned_choice)
                    if cleaned_choice:
                        parsed_info['choices'].append(cleaned_choice)
        
        # If still no choices found, generate them dynamically for visualization
        if not parsed_info['choices']:
            parsed_info['choices'] = self._generate_visualization_choices(qa_item)
        
        return parsed_info

    def visualize_hard_qa_sample(self, qa_item, rgb_img, svf_img, seg_img, dsm_img=None, save_path=None, per_modality_dir=None):
        """
        ハードカテゴリQAサンプルの可視化（qa_hard_visualization.pyから移植・改良）
        
        Args:
            qa_item: dict (jsonlの1行分)
            rgb_img, svf_img, seg_img: np.ndarray or PIL.Image
            dsm_img: np.ndarray or None (DSM画像)
            save_path: 保存先 (Noneならplt.show)
            per_modality_dir: ここに指定があれば、各モダリティ(RGB/SVF/DSM/Seg)を個別PNGで保存（キャプションなし）
        """
        category = qa_item.get('category', '')
        answer = qa_item.get('answer')
        
        # 質問テキストから情報を抽出
        parsed_info = self.parse_hard_question_info(qa_item)
        
        # 画像をPIL化
        if isinstance(rgb_img, np.ndarray):
            rgb = Image.fromarray(rgb_img.astype(np.uint8))
        else:
            rgb = rgb_img.copy()
        if isinstance(seg_img, np.ndarray):
            seg = Image.fromarray(seg_img.astype(np.uint8))
        else:
            seg = seg_img.copy()
        if isinstance(dsm_img, np.ndarray):
            dsm = Image.fromarray(dsm_img.astype(np.uint8))
        else:
            dsm = dsm_img.copy()
        
        w, h = rgb.size
        
        # 描画用
        rgb_draw = ImageDraw.Draw(rgb)
        seg_draw = ImageDraw.Draw(seg)
        dsm_draw = ImageDraw.Draw(dsm)
        # カテゴリ別の特別処理
        if category == 'hard_ranking' and parsed_info['regions']:
            # ランキング結果に応じた色分け  
            # "Region A, Region B, Region C" -> ["Region A", "Region B", "Region C"]
            ranking_order = [item.strip() for item in answer.split(',')] if ',' in answer else [answer.strip()]
            colors = ['red', 'blue', 'green', 'orange', 'purple']  # より視認性の高い色に変更
            
            print(f" Ranking visualization:")
            print(f"  Answer: {answer}")
            print(f"  Ranking order: {ranking_order}")
            
            for i, region in enumerate(parsed_info['regions']):
                bbox = region.get('bbox')
                label = region.get('label', f'Region_{i}')
                actual_label = region.get('actual_label', label)
                
                # ランキング順位に応じて色を決定
                color = 'white'  # デフォルト（より視認性の高い色）
                rank_info = ""
                
                # Region A, B, C を ranking_order と照合
                if label in ranking_order:
                    rank = ranking_order.index(label)
                    if rank < len(colors):
                        color = colors[rank]
                    rank_info = f" (Rank {rank+1})"
                
                print(f"  {label}: bbox={bbox}, color={color}, rank={rank_info}")
                
                if bbox:
                    self._draw_hard_bbox(seg_draw, bbox, (w, h), color, f"{label}{rank_info}")
                    # RGB画像にも同様に描画
                    self._draw_hard_bbox(rgb_draw, bbox, (w, h), color, f"{label}{rank_info}")
            
            # 選択肢情報をテキストと点線で表示
            print(f" [hard_ranking] Choices available: {len(parsed_info['choices'])}")
            
        elif category in ['regional_svf_variability', 'sun_exposure', 'highest_region', 'urban_density', 
                         'openness_assessment', 'natural_artificial_ratio', 'scenic_quality'] and parsed_info['regions']:
            # Region-based categories の可視化
            # Distinct palette; ensure the 4th option (D) is clearly different
            colors = ['green', 'red', 'blue', 'orange', 'cyan', 'magenta']
            
            print(f" {category} visualization:")
            print(f"  Answer: {answer}")
            print(f"  Found regions: {[r['label'] for r in parsed_info['regions']]}")
            
            # debug_infoから各地域の統計情報を抽出（カテゴリ固有）
            debug_info = qa_item.get('debug_info', [])
            region_stats = {}
            
            # regional_svf_variabilityの場合
            if category == 'regional_svf_variability':
                for debug_line in debug_info:
                    match = re.search(r'Region ([ABCDEF]): mean=([\d.]+), std=([\d.]+), bbox=\[([^\]]+)\]', debug_line)
                    if match:
                        region_label = f"Region {match.group(1)}"
                        mean_val = float(match.group(2))
                        std_val = float(match.group(3))
                        region_stats[region_label] = {'mean': mean_val, 'std': std_val}
            
            # urban_densityやopenness_assessmentの場合はscoreを抽出
            elif category in ['urban_density', 'openness_assessment', 'natural_artificial_ratio', 'scenic_quality']:
                scores = qa_item.get('scores', {})
                if scores:
                    for i, (score_key, score_val) in enumerate(scores.items()):
                        region_label = f"Region {chr(65+i)}"  # A, B, C, ...
                        region_stats[region_label] = {'score': float(score_val)}
            
            for i, region in enumerate(parsed_info['regions']):
                bbox = region.get('bbox')
                label = region.get('label', f'Region_{i}')
                
                # 正解の地域は緑、不正解は他の色で交互に
                if label == answer or answer in label:
                    color = 'green'
                    info_suffix = " ✓CORRECT"
                else:
                    available_colors = [c for c in colors if c != 'green']
                    color = available_colors[i % len(available_colors)]
                    info_suffix = " ✗Wrong"
                
                # 統計情報を追加
                stats_info = ""
                if label in region_stats:
                    stats = region_stats[label]
                    if 'std' in stats:
                        stats_info = f" (std={stats['std']:.3f})"
                    elif 'score' in stats:
                        stats_info = f" (score={stats['score']:.3f})"
                
                print(f"  {label}: bbox={bbox}, color={color}, stats={stats_info}")
                
                if bbox:
                    self._draw_hard_bbox(seg_draw, bbox, (w, h), color, f"{label}{stats_info}{info_suffix}")
                    # RGB画像にも同様に描画
                    self._draw_hard_bbox(rgb_draw, bbox, (w, h), color, f"{label}{stats_info}{info_suffix}")
            
            print(f" [{category}] Regions visualized: {len(parsed_info['regions'])}")
            if parsed_info['choices']:
                # 選択肢テキストを描画
                choices_text = f"Choices: {', '.join(parsed_info['choices'][:3])}..."
                if len(parsed_info['choices']) > 3:
                    choices_text += f" (+{len(parsed_info['choices'])-3} more)"
                print(f" Drawing choices text: {choices_text}")
                self._draw_choices_text(seg_draw, (w, h), choices_text, 'white')
                
                # 不正解選択肢を点線で描画（正解以外の領域）
                self._draw_incorrect_choices_dotted(seg_draw, parsed_info, answer, (w, h))
            else:
                print("  No choices found for hard_ranking - this is why choices are not displayed")
                    
        elif category == 'hard_grid_5×5':
            # 5×5グリッドの描画とハイライト
            highlight = parsed_info.get('grid_answer')
            if highlight:
                row, col = highlight
                highlight = (row - 1, col - 1)  # 1-indexedから0-indexedに変換
            
            self._draw_hard_5×5_grid(seg_draw, (w, h), highlight)
            
            # 選択肢情報を表示
            if parsed_info['choices']:
                choices_text = f"Choices: {', '.join(parsed_info['choices'][:4])}"
                self._draw_choices_text(seg_draw, (w, h), choices_text, 'white')
                
                # 不正解選択肢を点線で表示（グリッド）
                self._draw_incorrect_grid_choices_dotted(seg_draw, parsed_info, answer, (w, h))
        
        elif category in ['sky_visibility_grid', 'building_density_grid', 'svf_extreme_grid', 'visibility_range_grid']:
            # 3×3グリッド系の可視化
            print(f" {category} - 3×3 grid visualization")
            print(f"  Answer: {answer}")
            
            # 3×3グリッドを描画
            self._draw_3×3_grid(seg_draw, (w, h), highlight=answer)
            self._draw_3×3_grid(rgb_draw, (w, h), highlight=answer)
            
            # 選択肢情報を表示
            choices = parsed_info.get('choices', [])
            if choices:
                choices_text = f"Choices: {', '.join(choices[:4])}"
                self._draw_choices_text(seg_draw, (w, h), choices_text, 'white')
                print(f" [3×3_grid] Choices available: {len(choices)}")
            else:
                print(f"  No choices found for {category} - generating from common 3×3 positions")
                # 共通の3×3位置から選択肢を生成
                common_positions = ['top left', 'top middle', 'top right', 'middle left', 
                                  'middle middle', 'middle right', 'bottom left', 'bottom middle', 'bottom right']
                # 正解を含む4個の選択肢を生成
                if answer in common_positions:
                    wrong_choices = [pos for pos in common_positions if pos != answer]
                    choices = [answer] + wrong_choices[:3]
                    choices_text = f"Choices: {', '.join(choices)}"
                    self._draw_choices_text(seg_draw, (w, h), choices_text, 'white')
            
        elif category == 'hard_pixel':
            # ピクセル値の可視化
            coordinates = parsed_info.get('coordinates')
            bbox = parsed_info.get('bbox')
            
            print(f" {category} visualization:")
            print(f"  Answer: {answer}")
            print(f"  Coordinates: {coordinates}")
            print(f"  BBox: {bbox}")
            
            if coordinates:
                x_pct, y_pct = coordinates
                
                # If bbox is available, draw the region rectangle
                if bbox:
                    xmin, ymin, xmax, ymax = bbox
                    label_text = f"SVF Region: {answer}"
                    self._draw_hard_bbox(seg_draw, bbox, (w, h), 'green', label_text)
                    self._draw_hard_bbox(rgb_draw, bbox, (w, h), 'green', label_text)
                    print(f"  Region bbox drawn: [{xmin}%, {ymin}%, {xmax}%, {ymax}%]")
                else:
                    # Single point visualization
                    label_text = f"SVF={answer}"
                    self._draw_hard_point(seg_draw, (x_pct, y_pct), (w, h), 'red', label_text)
                    self._draw_hard_point(rgb_draw, (x_pct, y_pct), (w, h), 'red', label_text)
                    print(f"  Point drawn: ({x_pct:.1f}%, {y_pct:.1f}%)")
                
                # 選択肢情報を別途表示
                if parsed_info['choices']:
                    choices_text = f"Choices: {', '.join(parsed_info['choices'][:4])}"
                    if answer in parsed_info['choices']:
                        choice_idx = parsed_info['choices'].index(answer)
                        choices_text += f" | Answer: #{choice_idx + 1}"
                    self._draw_choices_text(seg_draw, (w, h), choices_text, 'white')
                    print(f" [{category}] Choices drawn: {len(parsed_info['choices'])}")
                    
            # Region bbox表示（hard_pixelでregion情報がある場合）
            elif parsed_info['regions']:
                colors = ['green', 'red', 'blue', 'orange', 'magenta']
                print(f" {category} with regions visualization:")
                print(f"  Answer: {answer}")
                print(f"  Found regions: {[r['label'] for r in parsed_info['regions']]}")
                
                for i, region in enumerate(parsed_info['regions']):
                    bbox = region.get('bbox')
                    label = region.get('label', f'Region_{i}')
                    
                    if label == answer or answer in label:
                        color = 'green'
                        info_suffix = " ✓CORRECT"
                    else:
                        available_colors = [c for c in colors if c != 'green']
                        color = available_colors[i % len(available_colors)]
                        info_suffix = " ✗Wrong"
                    
                    print(f"  {label}: bbox={bbox}, color={color}")
                    
                    if bbox:
                        self._draw_hard_bbox(seg_draw, bbox, (w, h), color, f"{label}{info_suffix}")
                        self._draw_hard_bbox(rgb_draw, bbox, (w, h), color, f"{label}{info_suffix}")
        
        elif category == 'height_inference':
            # Height inference の可視化
            # Region bbox表示（height_inferenceでregion情報がある場合）
            if parsed_info['regions']:
                colors = ['green', 'red', 'blue', 'orange', 'magenta']
                print(f" {category} with regions visualization:")
                print(f"  Answer: {answer}")
                print(f"  Found regions: {[r['label'] for r in parsed_info['regions']]}")
                
                for i, region in enumerate(parsed_info['regions']):
                    bbox = region.get('bbox')
                    label = region.get('label', f'Region_{i}')
                    
                    if label == answer or answer in label:
                        color = 'green'
                        info_suffix = " ✓CORRECT"
                    else:
                        available_colors = [c for c in colors if c != 'green']
                        color = available_colors[i % len(available_colors)]
                        info_suffix = " ✗Wrong"
                    
                    print(f"  {label}: bbox={bbox}, color={color}")
                    
                    if bbox:
                        self._draw_hard_bbox(seg_draw, bbox, (w, h), color, f"{label}{info_suffix}")
                        self._draw_hard_bbox(rgb_draw, bbox, (w, h), color, f"{label}{info_suffix}")
            else:
                # Region情報がない場合は、通常のheight_inference表示
                print(f" {category} (no regions): Answer = {answer}")
                # 画像全体の高度情報表示などを追加可能
        
        elif category == 'height_average':
            # Height average の可視化 - regionのbboxを描画
            coordinates = parsed_info.get('coordinates')
            bbox = parsed_info.get('bbox')
            
            print(f" {category} visualization:")
            print(f"  Answer: {answer}")
            print(f"  Coordinates: {coordinates}")
            print(f"  BBox: {bbox}")
            
            if bbox:
                xmin, ymin, xmax, ymax = bbox
                label_text = f"Height Region: {answer}"
                self._draw_hard_bbox(seg_draw, bbox, (w, h), 'green', label_text)
                self._draw_hard_bbox(rgb_draw, bbox, (w, h), 'green', label_text)
                self._draw_hard_bbox(dsm_draw, bbox, (w, h), 'green', label_text)
                print(f"  Height region bbox drawn: [{xmin}%, {ymin}%, {xmax}%, {ymax}%]")
                
                # Show region info as text
                region_info = f"Height Average Region\nBBox: [{xmin:.0f}%, {ymin:.0f}%, {xmax:.0f}%, {ymax:.0f}%]\nAnswer: {answer}"
                self._draw_choices_text(seg_draw, (w, h), region_info, 'white')
        
        elif category in ['sky_visibility', 'visibility_range']:
            # Point-based categories: sky_visibility, visibility_range
            print(f" {category} - Point-based visualization")
            print(f"  Answer: {answer}")
            
            # Get answer coordinates for comparison
            answer_coord = parsed_info.get('coordinates')
            
            # Parse and draw ALL choice points (including correct answer)
            choices = parsed_info.get('choices', [])
            points = parsed_info.get('points', [])
            
            if points:
                colors = ['red', 'blue', 'orange', 'magenta']
                choice_points_drawn = 0
                
                for i, (x_pct, y_pct) in enumerate(points):
                    coord_str = f"({x_pct:.1f}%, {y_pct:.1f}%)"
                    
                    # Check if this point is the correct answer
                    if answer_coord and abs(x_pct - answer_coord[0]) < 0.1 and abs(y_pct - answer_coord[1]) < 0.1:
                        # Draw correct answer with green color and special marker
                        self._draw_hard_point(seg_draw, (x_pct, y_pct), (w, h), 'green', f"✓CORRECT: {coord_str}")
                        self._draw_hard_point(rgb_draw, (x_pct, y_pct), (w, h), 'green', f"✓CORRECT: {coord_str}")
                        print(f"  ✓ CORRECT point: ({x_pct:.1f}%, {y_pct:.1f}%) - green")
                    else:
                        # Draw incorrect choice points
                        color = colors[choice_points_drawn % len(colors)]
                        self._draw_hard_point(seg_draw, (x_pct, y_pct), (w, h), color, f"✗{coord_str}")
                        self._draw_hard_point(rgb_draw, (x_pct, y_pct), (w, h), color, f"✗{coord_str}")
                        choice_points_drawn += 1
                        print(f"  ✗ Choice point {choice_points_drawn}: ({x_pct:.1f}%, {y_pct:.1f}%) - {color}")
                
                # Draw choices text
                if choices:
                    choices_text = f"Choices: {', '.join(choices[:4])}"
                    self._draw_choices_text(seg_draw, (w, h), choices_text, 'white')
                    print(f" [{category}] All points visualized: {len(points)} total (1 correct + {choice_points_drawn} incorrect)")
            else:
                print(f"  No coordinate points found for {category}")
        
        # 全カテゴリで4枚構成（RGB + SVF + DSM + Segmentation）
        hard_categories = ['hard_ranking', 'hard_pixel', 'hard_grid_5×5', 'hard_sun_exposure', 'hard_scenic_quality', 
                          'hard_metric', 'hard_urban_analysis', 'hard_scenic_analysis', 'hard_openness_analysis']
        
        # 全カテゴリで4枚構成を使用
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # RGB画像
        axs[0, 0].imshow(rgb)
        axs[0, 0].set_title('RGB Image')
        axs[0, 0].axis('off')
        
        # SVFヒートマップ
        svf_array = np.array(svf_img)
        # If multi-channel, convert to single channel by averaging
        if svf_array.ndim == 3 and svf_array.shape[2] >= 3:
            svf_array = svf_array.mean(axis=2)
        # Normalize to [0,1] depending on dtype/range
        if svf_array.dtype == np.uint8:
            svf_display = svf_array.astype(np.float32) / 255.0
        elif svf_array.dtype == np.uint16:
            svf_display = svf_array.astype(np.float32) / 65535.0
        else:
            max_val = float(np.nanmax(svf_array)) if np.size(svf_array) > 0 else 1.0
            if max_val > 2.0:
                svf_display = svf_array.astype(np.float32) / 255.0
            else:
                svf_display = svf_array.astype(np.float32)
        svf_display = np.clip(svf_display, 0.0, 1.0)
        im_svf = axs[0, 1].imshow(svf_display, cmap=self.svf_cmap, vmin=0, vmax=1)
        axs[0, 1].set_title('SVF Heatmap')
        axs[0, 1].axis('off')
        # Inset horizontal colorbar at bottom to keep image size consistent
        try:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            # slightly reduce bar height and increase bottom borderpad to avoid title overlap with next axis
            cax_svf = inset_axes(axs[0, 1], width="90%", height="3.2%", loc='lower center',
                                 bbox_to_anchor=(0, 0.02, 1, 1), bbox_transform=axs[0, 1].transAxes, borderpad=1.0)
            plt.colorbar(im_svf, cax=cax_svf, orientation='horizontal', label='SVF Value')
        except Exception:
            plt.colorbar(im_svf, ax=axs[0, 1], orientation='horizontal', fraction=0.045, pad=0.14, label='SVF Value')
        
        # DSM画像（高さマップ）
        if dsm_img is not None:
            # statisticsのfreeform_caption_generatorと同じDSM可視化を適用
            from matplotlib.colors import LinearSegmentedColormap
            height_cmap = LinearSegmentedColormap.from_list(
                'height', ['green', 'yellow', 'orange', 'red', 'purple'], N=256)
            
            # 有効な高さ値のみを表示
            valid_dsm = np.ma.masked_invalid(dsm_img)
            valid_dsm = np.ma.masked_less(valid_dsm, 0)
            
            im_dsm = axs[1, 0].imshow(valid_dsm, cmap=height_cmap)
            axs[1, 0].set_title('DSM (Digital Surface Model)')
            axs[1, 0].axis('off')
            # Inset horizontal colorbar at bottom to keep image size consistent
            try:
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                cax_dsm = inset_axes(axs[1, 0], width="90%", height="3.2%", loc='lower center',
                                     bbox_to_anchor=(0, 0.02, 1, 1), bbox_transform=axs[1, 0].transAxes, borderpad=1.0)
                plt.colorbar(im_dsm, cax=cax_dsm, orientation='horizontal', label='Height (m)')
            except Exception:
                plt.colorbar(im_dsm, ax=axs[1, 0], orientation='horizontal', fraction=0.045, pad=0.14, label='Height (m)')
        else:
            axs[1, 0].text(0.5, 0.5, 'DSM Data\nNot Available', 
                           ha='center', va='center', transform=axs[1, 0].transAxes,
                           fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
            axs[1, 0].set_title('DSM (Digital Surface Model)')
            axs[1, 0].axis('off')
        
        # セグメンテーション画像（カラーマップ適用）
        # 元のnumpy arrayを使用してカラーマップを適用
        seg_unique_vals_for_export = None
        seg_class_names_for_export = None
        if isinstance(seg_img, np.ndarray):
            seg_cmap, unique_vals, class_names, vmax_val = self.create_segmentation_colormap(seg_img)
            if seg_cmap is not None:
                # freeform_caption_generator.pyと同じvmin/vmax設定
                im_seg = axs[1, 1].imshow(seg_img, cmap=seg_cmap, vmin=0, vmax=vmax_val)
                axs[1, 1].set_title('Segmentation + QA')
                axs[1, 1].axis('off')
                # セグメンテーションカラーバーを追加（クラス名付き）
                self.setup_segmentation_colorbar(im_seg, axs[1, 1], unique_vals, class_names)
                seg_unique_vals_for_export = unique_vals
                seg_class_names_for_export = class_names
                
                # デバッグ情報
                print(f" Segmentation: unique values = {unique_vals}, vmax = {vmax_val}")
                print(f" Class names: {[class_names.get(int(val), f'Class {int(val)}') for val in unique_vals]}")
            else:
                # カラーマップ作成に失敗した場合のフォールバック
                axs[1, 1].imshow(seg_img, cmap='tab10')
                axs[1, 1].set_title('Segmentation + QA')
                axs[1, 1].axis('off')
                print("  Segmentation colormap creation failed, using tab10")
        else:
            # seg_imgがnumpy arrayでない場合、PIL Imageから変換
            seg_array = np.array(seg_img)
            seg_cmap, unique_vals, class_names, vmax_val = self.create_segmentation_colormap(seg_array)
            if seg_cmap is not None:
                # freeform_caption_generator.pyと同じvmin/vmax設定
                im_seg = axs[1, 1].imshow(seg_array, cmap=seg_cmap, vmin=0, vmax=vmax_val)
                axs[1, 1].set_title('Segmentation + QA')
                axs[1, 1].axis('off')
                self.setup_segmentation_colorbar(im_seg, axs[1, 1], unique_vals, class_names)
                seg_unique_vals_for_export = unique_vals
                seg_class_names_for_export = class_names
                
                print(f" Segmentation (PIL→array): unique values = {unique_vals}, vmax = {vmax_val}")
                print(f" Class names: {[class_names.get(int(val), f'Class {int(val)}') for val in unique_vals]}")
            else:
                axs[1, 1].imshow(seg_array, cmap='gray')
                axs[1, 1].set_title('Segmentation + QA')
                axs[1, 1].axis('off')
                print("  Segmentation conversion failed, using grayscale")
        
        # Hard categoryの場合、matplotlibでの可視化を追加
        import matplotlib.patches as patches
        
        if category == 'hard_ranking' and parsed_info['regions']:
            # hard_ranking: bboxを描画 - 小数点対応
            ranking_order = [item.strip() for item in answer.split(',')] if ',' in answer else [answer.strip()]
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            
            for i, region in enumerate(parsed_info['regions']):
                bbox = region.get('bbox')
                label = region.get('label', f'Region_{i}')
                
                if bbox:
                    xmin, ymin, xmax, ymax = bbox
                    # パーセント座標を実際の座標に変換（小数点対応）
                    px_xmin = int(xmin / 100 * w)
                    px_ymin = int(ymin / 100 * h)
                    px_xmax = int(xmax / 100 * w)
                    px_ymax = int(ymax / 100 * h)
                    
                    # 色を決定
                    color = 'white'
                    if label in ranking_order:
                        rank = ranking_order.index(label)
                        if rank < len(colors):
                            color = colors[rank]
                    
                    # RGB、SVF、DSM、セグメンテーション画像にbboxを描画（全4パネル）
                    # 全カテゴリ2×2レイアウト: axs[0,0]=RGB, axs[0,1]=SVF, axs[1,0]=DSM, axs[1,1]=Segmentation
                    for row, col in [(0, 0), (0, 1), (1, 0), (1, 1)]:  # RGB, SVF, DSM, and Segmentation panels
                        # Emphasize only the top-ranked region (if identifiable); others stay as outlines
                        if label in ranking_order and ranking_order.index(label) == 0:
                            rect = patches.Rectangle((px_xmin, px_ymin), px_xmax-px_xmin, px_ymax-px_ymin,
                                                   linewidth=4, edgecolor=color, facecolor='none', zorder=3)
                        else:
                            rect = patches.Rectangle((px_xmin, px_ymin), px_xmax-px_xmin, px_ymax-px_ymin,
                                                   linewidth=2, edgecolor=color, facecolor='none')
                        axs[row, col].add_patch(rect)
                        if (label in ranking_order and ranking_order.index(label) == 0):
                            rect.set_path_effects([
                                patheffects.Stroke(linewidth=rect.get_linewidth()+2, foreground='white'),
                                patheffects.Normal()
                            ])
                        # 座標情報も表示（小数点第一位まで）
                        coord_text = f"{label}\n({xmin:.1f}%, {ymin:.1f}%)"
                        axs[row, col].text(px_xmin, px_ymin-5, coord_text, color=color, fontweight='bold', fontsize=10)
        
        elif category in ['regional_svf_variability', 'sun_exposure', 'highest_region', 'urban_density', 
                         'openness_assessment', 'natural_artificial_ratio', 'scenic_quality'] and parsed_info['regions']:
            # region-based categories: 地域の可視化
            debug_info = qa_item.get('debug_info', [])
            region_stats = {}
            
            # debug_infoから各地域の統計情報を抽出（カテゴリ別）
            if category == 'regional_svf_variability':
                for debug_line in debug_info:
                    match = re.search(r'Region ([ABCDEF]): mean=([\d.]+), std=([\d.]+), bbox=\[([^\]]+)\]', debug_line)
                    if match:
                        region_label = f"Region {match.group(1)}"
                        mean_val = float(match.group(2))
                        std_val = float(match.group(3))
                        region_stats[region_label] = {'mean': mean_val, 'std': std_val}
            
            # 他のカテゴリの場合はscoresから統計情報を取得
            elif 'scores' in qa_item:
                scores = qa_item['scores']
                for i, (score_key, score_val) in enumerate(scores.items()):
                    region_label = f"Region {chr(65+i)}"  # A, B, C, ...
                    region_stats[region_label] = {'score': float(score_val)}
            
            colors = ['green', 'red', 'blue', 'orange', 'magenta']
            
            for i, region in enumerate(parsed_info['regions']):
                bbox = region.get('bbox')
                label = region.get('label', f'Region_{i}')
                
                if bbox:
                    xmin, ymin, xmax, ymax = bbox
                    px_xmin = int(xmin / 100 * w)
                    px_ymin = int(ymin / 100 * h)
                    px_xmax = int(xmax / 100 * w)
                    px_ymax = int(ymax / 100 * h)
                    
                    # 正解の地域は緑、不正解は他の色で交互に
                    if label == answer or answer in label:
                        color = 'green'
                        suffix = " ✓"
                    else:
                        available_colors = [c for c in colors if c != 'green']
                        color = available_colors[i % len(available_colors)]
                        suffix = " ✗"
                    
                    # 統計情報を追加
                    stats_text = ""
                    if label in region_stats:
                        stats = region_stats[label]
                        if 'std' in stats:
                            stats_text = f"\nstd={stats['std']:.3f}"
                        elif 'score' in stats:
                            stats_text = f"\nscore={stats['score']:.3f}"
                    
                    # RGB、SVF、DSM、セグメンテーション画像にbboxを描画（全4パネル）
                    for row, col in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                        is_correct = (label == answer) or (answer in label)
                        if is_correct:
                            rect = patches.Rectangle((px_xmin, px_ymin), px_xmax-px_xmin, px_ymax-px_ymin,
                                                   linewidth=4, edgecolor=color, facecolor='none', zorder=3)
                        else:
                            rect = patches.Rectangle((px_xmin, px_ymin), px_xmax-px_xmin, px_ymax-px_ymin,
                                                   linewidth=2, edgecolor=color, facecolor='none')
                        axs[row, col].add_patch(rect)
                        if is_correct:
                            rect.set_path_effects([
                                patheffects.Stroke(linewidth=rect.get_linewidth()+2, foreground='white'),
                                patheffects.Normal()
                            ])
                        # 座標と統計情報を表示
                        coord_text = f"{label}{suffix}\n({xmin:.1f}%, {ymin:.1f}%){stats_text}"
                        axs[row, col].text(px_xmin, px_ymin-5, coord_text, color=color, fontweight='bold', fontsize=9)
        
        elif category == 'hard_pixel' and parsed_info['regions']:
            # hard_pixel with regions: 地域の可視化
            colors = ['green', 'red', 'blue', 'orange', 'magenta']
            
            for i, region in enumerate(parsed_info['regions']):
                bbox = region.get('bbox')
                label = region.get('label', f'Region_{i}')
                
                if bbox:
                    xmin, ymin, xmax, ymax = bbox
                    px_xmin = int(xmin / 100 * w)
                    px_ymin = int(ymin / 100 * h)
                    px_xmax = int(xmax / 100 * w)
                    px_ymax = int(ymax / 100 * h)
                    
                    # 正解の地域は緑、不正解は他の色で交互に
                    if label == answer or answer in label:
                        color = 'green'
                        suffix = " ✓"
                    else:
                        available_colors = [c for c in colors if c != 'green']
                        color = available_colors[i % len(available_colors)]
                        suffix = " ✗"
                    
                    # RGB、SVF、DSM、セグメンテーション画像にbboxを描画（全4パネル）
                    for row, col in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                        is_correct = (label == answer) or (answer in label)
                        if is_correct:
                            rect = patches.Rectangle((px_xmin, px_ymin), px_xmax-px_xmin, px_ymax-px_ymin,
                                                   linewidth=4, edgecolor=color, facecolor='none', zorder=3)
                        else:
                            rect = patches.Rectangle((px_xmin, px_ymin), px_xmax-px_xmin, px_ymax-px_ymin,
                                                   linewidth=2, edgecolor=color, facecolor='none')
                        axs[row, col].add_patch(rect)
                        if is_correct:
                            rect.set_path_effects([
                                patheffects.Stroke(linewidth=rect.get_linewidth()+2, foreground='white'),
                                patheffects.Normal()
                            ])
                        # 座標と統計情報を表示
                        coord_text = f"{label}{suffix}\n({xmin:.1f}%, {ymin:.1f}%)"
                        axs[row, col].text(px_xmin, px_ymin-5, coord_text, color=color, fontweight='bold', fontsize=9)
        
        elif category == 'height_inference' and parsed_info['regions']:
            # height_inference with regions: 地域の可視化
            colors = ['green', 'red', 'blue', 'orange', 'magenta']
            
            for i, region in enumerate(parsed_info['regions']):
                bbox = region.get('bbox')
                label = region.get('label', f'Region_{i}')
                
                if bbox:
                    xmin, ymin, xmax, ymax = bbox
                    px_xmin = int(xmin / 100 * w)
                    px_ymin = int(ymin / 100 * h)
                    px_xmax = int(xmax / 100 * w)
                    px_ymax = int(ymax / 100 * h)
                    
                    # 正解の地域は緑、不正解は他の色で交互に
                    if label == answer or answer in label:
                        color = 'green'
                        suffix = " ✓"
                    else:
                        available_colors = [c for c in colors if c != 'green']
                        color = available_colors[i % len(available_colors)]
                        suffix = " ✗"
                    
                    # RGB、SVF、DSM、セグメンテーション画像にbboxを描画（全4パネル）
                    # height_inferenceはDSMが特に重要なのでDSMパネルを強調
                    for row, col in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                        linewidth = 3 if (row, col) == (1, 0) else 2  # DSMパネルは太線
                        is_correct = (label == answer) or (answer in label)
                        if is_correct:
                            rect = patches.Rectangle((px_xmin, px_ymin), px_xmax-px_xmin, px_ymax-px_ymin,
                                                   linewidth=linewidth+1, edgecolor=color, facecolor='none', zorder=3)
                        else:
                            rect = patches.Rectangle((px_xmin, px_ymin), px_xmax-px_xmin, px_ymax-px_ymin,
                                                   linewidth=linewidth, edgecolor=color, facecolor='none')
                        axs[row, col].add_patch(rect)
                        if is_correct:
                            rect.set_path_effects([
                                patheffects.Stroke(linewidth=rect.get_linewidth()+2, foreground='white'),
                                patheffects.Normal()
                            ])
                        # 座標と統計情報を表示
                        coord_text = f"{label}{suffix}\n({xmin:.1f}%, {ymin:.1f}%)"
                        axs[row, col].text(px_xmin, px_ymin-5, coord_text, color=color, fontweight='bold', fontsize=9)
        
        elif category == 'hard_pixel' and parsed_info.get('coordinates'):
            # hard_pixel: 対象座標/領域を可視化 - 小数点対応
            x_pct, y_pct = parsed_info['coordinates']
            px_x = int(x_pct / 100 * w)
            px_y = int(y_pct / 100 * h)
            
            # bbox情報がある場合は矩形として描画、それ以外は円形マーカー
            if parsed_info.get('bbox'):
                xmin, ymin, xmax, ymax = parsed_info['bbox']
                px_xmin = int(xmin / 100 * w)
                px_ymin = int(ymin / 100 * h)
                px_xmax = int(xmax / 100 * w)
                px_ymax = int(ymax / 100 * h)
                
                # 全カテゴリ2×2レイアウト: axs[0,0]=RGB, axs[0,1]=SVF, axs[1,1]=Segmentation
                for row, col in [(0, 0), (0, 1), (1, 1)]:  # RGB, SVF, and Segmentation panels
                    # Draw rectangle as thick outline with halo (no fill)
                    rect = patches.Rectangle((px_xmin, px_ymin), px_xmax-px_xmin, px_ymax-px_ymin,
                                           linewidth=4, edgecolor='green', facecolor='none')
                    axs[row, col].add_patch(rect)
                    rect.set_path_effects([
                        patheffects.Stroke(linewidth=rect.get_linewidth()+2, foreground='white'),
                        patheffects.Normal()
                    ])
                    # 座標とSVF値を表示（小数点第一位まで表示）
                    axs[row, col].text(px_xmin, px_ymin-10, f"Target Region\n[{xmin:.0f}%, {ymin:.0f}%, {xmax:.0f}%, {ymax:.0f}%]\nSVF={answer}", 
                                   color='green', fontweight='bold', fontsize=10,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
            else:
                # 座標のみの場合は円形マーカー
                for row, col in [(0, 0), (0, 1), (1, 1)]:  # RGB, SVF, and Segmentation panels
                    # 円形マーカーを描画（ターゲットなので強調）
                    circle = patches.Circle((px_x, px_y), radius=12, linewidth=3, 
                                          edgecolor='red', facecolor='yellow', alpha=0.8, zorder=5)
                    axs[row, col].add_patch(circle)
                    # 座標とSVF値を表示（小数点第一位まで表示）
                    axs[row, col].text(px_x+15, px_y, f"({x_pct:.1f}%, {y_pct:.1f}%)\nSVF={answer}", 
                                   color='red', fontweight='bold', fontsize=10,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        elif category in ['sky_visibility_grid', 'building_density_grid', 'svf_extreme_grid', 'visibility_range_grid']:
            # 3×3グリッド系カテゴリの可視化
            grid_w = w / 3
            grid_h = h / 3
            
            # 3×3グリッドの位置マッピング
            position_map = {
                'top left': (0, 0), 'top middle': (0, 1), 'top right': (0, 2),
                'middle left': (1, 0), 'middle middle': (1, 1), 'middle right': (1, 2),
                'bottom left': (2, 0), 'bottom middle': (2, 1), 'bottom right': (2, 2)
            }
            
            # 正解セルの位置を特定
            answer_position = position_map.get(answer)
            
            # 全カテゴリ2×2レイアウト: axs[0,0]=RGB, axs[0,1]=SVF, axs[1,1]=Segmentation
            for ax_row, ax_col in [(0, 0), (0, 1), (1, 1)]:  # RGB, SVF, and Segmentation panels
                # 3×3グリッド線を描画
                for i in range(4):  # 0-3の4本の線
                    # 縦線
                    axs[ax_row, ax_col].axvline(x=i * grid_w, color='lightgreen', linewidth=2, alpha=0.7)
                    # 横線
                    axs[ax_row, ax_col].axhline(y=i * grid_h, color='lightgreen', linewidth=2, alpha=0.7)
                
                # 正解セルをハイライト
                if answer_position:
                    grid_row, grid_col = answer_position
                    highlight_x = grid_col * grid_w
                    highlight_y = grid_row * grid_h
                    
                    # 正解セルに赤い枠を描画
                    rect = patches.Rectangle((highlight_x, highlight_y), grid_w, grid_h,
                                           linewidth=5, edgecolor='red', facecolor='none')
                    axs[ax_row, ax_col].add_patch(rect)
                    rect.set_path_effects([
                        patheffects.Stroke(linewidth=rect.get_linewidth()+2, foreground='white'),
                        patheffects.Normal()
                    ])
                    
                    # セル中央にラベルを表示
                    center_x = highlight_x + grid_w / 2
                    center_y = highlight_y + grid_h / 2
                    axs[ax_row, ax_col].text(center_x, center_y, answer.upper().replace(' ', '\n'), 
                                           ha='center', va='center', fontsize=9, fontweight='bold',
                                           color='red', bbox=dict(boxstyle="round,pad=0.3", 
                                                                facecolor='yellow', alpha=0.8))
        
        elif category == 'hard_grid_5×5':
            # hard_grid_5×5: 5×5グリッドと正解セルを描画
            grid_answer = parsed_info.get('grid_answer')
            
            # 5×5グリッドを描画
            grid_w = w / 5
            grid_h = h / 5
            
            # 全カテゴリ2×2レイアウト: axs[0,0]=RGB, axs[0,1]=SVF, axs[1,1]=Segmentation
            for row, col in [(0, 0), (0, 1), (1, 1)]:  # RGB, SVF, and Segmentation panels
                # グリッド線を描画
                for i in range(6):  # 0-5の6本の線
                    # 縦線
                    axs[row, col].axvline(x=i*grid_w, color='cyan', linewidth=1, alpha=0.7)
                    # 横線  
                    axs[row, col].axhline(y=i*grid_h, color='cyan', linewidth=1, alpha=0.7)
                
                # 正解セルをハイライト
                if grid_answer:
                    answer_row, answer_col = grid_answer
                    # 1-indexedから0-indexedに変換
                    grid_row = answer_row - 1
                    grid_col = answer_col - 1
                    
                    if 0 <= grid_row < 5 and 0 <= grid_col < 5:
                        # ハイライト矩形を描画
                        highlight_rect = patches.Rectangle((grid_col*grid_w, grid_row*grid_h), 
                                                         grid_w, grid_h,
                                                         linewidth=5, edgecolor='red', 
                                                         facecolor='none')
                        axs[row, col].add_patch(highlight_rect)
                        highlight_rect.set_path_effects([
                            patheffects.Stroke(linewidth=highlight_rect.get_linewidth()+2, foreground='white'),
                            patheffects.Normal()
                        ])
                        
                        # セル中央に答えを表示
                        center_x = grid_col*grid_w + grid_w/2
                        center_y = grid_row*grid_h + grid_h/2
                        axs[row, col].text(center_x, center_y, f"({answer_row},{answer_col})", 
                                       ha='center', va='center', color='red', 
                                       fontweight='bold', fontsize=12,
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
        
        elif category in ['sky_visibility', 'visibility_range']:
            # sky_visibility with matplotlib: 全パネルに座標点を描画
            answer_coord = parsed_info.get('coordinates')
            points = parsed_info.get('points', [])
            
            if points:
                colors = ['red', 'blue', 'orange', 'magenta']
                choice_points_drawn = 0
                
                for i, (x_pct, y_pct) in enumerate(points):
                    px_x = int(x_pct / 100 * w)
                    px_y = int(y_pct / 100 * h)
                    coord_str = f"({x_pct:.1f}%, {y_pct:.1f}%)"
                    
                    # Check if this point is the correct answer
                    if answer_coord and abs(x_pct - answer_coord[0]) < 0.1 and abs(y_pct - answer_coord[1]) < 0.1:
                        # Draw correct answer with green color on all panels
                        for row, col in [(0, 0), (0, 1), (1, 0), (1, 1)]:  # RGB, SVF, DSM, Segmentation
                            circle = patches.Circle((px_x, px_y), radius=16, linewidth=3, 
                                                  edgecolor='green', facecolor='lightgreen', alpha=0.8, zorder=5)
                            axs[row, col].add_patch(circle)
                            axs[row, col].text(px_x+15, px_y, f"✓CORRECT\n{coord_str}", 
                                           color='green', fontweight='bold', fontsize=10,
                                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
                        print(f"  ✓ CORRECT point: ({x_pct:.1f}%, {y_pct:.1f}%) - green")
                    else:
                        # Draw incorrect choice points on all panels
                        color = colors[choice_points_drawn % len(colors)]
                        for row, col in [(0, 0), (0, 1), (1, 0), (1, 1)]:  # RGB, SVF, DSM, Segmentation
                            circle = patches.Circle((px_x, px_y), radius=10, linewidth=2, 
                                                  edgecolor=color, facecolor='none', alpha=0.8)
                            axs[row, col].add_patch(circle)
                            axs[row, col].text(px_x+15, px_y, f"✗{coord_str}", 
                                           color=color, fontweight='bold', fontsize=9,
                                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                        choice_points_drawn += 1
                        print(f"  ✗ Choice point {choice_points_drawn}: ({x_pct:.1f}%, {y_pct:.1f}%) - {color}")
                
                print(f" [{category}] All points visualized on 4 panels: {len(points)} total (1 correct + {choice_points_drawn} incorrect)")
            else:
                print(f"  No coordinate points found for {category}")
        
        
        # タイトルにプロンプトバリエーション情報を追加
        prompt_base = extract_prompt_variation(qa_item.get('text', ''))
        title = f"Q{qa_item.get('question_id','?')}: {category}"
        if prompt_base:
            title += f"\nPrompt: {prompt_base[:70]}..."
        
        # 回答情報を追加
        if "<FINAL_ANSWER>" in answer:
            # <FINAL_ANSWER> ... </FINAL_ANSWER> の間のテキストを取得
            final_answer = answer.split("<FINAL_ANSWER>")[1].split("</FINAL_ANSWER>")[0]
            title += f"\nAnswer: {final_answer[:100]}..."
        elif len(answer) > 100:
            title += f"\nAnswer: {answer[:100]}..."
        else:
            title += f"\nAnswer: {answer}"
        
        plt.suptitle(title, fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            
            # RGB単独画像（キャプションなし・オーバレイのみ）を別フォルダに保存
            rgb_overlay_dir = os.path.join(os.path.dirname(save_path), 'rgb_overlays')
            os.makedirs(rgb_overlay_dir, exist_ok=True)
            rgb_overlay_path = os.path.join(
                rgb_overlay_dir,
                os.path.basename(save_path).replace('.png', '_rgb.png')
            )
            self._save_rgb_overlays_only(axs[0, 0], rgb_overlay_path)
            
            # 追加: モダリティごとの個別保存（キャプション・テキストなし）
            if per_modality_dir:
                try:
                    os.makedirs(per_modality_dir, exist_ok=True)
                    qid = qa_item.get('question_id', '')
                    category = qa_item.get('category', '')
                    base = f"Q{qid}_{category}" if qid != '' else f"{category}"
                    # RGB
                    self._save_axis_overlays_only(axs[0, 0], os.path.join(per_modality_dir, f"{base}_rgb.png"))
                    # SVF
                    self._save_axis_overlays_only(axs[0, 1], os.path.join(per_modality_dir, f"{base}_svf.png"))
                    # DSM
                    self._save_axis_overlays_only(axs[1, 0], os.path.join(per_modality_dir, f"{base}_dsm.png"))
                    # Segmentation
                    self._save_axis_overlays_only(
                        axs[1, 1],
                        os.path.join(per_modality_dir, f"{base}_seg.png"),
                        seg_unique_vals=seg_unique_vals_for_export,
                        seg_class_names=seg_class_names_for_export,
                    )
                    print(f" モダリティ別保存（キャプションなし）: {per_modality_dir}")
                except Exception as e:
                    print(f"  モダリティ別保存に失敗: {e}")
            plt.close()
            print(f" ハードカテゴリ可視化保存: {save_path}")
            print(f" RGB(オーバレイのみ)保存: {rgb_overlay_path}")
        else:
            plt.show()
        
        return fig

    def visualize_qa_auto(self, qa_item, rgb_img, svf_img, seg_img, save_path=None, per_modality_dir=None):
        """
        質問カテゴリに応じて可視化関数を自動選択
        hard_カテゴリは統合されたハード可視化、その他は従来の可視化関数を使用
        """
        category = qa_item.get('category', '')
        
        # ハードカテゴリ判定
        hard_categories = [
            'hard_sun_exposure', 'hard_scenic_quality', 'hard_pixel', 'hard_grid_5×5',
            'hard_metric', 'hard_ranking', 'hard_urban_analysis', 'hard_scenic_analysis',
            'hard_openness_analysis', 'sky_visibility', 'visibility_range'
        ]
        
        if category in hard_categories:
            return self.visualize_hard_qa_sample(qa_item, rgb_img, svf_img, seg_img, save_path=save_path, per_modality_dir=per_modality_dir)
        else:
            # 既存の標準可視化関数を使用
            if self.visualizer:
                return self.visualizer.visualize_qa_result(qa_item, save_path)
            else:
                print(f"  標準カテゴリ {category} の可視化にはvisualizerが必要です")
                return None

    # ===== ハードカテゴリ可視化機能（qa_hard_visualization.pyから移植） =====
    
    def _save_rgb_overlays_only(self, rgb_ax, save_path):
        """Save original RGB with only overlays (no captions/labels)."""
        fig_single, ax_single = plt.subplots(1, 1, figsize=(8, 8))
        # Copy base RGB image
        for image in rgb_ax.images:
            img_array = image.get_array()
            extent = image.get_extent()
            cmap = image.get_cmap()
            ax_single.imshow(img_array, extent=extent, cmap=cmap)
            break
        # Copy only geometric overlays (no text)
        import matplotlib.patches as patches
        for patch in rgb_ax.patches:
            if isinstance(patch, patches.Rectangle):
                new_patch = patches.Rectangle(
                    patch.get_xy(), patch.get_width(), patch.get_height(),
                    linewidth=patch.get_linewidth(),
                    edgecolor=patch.get_edgecolor(),
                    facecolor='none',
                    alpha=patch.get_alpha()
                )
            elif isinstance(patch, patches.Circle):
                new_patch = patches.Circle(
                    patch.get_center(), patch.get_radius(),
                    linewidth=patch.get_linewidth(),
                    edgecolor=patch.get_edgecolor(),
                    facecolor=patch.get_facecolor(),
                    alpha=patch.get_alpha()
                )
            else:
                continue
            # keep zorder
            try:
                new_patch.set_zorder(patch.get_zorder())
            except Exception:
                pass
            ax_single.add_patch(new_patch)
        ax_single.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig_single)

    def _save_axis_overlays_only(self, src_ax, save_path, seg_unique_vals=None, seg_class_names=None):
        """Save given axis' image and geometric overlays only (no titles/text/captions)."""
        fig_single, ax_single = plt.subplots(1, 1, figsize=(8, 8))
        # Copy first image from the source axis
        im = None
        for image in src_ax.images:
            img_array = image.get_array()
            extent = image.get_extent()
            cmap = image.get_cmap()
            clim = getattr(image, 'get_clim', None)
            if callable(clim):
                vmin, vmax = image.get_clim()
                im = ax_single.imshow(img_array, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                im = ax_single.imshow(img_array, extent=extent, cmap=cmap)
            # Add colorbar if this looks like a continuous numeric map (e.g., SVF/DSM)
            try:
                import numpy as np
                if hasattr(img_array, 'mask'):
                    # handle masked arrays
                    base_array = np.asarray(img_array)
                else:
                    base_array = img_array
                is_2d = isinstance(base_array, np.ndarray) and base_array.ndim == 2
                if is_2d:
                    # Decide bar type: categorical (segmentation) vs continuous (svf/dsm)
                    unique_count = None
                    if base_array.size <= 200000:  # avoid heavy unique on huge arrays
                        unique_vals = np.unique(base_array)
                        unique_count = len(unique_vals)
                    else:
                        unique_vals = None
                    is_float = np.issubdtype(base_array.dtype, np.floating)
                    is_continuous = is_float or (unique_count is not None and unique_count > 20)
                    if im is not None:
                        if is_continuous:
                            # Single image: bottom horizontal colorbar with improved font size
                            cbar = fig_single.colorbar(im, ax=ax_single, orientation='horizontal', 
                                                     fraction=0.06, pad=0.10)
                            cbar.ax.tick_params(labelsize=10)
                        else:
                            # Categorical (segmentation) with improved legend styling
                            try:
                                cbar = fig_single.colorbar(im, ax=ax_single, orientation='horizontal', 
                                                         fraction=0.06, pad=0.10)
                                ticks_vals = seg_unique_vals if seg_unique_vals is not None else unique_vals
                                if ticks_vals is not None:
                                    cbar.set_ticks(ticks_vals)
                                    if seg_class_names is not None:
                                        tick_labels = [seg_class_names.get(int(v), f"Class {int(v)}") 
                                                     for v in ticks_vals]
                                        # Improved font size calculation for modality-based saving
                                        num_classes = len(ticks_vals)
                                        if num_classes <= 4:
                                            fontsize = 11
                                        elif num_classes <= 7:
                                            fontsize = 10
                                        else:
                                            fontsize = 9
                                        cbar.set_ticklabels(tick_labels, fontsize=fontsize, fontweight='bold')
                                    else:
                                        cbar.set_ticklabels([str(int(v)) for v in ticks_vals], 
                                                          fontsize=10, fontweight='bold')
                            except Exception:
                                try:
                                    cbar = fig_single.colorbar(im, ax=ax_single, orientation='horizontal', 
                                                             fraction=0.06, pad=0.10)
                                    cbar.ax.tick_params(labelsize=10)
                                except Exception:
                                    pass
            except Exception:
                pass
            break
        # Copy only geometric patches (no text)
        import matplotlib.patches as patches
        for patch in src_ax.patches:
            new_patch = None
            if isinstance(patch, patches.Rectangle):
                new_patch = patches.Rectangle(
                    patch.get_xy(), patch.get_width(), patch.get_height(),
                    linewidth=patch.get_linewidth(),
                    edgecolor=patch.get_edgecolor(),
                    facecolor='none',
                    alpha=patch.get_alpha(),
                )
            elif isinstance(patch, patches.Circle):
                new_patch = patches.Circle(
                    patch.get_center(), patch.get_radius(),
                    linewidth=patch.get_linewidth(),
                    edgecolor=patch.get_edgecolor(),
                    facecolor=patch.get_facecolor(),
                    alpha=patch.get_alpha(),
                )
            if new_patch is not None:
                try:
                    new_patch.set_zorder(patch.get_zorder())
                except Exception:
                    pass
                ax_single.add_patch(new_patch)
        ax_single.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig_single)

    def _draw_correct_only_overlay(self, ax, qa_item, img_w, img_h, color='green', label_text=None):
        """Draw only the correct choice (bbox/point/grid) onto ax. Returns True if drawn.
        Optionally annotate with label_text (e.g., "#1")."""
        category = qa_item.get('category', '')
        answer = qa_item.get('answer', '') or ''
        parsed_info = self.parse_hard_question_info(qa_item)
        drew = False
        def _rect(xmin, ymin, xmax, ymax, lw=5):
            px_xmin = int(xmin / 100 * img_w)
            px_ymin = int(ymin / 100 * img_h)
            px_xmax = int(xmax / 100 * img_w)
            px_ymax = int(ymax / 100 * img_h)
            rect = patches.Rectangle((px_xmin, px_ymin), px_xmax - px_xmin, px_ymax - px_ymin,
                                     linewidth=lw, edgecolor=color, facecolor='none', zorder=4)
            rect.set_path_effects([patheffects.Stroke(linewidth=lw+2, foreground='white'), patheffects.Normal()])
            ax.add_patch(rect)
            return (px_xmin, px_ymin)
        def _circle(x_pct, y_pct, r=14, lw=4):
            px_x = int(x_pct / 100 * img_w)
            px_y = int(y_pct / 100 * img_h)
            circ = patches.Circle((px_x, px_y), radius=r, linewidth=lw, edgecolor=color,
                                  facecolor='none', zorder=5)
            circ.set_path_effects([patheffects.Stroke(linewidth=lw+2, foreground='white'), patheffects.Normal()])
            ax.add_patch(circ)
            return (px_x, px_y)
        # Region-type with labels
        if parsed_info.get('regions'):
            target_label = None
            if category == 'hard_ranking' and answer:
                target_label = answer.split(',')[0].strip()
            else:
                target_label = answer
            for region in parsed_info['regions']:
                bbox = region.get('bbox')
                label = region.get('label', '')
                if bbox and (label == target_label or (target_label and target_label in label)):
                    xy = _rect(*bbox)
                    drew = True
                    ax.text(xy[0]+4, xy[1]-8, label_text or f"Q{qa_item.get('question_id','?')}", color='black', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85), zorder=6)
                    break
        # Pixel/point-type
        elif category.startswith('hard_pixel') or ('coordinates' in parsed_info and parsed_info['coordinates']):
            if parsed_info.get('bbox'):
                xy = _rect(*parsed_info['bbox'])
                drew = True
                ax.text(xy[0]+4, xy[1]-8, label_text or f"Q{qa_item.get('question_id','?')}", color='black', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85), zorder=6)
            else:
                coords = parsed_info.get('coordinates')
                if coords and len(coords) == 2:
                    x_pct, y_pct = coords
                    xy = _circle(x_pct, y_pct)
                    drew = True
                    ax.text(xy[0]+6, xy[1]-6, label_text or f"Q{qa_item.get('question_id','?')}", color='black', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85), zorder=6)
                else:
                    # No drawable data for this item
                    drew = False
        # 3×3 grid
        elif category in ['sky_visibility_grid', 'building_density_grid', 'svf_extreme_grid', 'visibility_range_grid']:
            position_map = {
                'top left': (0, 0), 'top middle': (0, 1), 'top right': (0, 2),
                'middle left': (1, 0), 'middle middle': (1, 1), 'middle right': (1, 2),
                'bottom left': (2, 0), 'bottom middle': (2, 1), 'bottom right': (2, 2)
            }
            if answer in position_map:
                grid_w = img_w / 3
                grid_h = img_h / 3
                r, c = position_map[answer]
                xmin = c * grid_w / img_w * 100
                ymin = r * grid_h / img_h * 100
                xmax = (c+1) * grid_w / img_w * 100
                ymax = (r+1) * grid_h / img_h * 100
                xy = _rect(xmin, ymin, xmax, ymax)
                drew = True
                ax.text(xy[0]+4, xy[1]-8, label_text or f"Q{qa_item.get('question_id','?')}", color='black', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85), zorder=6)
        # 5×5 grid
        elif category == 'hard_grid_5×5' and parsed_info.get('grid_answer'):
            grid_w = img_w / 5
            grid_h = img_h / 5
            r, c = parsed_info['grid_answer']
            r -= 1; c -= 1
            xmin = c * grid_w / img_w * 100
            ymin = r * grid_h / img_h * 100
            xmax = (c+1) * grid_w / img_w * 100
            ymax = (r+1) * grid_h / img_h * 100
            xy = _rect(xmin, ymin, xmax, ymax)
            drew = True
            ax.text(xy[0]+4, xy[1]-8, label_text or f"Q{qa_item.get('question_id','?')}", color='black', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85), zorder=6)
        return drew

    def export_grouped_correct_overlays(self, output_dir='grouped_correct_overlays', max_per_image=5):
        """Group QA by image and export figures overlaying only correct answers for 3-5 items per image."""
        if not self.image_dir:
            print("  image_dir is required for grouped export")
            return
        os.makedirs(output_dir, exist_ok=True)
        # Group by image
        image_to_items = defaultdict(list)
        for item in self.data:
            if item.get('image'):
                image_to_items[item['image']].append(item)
        colors = ['#00FF00', '#FF0000', '#00AAFF', '#FF9900', '#AA00FF']
        for image_name, items in image_to_items.items():
            rgb_path = os.path.join(self.image_dir, image_name)
            if not os.path.exists(rgb_path):
                print(f"  missing RGB for grouping: {rgb_path}")
                continue
            try:
                # Decode JP2 as standard 8-bit RGB for consistent visualization
                rgb_img = Image.open(rgb_path).convert('RGB')
            except Exception as e:
                print(f"  failed to open {rgb_path}: {e}")
                continue
            img_w, img_h = rgb_img.size
            # chunk into batches
            for start in range(0, len(items), max_per_image):
                batch = items[start:start+max_per_image]
                if not batch:
                    continue
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                ax.imshow(rgb_img)
                ax.axis('off')
                drawn_any = False
                for idx, qa_item in enumerate(batch):
                    color = colors[idx % len(colors)]
                    label_text = f"#{idx+1}"
                    drawn = self._draw_correct_only_overlay(ax, qa_item, img_w, img_h, color=color, label_text=label_text)
                    drawn_any = drawn_any or drawn
                base = os.path.basename(image_name)
                base_no_ext = base.replace('_rgb.jp2', '')
                out_name = f"{base_no_ext}_group_{start//max_per_image+1}.png"
                out_path = os.path.join(output_dir, out_name)
                plt.tight_layout()
                plt.savefig(out_path, dpi=200, bbox_inches='tight')
                plt.close(fig)
                if drawn_any:
                    print(f" 保存: {out_path} ({len(batch)} items)")
                else:
                    print(f"  {out_path}: no overlays drawn")

    def _save_rgb_panel_only(self, original_fig, rgb_ax, save_path, title, qa_item, answer):
        """
        左上のRGB panel（選択肢表示含む）のみを単独の画像として保存
        """
        # 新しいfigureを作成してRGB panelの内容をコピー
        fig_single, ax_single = plt.subplots(1, 1, figsize=(8, 8))
        
        # RGB axisの画像データを取得してコピー
        for image in rgb_ax.images:
            # 画像データを直接取得
            img_array = image.get_array()
            extent = image.get_extent()
            cmap = image.get_cmap()
            
            # 新しいaxにimshow
            ax_single.imshow(img_array, extent=extent, cmap=cmap)
        
        # パッチ（bbox、円形マーカーなど）をコピー
        import matplotlib.patches as patches
        for patch in rgb_ax.patches:
            # パッチの種類によって処理を分岐
            if isinstance(patch, patches.Rectangle):
                new_patch = patches.Rectangle(
                    patch.get_xy(), patch.get_width(), patch.get_height(),
                    linewidth=patch.get_linewidth(),
                    edgecolor=patch.get_edgecolor(),
                    facecolor=patch.get_facecolor(),
                    alpha=patch.get_alpha()
                )
            elif isinstance(patch, patches.Circle):
                new_patch = patches.Circle(
                    patch.get_center(), patch.get_radius(),
                    linewidth=patch.get_linewidth(),
                    edgecolor=patch.get_edgecolor(),
                    facecolor=patch.get_facecolor(),
                    alpha=patch.get_alpha()
                )
            else:
                # その他のパッチ型への対応
                continue
            
            ax_single.add_patch(new_patch)
        
        # テキストアノテーションをコピー  
        for text in rgb_ax.texts:
            # Copy only safe bbox properties to avoid passing conflicting args (e.g., width/height/transform)
            bbox_props = None
            bbox_patch = text.get_bbox_patch()
            if bbox_patch:
                raw_props = bbox_patch.properties()
                allowed_keys = {
                    'boxstyle', 'facecolor', 'edgecolor', 'linewidth', 'linestyle', 'alpha',
                    'pad', 'mutation_aspect', 'mutation_scale', 'joinstyle', 'capstyle'
                }
                bbox_props = {k: raw_props[k] for k in allowed_keys if k in raw_props}
                # Try to preserve boxstyle from the original patch
                try:
                    get_bs = getattr(bbox_patch, 'get_boxstyle', None)
                    if callable(get_bs):
                        bs = bbox_patch.get_boxstyle()
                        if bs is not None:
                            bbox_props['boxstyle'] = bs
                except Exception:
                    pass

            ax_single.text(
                text.get_position()[0], text.get_position()[1], 
                text.get_text(),
                color=text.get_color(),
                fontweight=text.get_fontweight(),
                fontsize=text.get_fontsize(),
                bbox=bbox_props
            )
        
        # 選択肢情報を画像下部に追加（再パースして可視化時の抽出に合わせる）
        category = qa_item.get('category', '')
        parsed_info_for_choices = self.parse_hard_question_info(qa_item)
        choices = parsed_info_for_choices.get('choices', []) or qa_item.get('choices', [])
        if choices:
            choices_text = f"Choices: {', '.join(choices[:4])}"
            if len(choices) > 4:
                choices_text += f" (+{len(choices)-4} more)"
            ax_single.text(0.02, 0.02, choices_text, transform=ax_single.transAxes,
                         fontsize=12, color='white', fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.8))
        
        # タイトル設定
        prompt_base = extract_prompt_variation(qa_item.get('text', ''))
        single_title = f"Q{qa_item.get('question_id','?')}: {category} (RGB)"
        if prompt_base:
            single_title += f"\nPrompt: {prompt_base[:60]}..."
        single_title += f"\nAnswer: {answer}"
        
        ax_single.set_title(single_title, fontsize=11)
        ax_single.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig_single)
    
    def _draw_hard_bbox(self, draw, bbox, img_shape, color, label=None):
        """ハードカテゴリ用BBox描画"""
        w, h = img_shape
        xmin, ymin, xmax, ymax = bbox
        px_xmin = int(xmin / 100 * w)
        px_ymin = int(ymin / 100 * h)
        px_xmax = int(xmax / 100 * w)
        px_ymax = int(ymax / 100 * h)
        draw.rectangle([px_xmin, px_ymin, px_xmax, px_ymax], outline=color, width=3)
        if label:
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except (IOError, OSError):
                font = ImageFont.load_default()
            draw.text((px_xmin+5, px_ymin+5), label, fill=color, font=font)

    def _parse_coordinate_answer(self, answer):
        """Parse coordinate answer in format '(x%, y%)'"""
        if answer and '(' in answer and '%' in answer:
            match = re.search(r'\((\d+\.?\d*)%, (\d+\.?\d*)%\)', answer)
            if match:
                x_pct, y_pct = map(float, match.groups())
                return (x_pct, y_pct)
        return None

    def _draw_hard_point(self, draw, point, img_shape, color, label=None):
        """ハードカテゴリ用点描画"""
        w, h = img_shape
        x, y = point
        px_x = int(x / 100 * w)
        px_y = int(y / 100 * h)
        r = 6
        draw.ellipse([px_x - r, px_y - r, px_x + r, px_y + r], fill=color, outline='black')
        if label:
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except (IOError, OSError):
                font = ImageFont.load_default()
            draw.text((px_x + 10, px_y), label, fill=color, font=font)

    def _draw_hard_5×5_grid(self, draw, img_shape, highlight=None, highlight_color='red'):
        """ハードカテゴリ用5×5グリッド描画"""
        w, h = img_shape
        grid_h, grid_w = h // 5, w // 5
        
        # グリッド線を描画
        for i in range(1, 5):
            draw.line([(0, i * grid_h), (w, i * grid_h)], fill='lightblue', width=2)
            draw.line([(i * grid_w, 0), (i * grid_w, h)], fill='lightblue', width=2)
        
        # 外枠も描画
        draw.rectangle([0, 0, w-1, h-1], outline='lightblue', width=2)
        
        # ハイライト処理
        if highlight:
            row, col = highlight
            x1 = col * grid_w
            y1 = row * grid_h
            x2 = x1 + grid_w
            y2 = y1 + grid_h
            draw.rectangle([x1, y1, x2, y2], outline=highlight_color, width=8)
            
            # ハイライトセルにラベル
            center_x = x1 + grid_w // 2
            center_y = y1 + grid_h // 2
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except (IOError, OSError):
                font = ImageFont.load_default()
            draw.text((center_x-20, center_y), f"grid_{row+1}_{col+1}", fill=highlight_color, font=font)

    def _draw_3×3_grid(self, draw, img_shape, highlight=None, highlight_color='red'):
        """3×3グリッド描画（sky_visibility_grid等用）"""
        w, h = img_shape
        grid_h, grid_w = h // 3, w // 3
        
        # グリッド線を描画
        for i in range(1, 3):
            draw.line([(0, i * grid_h), (w, i * grid_h)], fill='lightgreen', width=3)
            draw.line([(i * grid_w, 0), (i * grid_w, h)], fill='lightgreen', width=3)
        
        # 外枠も描画
        draw.rectangle([0, 0, w-1, h-1], outline='lightgreen', width=3)
        
        # セルラベルを描画（位置参照用）
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except (IOError, OSError):
            font = ImageFont.load_default()
            
        positions = [
            ('top left', 0, 0), ('top middle', 0, 1), ('top right', 0, 2),
            ('middle left', 1, 0), ('middle middle', 1, 1), ('middle right', 1, 2),
            ('bottom left', 2, 0), ('bottom middle', 2, 1), ('bottom right', 2, 2)
        ]
        
        for label, row, col in positions:
            x1 = col * grid_w
            y1 = row * grid_h
            center_x = x1 + grid_w // 2
            center_y = y1 + grid_h // 2
            
            # 正解セルをハイライト
            if highlight and label == highlight:
                x2 = x1 + grid_w
                y2 = y1 + grid_h
                draw.rectangle([x1, y1, x2-1, y2-1], outline=highlight_color, width=6)
                
                # 正解ラベルを強調表示
                draw.text((center_x-30, center_y-10), label.upper(), fill=highlight_color, font=font)
            else:
                # 通常ラベル
                draw.text((center_x-25, center_y-5), label[:3], fill='white', font=font)

    def _draw_choices_text(self, draw, img_shape, choices_text, color='white'):
        """選択肢テキストを画像下部に描画"""
        w, h = img_shape
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except (IOError, OSError):
            font = ImageFont.load_default()
        
        # テキストの背景を黒で塗りつぶし
        bbox = draw.textbbox((0, 0), choices_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        # 画像下部に配置
        x = 10
        y = h - text_h - 10
        
        # 背景を描画
        draw.rectangle([x-5, y-2, x+text_w+5, y+text_h+2], fill='black', outline='gray')
        # テキストを描画
        draw.text((x, y), choices_text, fill=color, font=font)
    
    def _draw_incorrect_choices_dotted(self, draw, parsed_info, answer, img_shape):
        """不正解選択肢を点線で描画（hard_ranking用）"""
        w, h = img_shape
        
        # 正解順列を取得
        correct_order = [item.strip() for item in answer.split(',')] if ',' in answer else [answer.strip()]
        
        # 不正解選択肢を描画（正解以外のランキング順序）
        incorrect_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink']
        
        for i, region in enumerate(parsed_info['regions']):
            bbox = region.get('bbox')
            label = region.get('label', f'Region_{i}')
            
            if bbox and label not in correct_order:
                # 不正解領域を点線で描画
                color_idx = i % len(incorrect_colors)
                self._draw_dotted_bbox(draw, bbox, (w, h), incorrect_colors[color_idx], f"{label} (不正解)")
    
    def _draw_incorrect_grid_choices_dotted(self, draw, parsed_info, answer, img_shape):
        """不正解グリッド選択肢を点線で描画（hard_grid_5×5用）"""
        w, h = img_shape
        
        # 正解グリッドを取得
        correct_grid = None
        if answer.startswith('grid_'):
            parts = answer.split('_')
            if len(parts) == 3:
                try:
                    correct_grid = (int(parts[1]), int(parts[2]))
                except ValueError:
                    pass
        
        if correct_grid and parsed_info['choices']:
            grid_w = w / 5
            grid_h = h / 5
            
            # 不正解選択肢を点線で描画
            for choice in parsed_info['choices']:
                if choice != answer and choice.startswith('grid_'):
                    parts = choice.split('_')
                    if len(parts) == 3:
                        try:
                            row, col = int(parts[1]), int(parts[2])
                            # 1-indexedから0-indexedに変換
                            grid_row = row - 1
                            grid_col = col - 1
                            
                            if 0 <= grid_row < 5 and 0 <= grid_col < 5:
                                x1 = grid_col * grid_w
                                y1 = grid_row * grid_h
                                x2 = x1 + grid_w
                                y2 = y1 + grid_h
                                
                                # 点線で描画
                                self._draw_dotted_rectangle(draw, [x1, y1, x2, y2], 'lightblue')
                        except ValueError:
                            continue
    
    def _draw_incorrect_pixel_choices_dotted(self, draw, parsed_info, answer, correct_coords, img_shape):
        """不正解ピクセル選択肢を点線で描画（hard_pixel用）"""
        w, h = img_shape
        
        if parsed_info['choices'] and correct_coords:
            # 不正解選択肢の値を周囲に表示
            try:
                correct_val = float(answer)
                x_pct, y_pct = correct_coords
                px_x = int(x_pct / 100 * w)
                px_y = int(y_pct / 100 * h)
                
                # 不正解選択肢を周囲に表示
                incorrect_choices = [choice for choice in parsed_info['choices'] if choice != answer]
                for i, choice in enumerate(incorrect_choices[:3]):  # 最大3個を表示
                    try:
                        choice_val = float(choice)
                        angle = i * 120  # 120度あきで配置
                        offset_x = 30 * np.cos(np.radians(angle))
                        offset_y = 30 * np.sin(np.radians(angle))
                        
                        choice_x = px_x + offset_x
                        choice_y = px_y + offset_y
                        
                        # 点線で選択肢を描画
                        self._draw_dotted_circle(draw, (choice_x, choice_y), 6, 'lightblue')
                        
                        # 選択肢の値を表示
                        try:
                            font = ImageFont.truetype("arial.ttf", 10)
                        except (IOError, OSError):
                            font = ImageFont.load_default()
                        draw.text((choice_x+10, choice_y), f"{choice_val:.3f}", fill='lightblue', font=font)
                    except ValueError:
                        continue
            except ValueError:
                pass
    
    def _draw_dotted_bbox(self, draw, bbox, img_shape, color, label=None):
        """点線でBBoxを描画"""
        w, h = img_shape
        xmin, ymin, xmax, ymax = bbox
        px_xmin = int(xmin / 100 * w)
        px_ymin = int(ymin / 100 * h)
        px_xmax = int(xmax / 100 * w)
        px_ymax = int(ymax / 100 * h)
        
        # 点線で矩形を描画
        self._draw_dotted_rectangle(draw, [px_xmin, px_ymin, px_xmax, px_ymax], color)
        
        if label:
            try:
                font = ImageFont.truetype("arial.ttf", 10)
            except (IOError, OSError):
                font = ImageFont.load_default()
            draw.text((px_xmin+5, px_ymin+5), label, fill=color, font=font)
    
    def _draw_dotted_rectangle(self, draw, coords, color):
        """点線で矩形を描画"""
        x1, y1, x2, y2 = coords
        dash_length = 5
        
        # 上辺
        x = x1
        while x < x2:
            draw.line([(x, y1), (min(x + dash_length, x2), y1)], fill=color, width=2)
            x += dash_length * 2
        
        # 下辺  
        x = x1
        while x < x2:
            draw.line([(x, y2), (min(x + dash_length, x2), y2)], fill=color, width=2)
            x += dash_length * 2
        
        # 左辺
        y = y1
        while y < y2:
            draw.line([(x1, y), (x1, min(y + dash_length, y2))], fill=color, width=2)
            y += dash_length * 2
        
        # 右辺
        y = y1
        while y < y2:
            draw.line([(x2, y), (x2, min(y + dash_length, y2))], fill=color, width=2)
            y += dash_length * 2
    
    def _draw_dotted_circle(self, draw, center, radius, color):
        """点線で円を描画"""
        cx, cy = center
        # 簡易的な点線円を描画（小さな弧で近似）
        for angle in range(0, 360, 30):  # 30度ごとに点を描画
            x = cx + radius * np.cos(np.radians(angle))
            y = cy + radius * np.sin(np.radians(angle))
            draw.ellipse([x-2, y-2, x+2, y+2], fill=color)

    def visualize_sample_questions(self, num_samples=5, save_dir=None, samples_per_category=3, per_modality_dir=None, force_first_n=50):
        """サンプル質問の可視化（ハードカテゴリ対応）
        force_first_n: 最初のN問は必ず出力（num_samplesに関わらず）
        """
        if not self.visualizer and not self.image_dir:
            print("  画像ディレクトリが指定されていないため、可視化をスキップします")
            return
            
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        # まず最初のN問を必ず含める
        samples = []
        seen_ids = set()
        if force_first_n and self.data:
            mandatory = self.data[:min(force_first_n, len(self.data))]
            for qa in mandatory:
                qid = qa.get('question_id')
                if qid not in seen_ids:
                    samples.append(qa)
                    seen_ids.add(qid)
        
        # 各カテゴリから追加選択
        categories = list(set(item['category'] for item in self.data))
        
        import random
        for category in categories:
            category_items = [item for item in self.data if item['category'] == category]
            category_items = category_items[:10] if len(category_items) > 10 else category_items
            if category_items:
                # Shuffle category_items to ensure random selection
                random.shuffle(category_items)
                num_to_select = min(samples_per_category, len(category_items))
                for qa in category_items[:num_to_select]:
                    qid = qa.get('question_id')
                    if qid not in seen_ids:
                        samples.append(qa)
                        seen_ids.add(qid)
        # 指定数まで追加
        if len(samples) < num_samples:
            remaining = [item for item in self.data if item.get('question_id') not in seen_ids]
            for qa in remaining[:max(0, num_samples - len(samples))]:
                samples.append(qa)
                seen_ids.add(qa.get('question_id'))
            
        print(f"  {len(samples)}個のサンプル質問を可視化中（{len(categories)}カテゴリ、各カテゴリ最大{samples_per_category}個）...")
        
        for i, qa_item in enumerate(tqdm(samples)):
            try:
                category = qa_item.get('category', '')
                
                # すべてのカテゴリで4枚構成可視化を使用
                hard_categories = [
                    'hard_sun_exposure', 'hard_scenic_quality', 'hard_pixel', 'hard_grid_5×5',
                    'hard_metric', 'hard_ranking', 'hard_urban_analysis', 'hard_scenic_analysis',
                    'hard_openness_analysis'
                ]
                
                # 全カテゴリで4枚構成可視化（特にheightを含むカテゴリでDSMが重要）
                if self.image_dir:
                    # 画像を読み込んで4枚構成可視化を実行（選択肢点線表示付き）
                    image_name = qa_item.get('image', '')
                    print(f" 画像パス: {image_name}")
                    if image_name:
                        # RGB画像の読み込み
                        rgb_path = os.path.join(self.image_dir, image_name)
                        if os.path.exists(rgb_path):
                            # Decode JP2 as standard 8-bit RGB for consistent visualization
                            rgb_img = Image.open(rgb_path).convert('RGB')
                            
                            # SVF画像の読み込み（visualize_qa_results.pyと同じロジック）
                            svf_img = None
                            if self.svf_dir:
                                # Convert RGB path to SVF path
                                # duisburg/335_5696_rgb.jp2 -> 335_5696_dem_svf_umep.tif
                                base_name = os.path.basename(image_name)
                                area_name = image_name.split('/')[0] if '/' in image_name else None
                                
                                if '_rgb.jp2' in base_name:
                                    svf_name = base_name.replace('_rgb.jp2', '_dem_svf_umep.tif')
                                    svf_path = os.path.join(self.svf_dir, area_name, svf_name)
                                    
                                    if os.path.exists(svf_path):
                                        try:
                                            import cv2
                                            svf_img = cv2.imread(svf_path, cv2.IMREAD_UNCHANGED)
                                            if svf_img is not None:
                                                svf_img = svf_img.astype(np.float32)
                                        except Exception as e:
                                            print(f"SVF読み込みエラー {svf_path}: {e}")
                                            svf_img = np.zeros((rgb_img.height, rgb_img.width))
                                    else:
                                        print(f"  SVF画像が見つかりません: {svf_path}")
                                        svf_img = np.zeros((rgb_img.height, rgb_img.width))
                                else:
                                    svf_img = np.zeros((rgb_img.height, rgb_img.width))
                            else:
                                svf_img = np.zeros((rgb_img.height, rgb_img.width))
                            
                            # DSM画像の読み込み（RGB/SEGと同じフォルダにある）
                            dsm_img = None
                            if self.image_dir:
                                dsm_path = image_name.replace('_rgb.jp2', '_dem.tif')
                                full_dsm_path = os.path.join(self.image_dir, dsm_path)
                                
                                if not os.path.exists(full_dsm_path):
                                    # Try without directory structure
                                    dsm_name = os.path.basename(dsm_path)
                                    full_dsm_path = os.path.join(self.image_dir, dsm_name)
                                
                                if os.path.exists(full_dsm_path):
                                    try:
                                        import cv2
                                        dsm_img = cv2.imread(full_dsm_path, cv2.IMREAD_UNCHANGED)
                                        if dsm_img is not None:
                                            dsm_img = dsm_img.astype(np.float32)
                                    except Exception as e:
                                        print(f"DSM読み込みエラー {full_dsm_path}: {e}")
                                        dsm_img = None
                                else:
                                    print(f"  DSM画像が見つかりません: {full_dsm_path}")
                                    dsm_img = None
                            
                            # セグメンテーション画像の読み込み
                            seg_path = image_name.replace('_rgb.jp2', '_seg.tif')
                            full_seg_path = os.path.join(self.image_dir, seg_path)
                            
                            if not os.path.exists(full_seg_path):
                                # Try without directory structure
                                seg_name = os.path.basename(seg_path)
                                full_seg_path = os.path.join(self.image_dir, seg_name)
                                
                            if os.path.exists(full_seg_path):
                                seg_img = Image.open(full_seg_path)
                            else:
                                print(f"  セグメンテーション画像が見つかりません: {full_seg_path}")
                                # セグメンテーション画像がない場合はRGB画像をコピー
                                seg_img = rgb_img.copy()
                            
                            if save_dir:
                                # Create filename with Q number and category info
                                question_id = qa_item.get('question_id', i+1)
                                image_name = qa_item.get('image', 'unknown')
                                
                                # Extract base name without extension
                                if '/' in image_name:
                                    base_name = os.path.basename(image_name)
                                else:
                                    base_name = image_name
                                
                                # Remove extension and create filename
                                if '_rgb.jp2' in base_name:
                                    base_coord = base_name.replace('_rgb.jp2', '')
                                else:
                                    base_coord = os.path.splitext(base_name)[0]
                                
                                # Create filename: Q{number}_{category}_{coordinates}_rgb.png
                                filename = f"Q{question_id:03d}_{category}_{base_coord}_rgb.png"
                                save_path = os.path.join(save_dir, filename)
                                self.visualize_hard_qa_sample(qa_item, rgb_img, svf_img, seg_img, dsm_img, save_path, per_modality_dir=per_modality_dir)
                                print(f" 4枚構成可視化保存（選択肢点線付き）: {save_path}")
                            else:
                                self.visualize_hard_qa_sample(qa_item, rgb_img, svf_img, seg_img, dsm_img, per_modality_dir=per_modality_dir)
                        else:
                            print(f"  画像ファイルが見つかりません: {rgb_path}")
                else:
                    print(f"  画像ディレクトリが指定されていないため、{category}の可視化をスキップします")
                    
            except Exception as e:
                print(f"  Q{qa_item['question_id']}の可視化エラー: {e}")
                import traceback
                traceback.print_exc()
                
    def export_category_text_files(self, output_dir="category_text_output"):
        """カテゴリ別に質問内容と回答内容をテキストファイルで出力"""
        os.makedirs(output_dir, exist_ok=True)
        
        # カテゴリ別にデータをグループ化
        category_data = defaultdict(list)
        
        for item in self.data:
            category = item.get('category', 'unknown')
            category_data[category].append(item)
        
        print(f"カテゴリ別テキストファイル出力中...")
        
        exported_files = []
        for category, items in category_data.items():
            # ファイル名で使えない文字を適切な文字に置換
            safe_category = re.sub(r'[<>:"/\\|?*]', '_', category)
            file_path = os.path.join(output_dir, f"{safe_category}.txt")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"\n" + "=" * 80 + "\n")
                f.write(f"カテゴリ: {category}\n")
                f.write(f"総質問数: {len(items)}\n")
                f.write(f"出力日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                
                for i, item in enumerate(items, 1):
                    question_id = item.get('question_id', i)
                    question_text = item.get('text', '')
                    answer = item.get('answer', '')
                    image = item.get('image', '')
                    choices = item.get('choices', [])
                    
                    f.write(f"Q{i:03d} (ID: {question_id})\n")
                    f.write("-" * 40 + "\n")
                    
                    # 質問内容（改行付きで読みやすく）
                    f.write(" 質問内容:\n")
                    f.write(f"{question_text}\n\n")
                    
                    # 選択肢がある場合
                    if choices:
                        f.write("選択肢:\n")
                        for j, choice in enumerate(choices, 1):
                            f.write(f"  {j}. {choice}\n")
                        f.write("\n")
                    
                    # 回答
                    f.write(f"正解: {answer}\n")
                    
                    # 画像情報
                    if image:
                        f.write(f"画像: {image}\n")
                    
                    f.write("\n" + "=" * 60 + "\n\n")
                
                # サマリ情報をファイル末尾に追加
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"カテゴリ '{category}' のサマリ\n")
                f.write("=" * 80 + "\n")
                f.write(f"総質問数: {len(items)}\n")
                f.write(f"使用された画像数: {len(set(item.get('image', '') for item in items if item.get('image')))}\n")
                
                # プロンプトバリエーション情報
                prompt_types = []
                for item in items:
                    prompt_base = extract_prompt_variation(item.get('text', ''))
                    if prompt_base:
                        prompt_types.append(prompt_base)
                unique_prompts = len(set(prompt_types))
                f.write(f"プロンプトバリエーション数: {unique_prompts}\n")
                
                # 回答タイプの分析
                answer_types = Counter()
                for item in items:
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
                
                f.write(f"回答タイプ分布: {dict(answer_types)}\n")
                f.write(f"出力完了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            exported_files.append(file_path)
            print(f" {category}: {len(items)}項目を{file_path}に出力")
        
        # 全体サマリファイルを作成
        summary_file = os.path.join(output_dir, "_category_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("カテゴリ別テキストファイル出力サマリ\n")
            f.write("=" * 80 + "\n")
            f.write(f"出力日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"元データ: {self.jsonl_path}\n")
            f.write(f"総質問数: {len(self.data)}\n")
            f.write(f"カテゴリ数: {len(category_data)}\n")
            f.write(f"出力ファイル数: {len(exported_files)}\n\n")
            
            # カテゴリ別サマリ
            f.write("カテゴリ別質問数:\n")
            for category, items in sorted(category_data.items(), key=lambda x: len(x[1]), reverse=True):
                f.write(f"  {category}: {len(items)}項目\n")
            
            f.write("\n出力ファイル一覧:\n")
            for file_path in sorted(exported_files):
                file_name = os.path.basename(file_path)
                f.write(f"  {file_name}\n")
        
        exported_files.append(summary_file)
        
        print(f"\n カテゴリ別テキストファイル出力完了!")
        print(f" 出力ディレクトリ: {output_dir}")
        print(f" サマリファイル: {summary_file}")
        print(f"  合計ファイル数: {len(exported_files)}")
        
        return exported_files
                
    def run_full_analysis(self, output_dir="qa-vis", per_modality_dir=None, force_first_n=50):
        """完全な分析実行（プロンプトバリエーション分析を含む）"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(" QA出力の完全分析を開始...")
        
        # 1. 基本分析
        self.analyze_categories()
        
        # 2. プロンプトバリエーション分析
        prompt_report_path = os.path.join(output_dir, "prompt_variation_report.txt")
        self.create_prompt_variation_report(prompt_report_path)
        
        # 3. 品質チェック
        quality_issues = self.check_question_quality()
        
        # 4. 総合レポート生成
        report_path = os.path.join(output_dir, "quality_report.txt")
        self.create_quality_report(report_path)
        
        # 5. グラフ生成
        plot_path = os.path.join(output_dir, "category_distribution.png")
        self.create_category_distribution_plot(plot_path)
        
        # 6. バイアス検証
        bias_results = self.analyze_biases()
        bias_report_path = os.path.join(output_dir, "bias_analysis_report.txt") 
        self.create_bias_report(bias_results, bias_report_path)
        
        # 7. バイアス可視化
        bias_plot_path = os.path.join(output_dir, "bias_analysis_plots.png")
        self.create_bias_visualization(bias_results, bias_plot_path)
        
        # 8. サンプル可視化
        sample_dir = os.path.join(output_dir, "sample_visualizations")
        # モダリティ別保存先のデフォルト
        if per_modality_dir is None:
            per_modality_dir = os.path.join(sample_dir, "per_modality")
        self.visualize_sample_questions(save_dir=sample_dir, per_modality_dir=per_modality_dir, force_first_n=force_first_n)
        
        # 8.5 同一画像で3-5問ずつ正解のみ重畳した画像を追加出力
        grouped_dir = os.path.join(output_dir, "grouped_correct_overlays")
        self.export_grouped_correct_overlays(output_dir=grouped_dir, max_per_image=5)
        
        # 9. カテゴリ別テキストファイル出力
        text_dir = os.path.join(output_dir, "category_text_files")
        exported_text_files = self.export_category_text_files(text_dir)
            
        print(f" 分析完了! 結果は {output_dir} に保存されました")
        print(f" バイアス分析結果: {bias_report_path}")
        print(f" カテゴリ別テキストファイル: {len(exported_text_files)}個のファイルを{text_dir}に出力")
        
        return {
            'total_questions': len(self.data),
            'quality_issues': len(quality_issues),
            'categories': len(set(item['category'] for item in self.data)),
            'output_dir': output_dir,
            'exported_text_files': exported_text_files
        }

    def create_segmentation_colormap(self, seg_img):
        """Create colormap for segmentation image (freeform_caption_generator.pyと統一)"""
        if seg_img is None:
            return None, None, None
        
        # セグメンテーション用のカラーマップ作成（freeform_caption_generator.pyと同じ）
        seg_colors = {
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
        
        # クラス名のマッピング（freeform_caption_generator.pyと統一）
        class_names = {
            0: 'Background',
            1: 'Forest', 
            2: 'Water',
            3: 'Agricultural',
            4: 'Urban',
            5: 'Grassland',
            6: 'Railway',
            7: 'Highway',
            8: 'Airport',
            9: 'Roads',
            10: 'Buildings'
        }
        
        from matplotlib.colors import ListedColormap
        
        # セグメンテーション画像の実際の値を確認
        unique_vals = np.unique(seg_img)
        print(f" Segmentation unique values: {unique_vals}")
        
        # 動的な色数決定（freeform_caption_generator.pyと同じロジック）
        max_val = int(unique_vals.max()) if len(unique_vals) > 0 else 10
        num_colors = max(11, max_val + 1)
        
        # 色リストを動的に生成
        color_list = []
        for i in range(num_colors):
            if i in seg_colors:
                color_list.append(np.array(seg_colors[i])/255.0)
            else:
                color_list.append(np.array([255, 255, 255])/255.0)
        
        cmap = ListedColormap(color_list)
        
        # vmin/vmaxも適切に設定
        vmax_val = max(10, max_val)
        
        return cmap, unique_vals, class_names, vmax_val
    
    def setup_segmentation_colorbar(self, im_seg, ax, unique_vals, class_names):
        """Create discrete colorbar showing only land cover classes present in the image"""
        import numpy as np
        from matplotlib.colors import ListedColormap, BoundaryNorm
        from matplotlib.cm import ScalarMappable
        
        # Sort unique values as integers
        vals_sorted = sorted([int(v) for v in np.array(unique_vals).astype(int).tolist()])
        vmin, vmax = im_seg.get_clim()
        
        # Extract corresponding colors for a new discrete colormap
        colors_used = [im_seg.cmap((val - vmin) / (vmax - vmin) if vmax != vmin else 0.0) 
                      for val in vals_sorted]
        cmap_used = ListedColormap(colors_used)
        boundaries = np.arange(len(vals_sorted) + 1) - 0.5
        norm = BoundaryNorm(boundaries, cmap_used.N)
        sm = ScalarMappable(cmap=cmap_used, norm=norm)
        sm.set_array([])
        
        # Create inset colorbar at bottom
        try:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            cax = inset_axes(ax, width="90%", height="4%", loc='lower center',
                             bbox_to_anchor=(0, 0.02, 1, 1), bbox_transform=ax.transAxes, 
                             borderpad=1.0)
            cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
        except Exception:
            cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.045, pad=0.14)
        
        # Set label and ticks
        cbar.set_label('Land Cover Class', fontsize=12, fontweight='bold')
        cbar.set_ticks(range(len(vals_sorted)))
        tick_labels = [class_names.get(int(v), f'Class {int(v)}') for v in vals_sorted]
        
        # Improved font size calculation for better readability
        base_fontsize = 10
        if len(vals_sorted) <= 3:
            fontsize = base_fontsize + 2
        elif len(vals_sorted) <= 6:
            fontsize = base_fontsize
        elif len(vals_sorted) <= 8:
            fontsize = base_fontsize - 1
        else:
            fontsize = base_fontsize - 2
        
        fontsize = max(8, fontsize)  # Minimum readable size
        cbar.set_ticklabels(tick_labels, fontsize=fontsize, fontweight='bold')
        
        return cbar



    def analyze_biases(self):
        """QAデータセットのバイアス分析（カテゴリ別詳細分析付き）"""
        import scipy.stats as stats
        
        bias_results = {
            'position_bias': {},
            'spatial_bias': {},
            'score_bias': {},
            'category_bias': {},
            'category_detailed': {},  # 新追加：カテゴリ別詳細分析
            'overall_assessment': {}
        }
        
        # 1. 選択肢位置バイアス分析
        correct_positions = []
        for item in self.data:
            answer = item.get('answer', '')
            choices = item.get('choices', [])
            
            if answer in choices:
                correct_idx = choices.index(answer)
                correct_positions.append(correct_idx)
        
        if correct_positions:
            # カイ二乗検定による均等性テスト
            position_counts = Counter(correct_positions)
            expected_freq = len(correct_positions) / len(set(correct_positions))
            observed_freqs = [position_counts.get(i, 0) for i in range(max(position_counts.keys()) + 1)]
            
            try:
                chi2_stat, p_value = stats.chisquare(observed_freqs)
                bias_results['position_bias'] = {
                    'chi2_statistic': float(chi2_stat),
                    'p_value': float(p_value),
                    'distribution': dict(position_counts),
                    'is_biased': p_value < 0.05,
                    'severity': 'HIGH' if p_value < 0.01 else 'MEDIUM' if p_value < 0.05 else 'LOW'
                }
            except:
                bias_results['position_bias'] = {'error': 'Could not compute chi-square test'}
        
        # 2. 空間分布バイアス分析
        spatial_coords = []
        for item in self.data:
            if 'choices_coords' in item:
                coords = item['choices_coords']
                answer = item.get('answer', '')
                choices = item.get('choices', [])
                
                if answer in choices:
                    correct_idx = choices.index(answer)
                    if correct_idx < len(coords):
                        spatial_coords.append(coords[correct_idx])
        
        if spatial_coords:
            x_coords = [coord[0] for coord in spatial_coords]
            y_coords = [coord[1] for coord in spatial_coords]
            
            # 中央バイアステスト
            center_x, center_y = 50, 50
            distances_from_center = [np.sqrt((x-center_x)**2 + (y-center_y)**2) for x, y in zip(x_coords, y_coords)]
            avg_distance = np.mean(distances_from_center)
            
            # エッジバイアステスト  
            edge_threshold = 15  # 画像端から15%以内
            edge_points = sum(1 for x, y in zip(x_coords, y_coords) 
                            if x < edge_threshold or x > 100-edge_threshold or 
                                y < edge_threshold or y > 100-edge_threshold)
            edge_ratio = edge_points / len(spatial_coords)
            
            bias_results['spatial_bias'] = {
                'avg_distance_from_center': float(avg_distance),
                'edge_ratio': float(edge_ratio),
                'total_coords': len(spatial_coords),
                'center_bias': avg_distance < 25,  # 平均距離が25%未満なら中央バイアス
                'edge_bias': edge_ratio > 0.4,    # 40%以上が端ならエッジバイアス
                'x_distribution': {'mean': float(np.mean(x_coords)), 'std': float(np.std(x_coords))},
                'y_distribution': {'mean': float(np.mean(y_coords)), 'std': float(np.std(y_coords))}
            }
        
        # 3. スコア分布バイアス分析
        score_gaps = []
        for item in self.data:
            if 'scores' in item:
                scores = [float(s) for s in item['scores'].values()]
                if len(scores) >= 2:
                    sorted_scores = sorted(scores, reverse=True)
                    gap = sorted_scores[0] - sorted_scores[1]  # 1位と2位の差
                    score_gaps.append(gap)
        
        if score_gaps:
            gap_mean = np.mean(score_gaps)
            gap_std = np.std(score_gaps)
            gap_entropy = stats.entropy(np.histogram(score_gaps, bins=10)[0] + 1e-10)
            
            bias_results['score_bias'] = {
                'gap_mean': float(gap_mean),
                'gap_std': float(gap_std), 
                'gap_entropy': float(gap_entropy),
                'uniform_gaps': gap_std < 0.05,  # 標準偏差が小さすぎる場合
                'total_samples': len(score_gaps)
            }
        
        # 4. カテゴリ別バイアス分析（すべてのカテゴリを対象）
        category_position_bias = defaultdict(list)
        category_all_items = defaultdict(list)  # 全アイテムを記録
        
        for item in self.data:
            category = item.get('category', '')
            answer = item.get('answer', '')
            choices = item.get('choices', [])
            
            category_all_items[category].append(item)
            
            # choicesフィールドがない場合は自動生成を試行
            if not choices:
                choices = self._generate_visualization_choices(item)
                
            if choices and answer in choices:
                correct_idx = choices.index(answer)
                category_position_bias[category].append(correct_idx)
            elif choices:
                # 答えがchoicesに見つからない場合でも、推測で位置を決定
                # A, B, C, D形式や数値形式の場合
                if answer.startswith('Region '):
                    region_letter = answer.split()[-1]
                    if region_letter in ['A', 'B', 'C', 'D']:
                        correct_idx = ord(region_letter) - ord('A')
                        if correct_idx < len(choices):
                            category_position_bias[category].append(correct_idx)
        
        category_bias_summary = {}
        category_detailed_analysis = {}  # 新追加：詳細分析
        
        # すべてのカテゴリを分析対象にする（サンプル数制限を緩和）
        for category in category_all_items.keys():
            positions = category_position_bias.get(category, [])
            
            if len(positions) >= 3:  # 最低3サンプルがあれば分析
                pos_counts = Counter(positions)
                try:
                    expected = len(positions) / len(set(positions))
                    observed = [pos_counts.get(i, 0) for i in range(max(pos_counts.keys()) + 1)]
                    chi2_stat, p_value = stats.chisquare(observed)
                    
                    # 基本分析
                    category_bias_summary[category] = {
                        'p_value': float(p_value),
                        'is_biased': p_value < 0.05,
                        'sample_size': len(positions),
                        'distribution': dict(pos_counts)
                    }
                    
                    # 詳細分析
                    # 最も多い選択肢位置と偏差を計算
                    most_common_pos = max(pos_counts, key=pos_counts.get)
                    most_common_count = pos_counts[most_common_pos]
                    bias_percentage = (most_common_count / len(positions)) * 100
                    
                    # 均等分布からの偏差を計算
                    expected_per_choice = len(positions) / len(set(positions))
                    max_deviation = max(abs(count - expected_per_choice) for count in pos_counts.values())
                    
                    category_detailed_analysis[category] = {
                        'sample_size': len(positions),
                        'num_choices': len(set(positions)),
                        'position_distribution': dict(pos_counts),
                        'most_biased_position': most_common_pos,
                        'bias_percentage': bias_percentage,
                        'expected_percentage': 100.0 / len(set(positions)),
                        'max_deviation': max_deviation,
                        'chi2_statistic': float(chi2_stat),
                        'p_value': float(p_value),
                        'severity': 'HIGH' if p_value < 0.01 else 'MEDIUM' if p_value < 0.05 else 'LOW'
                    }
                    
                except Exception as e:
                    category_bias_summary[category] = {'error': f'Statistical test failed: {str(e)}'}
                    category_detailed_analysis[category] = {'error': f'Analysis failed: {str(e)}'}
            else:
                # サンプル数が不足している場合でも基本情報を記録
                total_items = len(category_all_items[category])
                analyzed_items = len(positions)
                
                error_msg = f'Insufficient data for statistical test (analyzed: {analyzed_items}/{total_items})'
                if analyzed_items == 0:
                    error_msg = f'No valid choices found ({total_items} total items)'
                    
                category_bias_summary[category] = {'error': error_msg, 'total_items': total_items}
                category_detailed_analysis[category] = {
                    'error': error_msg,
                    'total_items': total_items,
                    'analyzed_items': analyzed_items,
                    'sample_size': analyzed_items
                }
        
        bias_results['category_bias'] = category_bias_summary
        bias_results['category_detailed'] = category_detailed_analysis
        
        # 5. 総合評価
        overall_issues = []
        if bias_results['position_bias'].get('is_biased', False):
            overall_issues.append('Position bias detected')
        if bias_results['spatial_bias'].get('center_bias', False):
            overall_issues.append('Center spatial bias detected')
        if bias_results['spatial_bias'].get('edge_bias', False):
            overall_issues.append('Edge spatial bias detected')
        if bias_results['score_bias'].get('uniform_gaps', False):
            overall_issues.append('Uniform score gaps detected')
        
        biased_categories = sum(1 for cat_result in category_bias_summary.values() 
                                if cat_result.get('is_biased', False))
        
        bias_results['overall_assessment'] = {
            'total_issues': len(overall_issues),
            'issue_list': overall_issues,
            'biased_categories_count': biased_categories,
            'total_categories': len(category_bias_summary),
            'overall_bias_severity': 'HIGH' if len(overall_issues) >= 3 else 'MEDIUM' if len(overall_issues) >= 1 else 'LOW'
        }
        
        return bias_results

    def create_bias_report(self, bias_results, report_path):
        """バイアス分析レポート作成（カテゴリ別詳細分析付き）"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("QA Dataset Bias Analysis Report (Category-Specific Analysis)\n")
            f.write("=" * 70 + "\n\n")
            
            # 1. 総合評価
            overall = bias_results['overall_assessment']
            f.write(f" Overall Bias Assessment: {overall['overall_bias_severity']}\n")
            f.write(f" Total Issues Detected: {overall['total_issues']}\n")
            f.write(f" Issue List: {', '.join(overall['issue_list']) if overall['issue_list'] else 'None'}\n\n")
            
            # 2. 選択肢位置バイアス
            f.write("1. Choice Position Bias Analysis\n")
            f.write("-" * 40 + "\n")
            pos_bias = bias_results['position_bias']
            if 'p_value' in pos_bias:
                f.write(f"   Chi-square p-value: {pos_bias['p_value']:.6f}\n")
                f.write(f"   Is Biased: {'Yes' if pos_bias['is_biased'] else 'No'}\n")
                f.write(f"   Severity: {pos_bias['severity']}\n")
                f.write(f"   Distribution: {pos_bias['distribution']}\n")
            else:
                f.write(f"   Error: {pos_bias.get('error', 'Unknown error')}\n")
            f.write("\n")
            
            # 3. 空間分布バイアス
            f.write("2. Spatial Distribution Bias Analysis\n")
            f.write("-" * 40 + "\n")
            spatial = bias_results['spatial_bias']
            if spatial:
                f.write(f"   Average Distance from Center: {spatial.get('avg_distance_from_center', 'N/A'):.2f}%\n")
                f.write(f"   Edge Ratio: {spatial.get('edge_ratio', 'N/A'):.3f}\n")
                f.write(f"   Center Bias: {'Yes' if spatial.get('center_bias', False) else 'No'}\n")
                f.write(f"   Edge Bias: {'Yes' if spatial.get('edge_bias', False) else 'No'}\n")
                f.write(f"   Total Coordinates: {spatial.get('total_coords', 0)}\n")
            f.write("\n")
            
            # 4. スコア分布バイアス
            f.write("3. Score Distribution Bias Analysis\n")
            f.write("-" * 40 + "\n")
            score = bias_results['score_bias']
            if score:
                f.write(f"   Gap Mean: {score.get('gap_mean', 'N/A'):.4f}\n")
                f.write(f"   Gap Std: {score.get('gap_std', 'N/A'):.4f}\n")
                f.write(f"   Gap Entropy: {score.get('gap_entropy', 'N/A'):.4f}\n")
                f.write(f"   Uniform Gaps: {'Yes' if score.get('uniform_gaps', False) else 'No'}\n")
            f.write("\n")
            
            # 5. カテゴリ別バイアス詳細分析
            f.write("4. Category-wise Detailed Bias Analysis\n")
            f.write("-" * 50 + "\n")
            cat_detailed = bias_results.get('category_detailed', {})
            
            # サマリー統計
            total_categories = len(cat_detailed)
            biased_categories = sum(1 for result in cat_detailed.values() 
                                   if result.get('severity') in ['HIGH', 'MEDIUM'])
            f.write(f"   Total Categories Analyzed: {total_categories}\n")
            f.write(f"   Biased Categories: {biased_categories}\n")
            f.write(f"   Bias Rate: {biased_categories/total_categories*100:.1f}%\n\n")
            
            # 各カテゴリの詳細
            for category, result in sorted(cat_detailed.items(), 
                                         key=lambda x: x[1].get('p_value', 1.0)):
                f.write(f"   Category: {category}\n")
                if 'error' in result:
                    f.write(f"     {result['error']}\n")
                else:
                    f.write(f"     Sample Size: {result['sample_size']}\n")
                    f.write(f"     Number of Choices: {result['num_choices']}\n")
                    f.write(f"     Chi-square p-value: {result['p_value']:.6f}\n")
                    f.write(f"     Bias Severity: {result['severity']}\n")
                    f.write(f"     Most Biased Position: {result['most_biased_position']} ({result['bias_percentage']:.1f}%)\n")
                    f.write(f"     Expected Percentage: {result['expected_percentage']:.1f}%\n")
                    f.write(f"     Max Deviation: {result['max_deviation']:.2f}\n")
                    f.write(f"     Position Distribution: {result['position_distribution']}\n")
                f.write("\n")
            
            # 旧フォーマットとの互換性維持
            f.write("4.1 Legacy Category Bias Summary\n")
            f.write("-" * 40 + "\n")
            cat_bias = bias_results['category_bias']
            for category, result in cat_bias.items():
                f.write(f"   Category: {category}\n")
                if 'p_value' in result:
                    f.write(f"     P-value: {result['p_value']:.6f}\n")
                    f.write(f"     Is Biased: {'Yes' if result['is_biased'] else 'No'}\n")
                    f.write(f"     Sample Size: {result['sample_size']}\n")
                else:
                    f.write(f"     {result.get('error', 'No data')}\n")
                f.write("\n")
        
        print(f" バイアス分析レポート保存: {report_path}")

    def create_bias_visualization(self, bias_results, plot_path):
        """バイアス分析の可視化（カテゴリ別詳細分析付き）"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('QA Dataset Bias Analysis (Category-Specific)', fontsize=16, fontweight='bold')
        
        # 1. 選択肢位置分布
        ax1 = axes[0, 0]
        pos_bias = bias_results['position_bias']
        if 'distribution' in pos_bias:
            positions = list(pos_bias['distribution'].keys())
            counts = list(pos_bias['distribution'].values())
            
            bars = ax1.bar(positions, counts, color='skyblue', alpha=0.7)
            ax1.set_title(f"Choice Position Distribution\\n(p-value: {pos_bias.get('p_value', 'N/A'):.4f})")
            ax1.set_xlabel('Choice Position (0=A, 1=B, 2=C, 3=D)')
            ax1.set_ylabel('Frequency')
            ax1.grid(axis='y', alpha=0.3)
            
            # 期待値線を追加
            expected = sum(counts) / len(counts)
            ax1.axhline(y=expected, color='red', linestyle='--', alpha=0.8, label=f'Expected: {expected:.1f}')
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'No position data available', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title("Choice Position Distribution")
        
        # 2. 空間分布ヒートマップ
        ax2 = axes[0, 1]
        spatial = bias_results['spatial_bias']
        if spatial and 'total_coords' in spatial:
            # 簡易ヒートマップ（実際の座標データが必要）
            ax2.text(0.5, 0.5, f"Spatial Distribution\\nCenter Distance: {spatial.get('avg_distance_from_center', 'N/A'):.1f}%\\nEdge Ratio: {spatial.get('edge_ratio', 'N/A'):.3f}", 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title("Spatial Bias Summary")
            
            # バイアス警告
            if spatial.get('center_bias', False):
                ax2.text(0.5, 0.2, " CENTER BIAS DETECTED", ha='center', va='center', 
                        transform=ax2.transAxes, color='red', fontweight='bold')
            if spatial.get('edge_bias', False):
                ax2.text(0.5, 0.3, " EDGE BIAS DETECTED", ha='center', va='center',
                        transform=ax2.transAxes, color='red', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No spatial data available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title("Spatial Distribution")
        
        # 3. スコア分布
        ax3 = axes[1, 0]
        score = bias_results['score_bias']
        if score and 'gap_mean' in score:
            # スコアギャップのヒストグラム風表示
            ax3.bar(['Gap Mean', 'Gap Std', 'Gap Entropy'], 
                    [score['gap_mean'], score['gap_std'], score['gap_entropy']], 
                    color=['green', 'orange', 'purple'], alpha=0.7)
            ax3.set_title("Score Distribution Metrics")
            ax3.set_ylabel('Value')
            
            if score.get('uniform_gaps', False):
                ax3.text(0.5, 0.8, " UNIFORM GAPS DETECTED", ha='center', va='center',
                        transform=ax3.transAxes, color='red', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No score data available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title("Score Distribution")
        
        # 4. カテゴリバイアスサマリー
        ax4 = axes[1, 1]
        cat_bias = bias_results['category_bias']
        if cat_bias:
            categories = list(cat_bias.keys())
            biased_status = ['Biased' if result.get('is_biased', False) else 'OK' 
                            for result in cat_bias.values()]
            
            biased_count = sum(1 for status in biased_status if status == 'Biased')
            ok_count = len(biased_status) - biased_count
            
            ax4.pie([biased_count, ok_count], labels=['Biased', 'OK'], 
                    colors=['red', 'green'], autopct='%1.1f%%', startangle=90)
            ax4.set_title(f"Category Bias Status\\n({len(categories)} categories)")
        else:
            ax4.text(0.5, 0.5, 'No category data available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title("Category Bias Status")
        
        # 5. カテゴリ別バイアス詳細（新追加）
        ax5 = axes[0, 2]
        cat_detailed = bias_results.get('category_detailed', {})
        if cat_detailed:
            # バイアス重要度の高いカテゴリを上位10個表示
            sorted_categories = sorted([(cat, result) for cat, result in cat_detailed.items() 
                                      if 'p_value' in result and result['p_value'] < 0.1],
                                     key=lambda x: x[1]['p_value'])
            
            if sorted_categories:
                categories = [cat[:15] + '...' if len(cat) > 15 else cat for cat, _ in sorted_categories[:10]]
                p_values = [result['p_value'] for _, result in sorted_categories[:10]]
                
                colors = ['red' if p < 0.01 else 'orange' if p < 0.05 else 'yellow' for p in p_values]
                
                bars = ax5.barh(range(len(categories)), [-np.log10(p) for p in p_values], color=colors)
                ax5.set_yticks(range(len(categories)))
                ax5.set_yticklabels(categories)
                ax5.set_xlabel('-log10(p-value)')
                ax5.set_title('Category Bias Severity\\n(Higher = More Biased)')
                ax5.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
                ax5.axvline(x=-np.log10(0.01), color='darkred', linestyle='-', alpha=0.7, label='p=0.01')
                ax5.legend()
                ax5.grid(axis='x', alpha=0.3)
            else:
                ax5.text(0.5, 0.5, 'No significantly biased\\ncategories found', ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Category Bias Severity')
        else:
            ax5.text(0.5, 0.5, 'No detailed category data', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Category Bias Severity')
        
        # 6. 最も偏ったカテゴリの選択肢分布（新追加）
        ax6 = axes[1, 2]
        if cat_detailed:
            # 最もp値が小さい（最も偏った）カテゴリを選択
            most_biased = min([(cat, result) for cat, result in cat_detailed.items() 
                              if 'p_value' in result], 
                             key=lambda x: x[1]['p_value'], default=(None, None))
            
            if most_biased[0]:
                cat_name, result = most_biased
                dist = result['position_distribution']
                positions = list(dist.keys())
                counts = list(dist.values())
                
                bars = ax6.bar(positions, counts, color='lightcoral', alpha=0.7)
                ax6.set_title(f'Most Biased Category: {cat_name[:20]}\\n(p={result["p_value"]:.4f})')
                ax6.set_xlabel('Choice Position (0=A, 1=B, 2=C, 3=D)')
                ax6.set_ylabel('Frequency')
                
                # 期待値線を追加
                expected = sum(counts) / len(counts)
                ax6.axhline(y=expected, color='red', linestyle='--', alpha=0.8, label=f'Expected: {expected:.1f}')
                ax6.legend()
                ax6.grid(axis='y', alpha=0.3)
            else:
                ax6.text(0.5, 0.5, 'No biased categories found', ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('Most Biased Category')
        else:
            ax6.text(0.5, 0.5, 'No detailed category data', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Most Biased Category')
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f" バイアス可視化保存: {plot_path}")
        plt.close()
        
        # カテゴリ別詳細分析の結果を標準出力にも表示
        print("\n カテゴリ別バイアス分析結果:")
        cat_detailed = bias_results.get('category_detailed', {})
        for category, result in sorted(cat_detailed.items(), 
                                     key=lambda x: x[1].get('p_value', 1.0)):
            if 'p_value' in result:
                severity_icon = '' if result['severity'] == 'HIGH' else '🟡' if result['severity'] == 'MEDIUM' else '🟢'
                print(f"  {severity_icon} {category}: p={result['p_value']:.6f}, "
                      f"最多位置={result['most_biased_position']}({result['bias_percentage']:.1f}%), "
                      f"サンプル数={result['sample_size']}")
def extract_prompt_variation(question_text):
    """最初の\nまでの部分から数値・座標を除いたプロンプトを返す"""
    if not question_text:
        return ""
    first_line = question_text.split('\n')[0].strip()
    # 数値や座標だけの行は除外
    if re.fullmatch(r'[-+]?\d*\.?\d+%?', first_line) or re.search(r'\d+%|\(.*%.*\)', first_line):
        return ""
    return first_line
def main():
    """使用例（ハードカテゴリ対応・プロンプトバリエーション分析付き）"""
    # パスを設定
    # date = "0620"
    # version = "medium_questions_train"
    # hard = "hr0.25"
    # jsonl_path = f"svf_qa_output_revised0606/svf_{version}_{date}_combined_{hard}.json"
    jsonl_path = "svf_qa_output_revised0621/svf_medium_detailed_test_0622_combined_hr0.201_max2.json"
    import argparse
    
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='QA出力分析ツール')
    parser.add_argument('--json_path', type=str, 
                       default="svf_qa_output_revised0621/svf_15x_medium_answers_train_mixed_hr0.303_0624_hr0.303.jsonl", #svf_medium_detailed_test_0622_combined_hr0.201_max2.json",
                       help='分析対象のJSONファイルパス')
    parser.add_argument('--image_dir', type=str, 
                       default="../SynRS3D/GeoNRW_dsm",
                       help='画像ディレクトリのパス')
    parser.add_argument('--svf_dir', type=str, 
                       default="../SynRS3D/GeoNRW_dsm/svf/skyview_umep_test",
                       help='SVFディレクトリのパス')
    parser.add_argument('--per_modality_dir', type=str,
                       default=None,
                       help='各モダリティ画像を個別保存するディレクトリ（キャプションなし）')
    
    args = parser.parse_args()
    jsonl_path = args.json_path
    # jsonl_path = "svf_qa_output_revised0621/svf_15x_small_answers_train_mixed_hr0.305_0625_hr0.305.jsonl"
    image_dir = args.image_dir if os.path.exists(args.image_dir) else "../../SynRS3D/GeoNRW_dsm"
    svf_dir = args.svf_dir if os.path.exists(args.svf_dir) else "../../SynRS3D/GeoNRW_dsm/svf/skyview_umep_test"
    
    # trainが含まれる場合のSVFディレクトリ調整
    if "train" in jsonl_path:
        svf_dir = svf_dir.replace("skyview_umep_test", "skyview_umep_train")
    # "svf_qa_output_revised0621/svf_medium_answers_test_0621_combined_hr0.2.jsonl"
    if "train" in jsonl_path:
        svf_dir = svf_dir.replace("skyview_umep_test", "skyview_umep_train")
    print(" ハードカテゴリ対応 QA出力分析ツール（プロンプトバリエーション分析付き）")
    print(f" 分析対象: {jsonl_path}")
    
    # 分析実行
    analyzer = QAOutputAnalyzer(jsonl_path, image_dir, svf_dir)
    
    # 完全分析実行
    parsed = jsonl_path.split("/")[-1].split(".json")[0]
    results = analyzer.run_full_analysis(f"qa_vis_{parsed}", per_modality_dir=args.per_modality_dir, force_first_n=50)
    
    print("\n 分析サマリー:")
    print(f"  総質問数: {results['total_questions']}")
    print(f"  品質問題: {results['quality_issues']}個")
    print(f"  カテゴリ数: {results['categories']}")
    print(f"  出力ディレクトリ: {results['output_dir']}")
    
    print("\n 新機能:")
    print("   qa_hard_visualization.py統合: 外部依存なし")
    print("   プロンプトバリエーション分析: テンプレート多様性評価")
    print("   ハードカテゴリ可視化: hard_pixel, hard_grid_5×5, hard_ranking対応")
    print("   品質スコア向上: プロンプト多様性を含む5次元評価")
    print("   統合レポート: 全分析結果を一元化")

if __name__ == "__main__":
    main() 