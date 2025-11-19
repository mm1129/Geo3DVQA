"""
Metrics and Prompt Exporter for QA Generation Verification

This module exports metrics and prompts used in QA generation to readable formats
for manual verification that generated answers match the underlying scene data.
"""

import json
import yaml
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np


class MetricsPromptExporter:
    """Exports metrics and prompts used in QA generation for verification"""
    
    def __init__(self, output_dir: str = "verification_outputs", debug: bool = False):
        self.output_dir = output_dir
        self.debug = debug
        self.exported_data = []
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.debug:
            print(f" MetricsPromptExporter initialized - output dir: {self.output_dir}")
    
    def capture_qa_generation_data(self, scene_id: str, category: str, 
                                 metrics: Dict[str, Any], question: str, 
                                 answer: str, prompt_data: Optional[Dict[str, Any]] = None,
                                 scene_stats: Optional[Any] = None,
                                 data_sources: Optional[Dict[str, Any]] = None) -> None:
        """
        Capture QA generation data for later export
        
        Args:
            scene_id: Scene identifier
            category: Analysis category (e.g., 'urban_development_application')
            metrics: Computed metrics for the scene/category
            question: Generated question
            answer: Generated answer
            prompt_data: GPT prompt information (optional)
            scene_stats: Scene statistics object (optional)
            data_sources: Data source information for verification traceability (optional)
        """
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        # Prepare data for export
        export_data = {
            'scene_id': scene_id,
            'category': category,
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat(),
            'metrics': convert_numpy(metrics) if metrics else {},
            'prompt_data': convert_numpy(prompt_data) if prompt_data else {},
            'scene_statistics': {},
            'data_sources': convert_numpy(data_sources) if data_sources else {}
        }
        
        # Extract scene statistics if available
        if scene_stats:
            try:
                if hasattr(scene_stats, 'to_dict'):
                    export_data['scene_statistics'] = convert_numpy(scene_stats.to_dict())
                else:
                    # Extract key attributes manually
                    stats_dict = {}
                    for attr in ['svf_mean', 'svf_std', 'height_mean', 'height_std', 
                               'vegetation_ratio', 'built_ratio', 'openness_score']:
                        if hasattr(scene_stats, attr):
                            stats_dict[attr] = convert_numpy(getattr(scene_stats, attr))
                    export_data['scene_statistics'] = stats_dict
            except Exception as e:
                if self.debug:
                    print(f"  Failed to extract scene statistics: {e}")
                export_data['scene_statistics'] = {}
        
        self.exported_data.append(export_data)
        
        if self.debug:
            print(f" Captured data for scene {scene_id}, category {category}")
    
    def export_verification_files(self, filename_prefix: str = "qa_verification") -> Dict[str, str]:
        """
        Export captured data to verification files
        
        Args:
            filename_prefix: Prefix for output filenames
            
        Returns:
            Dictionary with paths to created files
        """
        if not self.exported_data:
            print("  No data to export")
            return {}
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"{filename_prefix}_{timestamp}"
        
        created_files = {}
        
        # 1. Export detailed YAML file for human review
        yaml_path = os.path.join(self.output_dir, f"{base_filename}_detailed.yaml")
        try:
            # Collect data source information from all QA pairs
            data_source_summary = self._summarize_data_sources()
            
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump({
                    'export_info': {
                        'timestamp': datetime.now().isoformat(),
                        'total_qa_pairs': len(self.exported_data),
                        'purpose': 'Verification of QA generation accuracy against scene metrics',
                        'data_source_summary': data_source_summary
                    },
                    'qa_data': self.exported_data
                }, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            created_files['detailed_yaml'] = yaml_path
            print(f" Detailed YAML exported: {yaml_path}")
            
        except Exception as e:
            print(f" Failed to export detailed YAML: {e}")
        
        # 2. Export human-readable verification report
        report_path = os.path.join(self.output_dir, f"{base_filename}_verification_report.md")
        try:
            self._create_verification_report(report_path)
            created_files['verification_report'] = report_path
            print(f" Verification report exported: {report_path}")
            
        except Exception as e:
            print(f" Failed to export verification report: {e}")
        
        # 3. Export compact JSON for programmatic access
        json_path = os.path.join(self.output_dir, f"{base_filename}_compact.json")
        try:
            compact_data = []
            for item in self.exported_data:
                compact_item = {
                    'scene_id': item['scene_id'],
                    'category': item['category'],
                    'question': item['question'],
                    'answer': item['answer'],
                    'key_metrics': self._extract_key_metrics(item['metrics']),
                    'verification_points': self._generate_verification_points(item)
                }
                compact_data.append(compact_item)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(compact_data, f, ensure_ascii=False, indent=2)
            
            created_files['compact_json'] = json_path
            print(f" Compact JSON exported: {json_path}")
            
        except Exception as e:
            print(f" Failed to export compact JSON: {e}")
        
        # 4. Export metrics comparison table (CSV)
        csv_path = os.path.join(self.output_dir, f"{base_filename}_metrics_table.csv")
        try:
            self._create_metrics_csv(csv_path)
            created_files['metrics_csv'] = csv_path
            print(f" Metrics CSV exported: {csv_path}")
            
        except Exception as e:
            print(f" Failed to export metrics CSV: {e}")
        
        print(f"\n Export completed! {len(created_files)} files created in {self.output_dir}")
        return created_files
    
    def _create_verification_report(self, output_path: str) -> None:
        """Create human-readable verification report"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# QA Generation Verification Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total QA pairs: {len(self.exported_data)}\n\n")
            f.write("## Purpose\n")
            f.write("This report shows the metrics and prompts used to generate each QA pair,\n")
            f.write("allowing manual verification that answers match the underlying scene data.\n\n")
            
            for i, item in enumerate(self.exported_data, 1):
                f.write(f"## QA Pair {i}: {item['scene_id']} - {item['category']}\n\n")
                
                f.write("### Question\n")
                f.write(f"```\n{item['question']}\n```\n\n")
                
                f.write("### Generated Answer\n")
                f.write(f"```\n{item['answer']}\n```\n\n")
                
                f.write("### Key Metrics Used\n")
                key_metrics = self._extract_key_metrics(item['metrics'])
                for metric, value in key_metrics.items():
                    f.write(f"- **{metric}**: {value}\n")
                f.write("\n")
                
                # Add data source information if available
                if item.get('data_sources'):
                    f.write("### Data Sources and Calculation Basis\n")
                    data_sources = item['data_sources']
                    for source_type, info in data_sources.items():
                        if isinstance(info, dict):
                            f.write(f"**{source_type.replace('_', ' ').title()}**:\n")
                            for key, value in info.items():
                                f.write(f"  - {key}: {value}\n")
                        else:
                            f.write(f"- **{source_type.replace('_', ' ').title()}**: {info}\n")
                    f.write("\n")
                
                f.write("### Verification Points\n")
                verification_points = self._generate_verification_points(item)
                for point in verification_points:
                    f.write(f"- [ ] {point}\n")
                f.write("\n")
                
                if item.get('prompt_data'):
                    f.write("### GPT Prompt Information\n")
                    prompt_data = item['prompt_data']
                    if 'system_prompt' in prompt_data:
                        f.write(f"**System Prompt**: {prompt_data['system_prompt'][:200]}...\n")
                    if 'user_prompt' in prompt_data:
                        f.write(f"**User Prompt**: {prompt_data['user_prompt'][:200]}...\n")
                    f.write("\n")
                
                f.write("---\n\n")
    
    def _extract_key_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics for verification"""
        key_metrics = {}
        
        # Sky visibility metrics
        if 'sky_visibility' in metrics:
            sky_vis = metrics['sky_visibility']
            key_metrics.update({
                'SVF Mean': f"{sky_vis.get('mean', 0):.3f}",
                'SVF Std': f"{sky_vis.get('std', 0):.3f}",
                'SVF Range': f"{sky_vis.get('min', 0):.3f} - {sky_vis.get('max', 1):.3f}"
            })
        
        # Terrain metrics
        if 'terrain' in metrics:
            terrain = metrics['terrain']
            key_metrics.update({
                'Height Mean': f"{terrain.get('mean_height', 0):.1f}m",
                'Height Std': f"{terrain.get('height_std', 0):.1f}m",
                'Elevation Range': f"{terrain.get('elevation_range', 0):.1f}m"
            })
        
        # Land use metrics
        if 'land_use' in metrics:
            land_use = metrics['land_use']
            # Show top 3 land use types
            sorted_land_use = sorted(land_use.items(), key=lambda x: x[1], reverse=True)[:3]
            for i, (land_type, percentage) in enumerate(sorted_land_use):
                key_metrics[f'Land Use {i+1}'] = f"{land_type}: {percentage:.1f}%"
        
        # Openness metrics
        if 'openness' in metrics:
            openness = metrics['openness']
            key_metrics['Openness Score'] = f"{openness.get('score', 0):.3f}"
        
        return key_metrics
    
    def _generate_verification_points(self, item: Dict[str, Any]) -> List[str]:
        """Generate specific verification points for each QA pair"""
        points = []
        category = item['category']
        metrics = item['metrics']
        answer = item['answer']
        
        # Common verification points
        points.append("Answer mentions specific numerical values that match the computed metrics")
        points.append("Answer correctly interprets the SVF values in context")
        
        # Category-specific verification points
        if category == 'urban_development_application':
            points.append("Answer discusses built-up areas percentage and relates to computed land use metrics")
            points.append("Answer considers building density based on SVF and height data")
            
        elif category == 'renewable_energy_installation':
            points.append("Answer mentions solar potential based on SVF values")
            points.append("Answer considers terrain suitability based on height data")
            
        elif category == 'landscape_analysis':
            points.append("Answer describes landscape character based on land cover distribution")
            points.append("Answer mentions vegetation coverage matching computed ratios")
            
        elif category == 'water_accumulation':
            points.append("Answer discusses terrain topology and flow patterns")
            points.append("Answer considers surface imperviousness based on land use")
        
        # Numerical consistency checks
        if 'sky_visibility' in metrics:
            svf_mean = metrics['sky_visibility'].get('mean', 0)
            if svf_mean < 0.3:
                points.append("Answer correctly identifies low sky visibility (dense urban/forest area)")
            elif svf_mean > 0.7:
                points.append("Answer correctly identifies high sky visibility (open area)")
        
        if 'land_use' in metrics:
            land_use = metrics['land_use']
            if land_use.get('forest', 0) > 50:
                points.append("Answer acknowledges forest-dominated landscape")
            if land_use.get('residential', 0) + land_use.get('commercial', 0) > 40:
                points.append("Answer acknowledges urban-dominated landscape")
        
        return points
    
    def _create_metrics_csv(self, output_path: str) -> None:
        """Create CSV table of metrics for easy comparison"""
        import csv
        
        # Define CSV headers
        headers = ['scene_id', 'category', 'svf_mean', 'svf_std', 'height_mean', 'height_std',
                  'forest_pct', 'residential_pct', 'commercial_pct', 'water_pct', 
                  'openness_score', 'answer_length', 'question_length']
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for item in self.exported_data:
                metrics = item['metrics']
                
                # Extract values with defaults
                sky_vis = metrics.get('sky_visibility', {})
                terrain = metrics.get('terrain', {})
                land_use = metrics.get('land_use', {})
                openness = metrics.get('openness', {})
                
                row = [
                    item['scene_id'],
                    item['category'],
                    sky_vis.get('mean', 0),
                    sky_vis.get('std', 0),
                    terrain.get('mean_height', 0),
                    terrain.get('height_std', 0),
                    land_use.get('forest', 0),
                    land_use.get('residential', 0),
                    land_use.get('commercial', 0),
                    land_use.get('water', 0),
                    openness.get('score', 0),
                    len(item['answer']),
                    len(item['question'])
                ]
                writer.writerow(row)
    
    def _summarize_data_sources(self) -> Dict[str, Any]:
        """Summarize data sources used across all QA pairs for the export info"""
        summary = {
            'dataset_info': {
                'name': 'GeoNRW Digital Elevation and Land Cover Dataset',
                'provider': 'State of North Rhine-Westphalia, Germany',
                'spatial_resolution': '1m x 1m',
                'coordinate_system': 'EPSG:25832 (ETRS89 / UTM zone 32N)'
            },
            'processing_pipeline': {
                'system': 'SVF-VQA Generation System',
                'version': 'v1.0',
                'bias_prevention': 'Automated bias-free shuffling applied',
                'quality_assurance': 'Metrics verification enabled'
            },
            'data_types': [],
            'scenes_processed': set(),
            'calculation_methods': {
                'elevation_statistics': 'numpy.nanmin/nanmax/nanmean/nanstd operations on DEM arrays',
                'sky_view_factor': 'Hemispherical sky visibility analysis',
                'land_use_classification': 'Segmentation mask analysis with area calculations',
                'terrain_analysis': 'Slope and roughness calculations from elevation gradients'
            }
        }
        
        # Collect unique data types and scenes from all QA pairs
        for item in self.exported_data:
            if 'scene_id' in item:
                summary['scenes_processed'].add(item['scene_id'])
            
            if 'data_sources' in item and item['data_sources']:
                data_sources = item['data_sources']
                for source_type in data_sources.keys():
                    if source_type not in summary['data_types']:
                        summary['data_types'].append(source_type)
        
        # Convert set to list for YAML serialization
        summary['scenes_processed'] = list(summary['scenes_processed'])
        
        return summary

    def clear_captured_data(self) -> None:
        """Clear captured data to start fresh"""
        self.exported_data = []
        if self.debug:
            print("  Captured data cleared")


# Utility functions for integration with existing code
def create_verification_exporter(output_dir: str = "verification_outputs", debug: bool = False) -> MetricsPromptExporter:
    """Create a metrics and prompt exporter instance"""
    return MetricsPromptExporter(output_dir=output_dir, debug=debug)


def export_qa_verification_data(qa_pairs: List[Dict[str, Any]], output_dir: str = "verification_outputs") -> Dict[str, str]:
    """
    Export QA pairs with their metrics for verification
    
    Args:
        qa_pairs: List of QA pairs with detailed metadata
        output_dir: Output directory for verification files
        
    Returns:
        Dictionary with paths to created verification files
    """
    exporter = create_verification_exporter(output_dir=output_dir, debug=True)
    
    for qa in qa_pairs:
        # Extract detailed metadata if available
        detailed_meta = qa.get('__detailed_metadata__', {})
        
        exporter.capture_qa_generation_data(
            scene_id=qa.get('scene_id', 'unknown'),
            category=qa.get('category', 'unknown'),
            metrics=detailed_meta.get('metrics', {}),
            question=qa.get('text', ''),
            answer=qa.get('answer', ''),
            prompt_data=detailed_meta.get('prompt_data', {}),
            scene_stats=None  # Would need to be passed separately
        )
    
    return exporter.export_verification_files()