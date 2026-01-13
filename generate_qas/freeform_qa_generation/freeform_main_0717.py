"""
Free-form Analysis QA Generation System for LandscapeVQA

This script generates free-form analysis questions and answers using real GeoNRW data.
Based on svf_questions_gpt4.py implementation pattern.

Features:
- Real GeoNRW data loading (SVF, DSM, RGB, SEG)
- GPT-4 powered free-form analysis answer generation
- Statistical fallback when API is unavailable
- Train/test split support
- JSONL output format (text, answer, question_id, image)
"""

import os
import argparse
import json
import random
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import traceback
import numpy as np
import yaml
import shutil
import base64
import io

# Import statistics modules
import sys
import os

# Add stats directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
stats_path = os.path.join(parent_dir, 'stats')

if stats_path not in sys.path:
    sys.path.insert(0, stats_path)

try:
    from scene_statistics import SceneStatistics
    from data_loader import load_scenes_from_svf_directory, create_sample_scene
    print(" Stats modules imported successfully")
except ImportError as e:
    print(f" Failed to import stats modules: {e}")
    print(f"   Stats path: {stats_path}")
    print(f"   Current sys.path: {sys.path[:3]}...")  # Show first 3 paths
    raise

from freeform_analysis_categories import FreeformAnalysisCategories
from metrics_prompt_exporter import MetricsPromptExporter, export_qa_verification_data
from freeform_qa_generation.verification.pipeline import verify_answer
from freeform_qa_generation.verification.conclusion_rules import extract_sections

# Import enhanced GPT-4 generator with advanced statistics
HAS_ENHANCED_GPT4_GENERATOR = False
GPT4QAGenerator = None

# Try multiple import strategies
try:
    # Strategy 1: Direct import
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from gpt4_generator import GPT4QAGenerator
    HAS_ENHANCED_GPT4_GENERATOR = True
    print(" Enhanced GPT-4 generator import successful (direct)")
except ImportError as e1:
    try:
        # Strategy 2: Relative import
        from .gpt4_generator import GPT4QAGenerator
        HAS_ENHANCED_GPT4_GENERATOR = True
        print(" Enhanced GPT-4 generator import successful (relative)")
    except (ImportError, SystemError) as e2:
        try:

            raise ImportError("Absolute import disabled in public release")
        except ImportError as e3:
            print(f"  Enhanced GPT-4 generator not available: {e1}, {e2}, {e3}")

# Import legacy GPT-4 answer generator as fallback
HAS_GPT4_GENERATOR = False
GPT4FreeformAnswerGenerator = None

try:
    from gpt4_freeform_answer_generator import GPT4FreeformAnswerGenerator
    HAS_GPT4_GENERATOR = True
except ImportError:
    try:
        # Try with current directory in path
        from .gpt4_freeform_answer_generator import GPT4FreeformAnswerGenerator
        HAS_GPT4_GENERATOR = True
    except ImportError:
        print("  GPT-4 freeform answer generator not available")


class FreeformQAGenerator:
    """Enhanced free-form analysis QA generation system with advanced statistics"""
    
    def __init__(self, debug: bool = False, use_gpt4: bool = True, 
                 api_key: str = None, model: str = "gpt-4o-mini", output_dir: str = 'output'):
        self.debug = debug
        self.use_gpt4 = use_gpt4 and (HAS_ENHANCED_GPT4_GENERATOR or HAS_GPT4_GENERATOR)
        self.model = model
        
        # Question ID counter for consistent numbering
        self.question_id_counter = 0
        
        # Initialize enhanced GPT-4 generator if available
        self.enhanced_gpt4_generator = None
        self.gpt4_generator = None
        self.metrics_dir = f'{output_dir}/metrics'
        
        # Initialize verification exporter
        self.verification_exporter = MetricsPromptExporter(
            output_dir=f'{output_dir}/verification',
            debug=debug
        )
        # One-time path print flags
        self._printed_failed_verification_path = False
        self._printed_regeneration_log_path = False
        
        # Always try to initialize enhanced GPT-4 generator for enhanced fallback
        if HAS_ENHANCED_GPT4_GENERATOR:
            try:
                self.enhanced_gpt4_generator = GPT4QAGenerator(
                    api_key=api_key,
                    model=model,
                    debug=debug
                )
                if debug:
                    print(f" Enhanced GPT-4 generator initialized with model: {model}")
            except Exception as e:
                if debug:
                    print(f"  Failed to initialize enhanced GPT-4 generator: {e}")
                self.enhanced_gpt4_generator = None
        
        # Additional GPT-4 setup only if use_gpt4 is True
        if self.use_gpt4:
            # Fallback to legacy generator if enhanced is not available
            if not self.enhanced_gpt4_generator and HAS_GPT4_GENERATOR:
                try:
                    self.gpt4_generator = GPT4FreeformAnswerGenerator(
                        api_key=api_key,
                        model=model,
                        debug=debug,
                        output_dir=self.metrics_dir
                    )
                    if debug:
                        print(f" Legacy GPT-4 generator initialized with model: {model}")
                except Exception as e:
                    if debug:
                        print(f"  Failed to initialize legacy GPT-4 generator: {e}")
                    self.use_gpt4 = False
        
        # Free-form categories available
        self.categories = [
            'urban_development_application',
            'renewable_energy_installation', 
            'landscape_analysis',
            'water_accumulation'
        ]
        
        if debug:
            print(f"  Available free-form categories: {self.categories}")
    
    def _convert_image_path(self, scene_data: Dict[str, Any]) -> str:
        """Convert scene_data to proper image path format compatible with existing outputs"""
        # Extract area and filename from scene data
        area = scene_data.get('area', 'unknown')
        scene_id = scene_data.get('id', '')
        
        # Generate filename in the expected format
        if scene_id and "_dem" in scene_id:
            # Remove "_dem" suffix and add "_rgb.jp2"
            base_name = scene_id.replace("_dem", "") + "_rgb.jp2"
            
            # Remove area name from the beginning if it exists to avoid duplication
            # e.g., "guetersloh_455_5754_rgb.jp2" -> "455_5754_rgb.jp2"
            if base_name.startswith(f"{area}_"):
                base_name = base_name[len(area)+1:]  # +1 for the underscore
        else:
            # Fallback case
            base_name = f"{scene_id}_rgb.jp2"
            # Same area prefix removal for fallback
            if base_name.startswith(f"{area}_"):
                base_name = base_name[len(area)+1:]
        
        # Return in the format: "area/filename"
        return f"{area}/{base_name}"
    
    def _construct_data_sources_info(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """Construct comprehensive data source information for verification"""
        area = scene_data.get('area', 'unknown')
        scene_id = scene_data.get('id', 'unknown')
        
        # Determine base filename for file path construction
        base_filename = scene_id if scene_id != 'unknown' else ''
        
        # Construct expected file paths based on naming convention
        data_sources = {
            'scene_identification': {
                'scene_id': scene_id,
                'geographic_area': area,
                'coordinate_tile': scene_id.replace('_dem', '') if '_dem' in scene_id else scene_id
            },
            'data_files': {
                'svf_file': scene_data.get('file_path', f"{area}/{base_filename}.tif"),
                'dsm_file': f"{area}/{base_filename}.tif",
                'segmentation_file': f"{area}/{base_filename.replace('_dem', '_seg')}.tif" if '_dem' in base_filename else f"{area}/{base_filename}_seg.tif",
                'rgb_file': f"{area}/{base_filename.replace('_dem', '_rgb')}.jp2" if '_dem' in base_filename else f"{area}/{base_filename}_rgb.jp2"
            },
            'data_availability': {
                'svf_data': scene_data.get('svf') is not None,
                'height_data': scene_data.get('height_map') is not None,
                'segmentation_data': scene_data.get('seg') is not None,
                'rgb_data': scene_data.get('rgb') is not None
            },
            'processing_metadata': {
                'calculation_method': 'freeform_analysis_categories',
                'elevation_range_calculation': 'numpy.nanmin/nanmax operations on DEM array',
                'svf_calculation': 'UMEP-based sky view factor analysis',
                'land_use_calculation': 'segmentation mask pixel counting and percentage analysis',
                'terrain_analysis': 'gradient-based slope and roughness calculations'
            },
            'dataset_info': {
                'dataset_name': 'GeoNRW',
                'spatial_resolution': '1m x 1m',
                'coordinate_system': 'EPSG:25832 (ETRS89 / UTM zone 32N)',
                'data_provider': 'State of North Rhine-Westphalia, Germany'
            },
            'verification_notes': {
                'elevation_basis': f'Elevation values (min/max/range) calculated directly from DEM file: {area}/{base_filename}.tif',
                'svf_basis': f'Sky view factor statistics from SVF raster analysis',
                'land_use_basis': f'Land cover percentages from segmentation classification: {area}/{base_filename.replace("_dem", "_seg")}.tif'
            }
        }
        
        return data_sources
    
    def _encode_image_to_data_url(self, image_obj: Any) -> Optional[str]:
        """Encode numpy/PIL image to PNG data URL (fallback when legacy encoder unavailable)."""
        try:
            # Lazy imports to avoid hard dependencies
            from PIL import Image
            pil_img = None
            if image_obj is None:
                return None
            # numpy array -> PIL
            try:
                import numpy as _np
                if isinstance(image_obj, _np.ndarray):
                    # Handle grayscale and RGB
                    if image_obj.ndim == 2:
                        pil_img = Image.fromarray(image_obj.astype('uint8'))
                    elif image_obj.ndim == 3:
                        # If channel last and in float [0,1], scale up
                        arr = image_obj
                        if arr.dtype != _np.uint8:
                            arr = _np.clip(arr, 0, 1) * 255.0
                            arr = arr.astype('uint8')
                        pil_img = Image.fromarray(arr)
            except Exception:
                pil_img = None
            # PIL Image passed directly
            if pil_img is None and hasattr(image_obj, 'save'):
                pil_img = image_obj
            if pil_img is None:
                return None
            buf = io.BytesIO()
            pil_img.save(buf, format='PNG')
            encoded = base64.b64encode(buf.getvalue()).decode('ascii')
            return f"data:image/png;base64,{encoded}"
        except Exception:
            # Fallback to OpenCV if available
            try:
                import cv2
                import numpy as _np
                if isinstance(image_obj, _np.ndarray):
                    arr = image_obj
                    if arr.dtype != _np.uint8:
                        arr = _np.clip(arr, 0, 1) * 255.0
                        arr = arr.astype('uint8')
                    # Ensure 3-channel for color
                    if arr.ndim == 2:
                        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                    success, buf = cv2.imencode('.png', arr)
                    if success:
                        encoded = base64.b64encode(buf.tobytes()).decode('ascii')
                        return f"data:image/png;base64,{encoded}"
            except Exception:
                return None
            return None

    def generate_qa_from_scene(self, scene_data: Dict[str, Any], dataset_type: str = "train") -> List[Dict[str, Any]]:
        """Generate free-form QA pairs from a single scene with enhanced statistics"""
        qa_pairs = []
        
        try:
            # Extract enhanced scene statistics if available
            enhanced_statistics = None
            if self.enhanced_gpt4_generator:
                try:
                    enhanced_statistics = self.enhanced_gpt4_generator.extract_scene_statistics(scene_data)
                    if self.debug:
                        print(f" Extracted enhanced statistics for scene {scene_data.get('id', 'unknown')}")
                except Exception as e:
                    if self.debug:
                        print(f"  Enhanced statistics extraction failed: {e}")
                    enhanced_statistics = None
            
            # Initialize FreeformAnalysisCategories with enhanced statistics if available
            freeform_analyzer = FreeformAnalysisCategories(
                svf_map=scene_data['svf'],
                height_map=scene_data['dsm'],
                segmentation_map=scene_data['seg'],
                rgb_image=scene_data['rgb'],
                file_path=scene_data.get('file_path'),
                debug=self.debug,
                use_gpt4=self.use_gpt4,
                api_key=os.getenv('OPENAI_API_KEY'),
                model=self.model
            )
            
            # Store enhanced statistics in analyzer if available
            if enhanced_statistics and hasattr(freeform_analyzer, 'set_enhanced_statistics'):
                freeform_analyzer.set_enhanced_statistics(enhanced_statistics)
            
            # Generate QA for each category
            for category in self.categories:
                try:
                    # Get the category method
                    category_method = getattr(freeform_analyzer, category)
                    qa_result = category_method()
                    
                    # Skip if no valid data for this category
                    if qa_result is None:
                        if self.debug:
                            print(f"  Skipping category {category} - no valid data")
                        continue
                    
                    # Extract data from the unified return format
                    answer = qa_result.get('answer', '')
                    qa_info = {
                        'question': qa_result.get('question', ''),
                        'analysis_type': qa_result.get('analysis_type', category),
                        'metrics': qa_result.get('metrics', {})
                    }
                    context = qa_result.get('context', [])
                    generation_method = qa_result.get('generation_method', 'statistical')
                    
                    # Enhanced answer generation using advanced GPT-4 generator or enhanced fallback
                    enhanced_answer = answer
                    
                    if self.enhanced_gpt4_generator and enhanced_statistics:
                        try:
                            # Try to generate enhanced answer using structured prompting
                            enhanced_answer = self.enhanced_gpt4_generator._generate_gpt_answer_for_question(
                                qa_info['question'], enhanced_statistics
                            )
                            if enhanced_answer:
                                generation_method = 'enhanced_gpt4'
                                if self.debug:
                                    print(f" Enhanced answer generated for {category}")
                            else:
                                enhanced_answer = answer  # Fallback to original
                        except Exception as e:
                            if self.debug:
                                print(f"  Enhanced answer generation failed for {category}: {e}")
                            enhanced_answer = answer  # Fallback to original
                    elif self.enhanced_gpt4_generator:
                        # Use enhanced fallback system with available statistics
                        try:
                            # Create a basic statistics object from available data
                            scene_stats = self._create_scene_statistics_from_data(scene_data, qa_info)
                            if scene_stats is None:
                                # Skip enhanced fallback when steatistics are invalid (e.g., total_land == 0)
                                enhanced_answer = answer
                                if self.debug:
                                    print(f"  Skipping enhanced fallback for {category}: invalid land cover totals")
                            else:
                                enhanced_answer = self.enhanced_gpt4_generator._generate_fallback_answer_for_question(
                                    qa_info['question'], scene_stats
                                )
                                generation_method = 'enhanced_fallback'
                                if self.debug:
                                    print(f" Enhanced fallback answer generated for {category}")
                        except Exception as e:
                            if self.debug:
                                print(f"  Enhanced fallback generation failed for {category}: {e}")
                            enhanced_answer = answer  # Fallback to original
                    
                    # Construct data source information for verification traceability
                    data_sources = self._construct_data_sources_info(scene_data)
                    
                    # Optional 2x2 panel for both generation (variation) and verification
                    panel_image_data_url = None
                    panel_img = None
                    try:
                        if hasattr(freeform_analyzer, '_build_quadrant_visual'):
                            panel_img = freeform_analyzer._build_quadrant_visual()
                            if panel_img is not None:
                                if hasattr(self.gpt4_generator, '_encode_image_to_data_url'):
                                    panel_image_data_url = self.gpt4_generator._encode_image_to_data_url(panel_img)
                                if not panel_image_data_url:
                                    # Local fallback encoder to always provide panel image
                                    panel_image_data_url = self._encode_image_to_data_url(panel_img)
                    except Exception:
                        panel_image_data_url = None

                    # Canonicalize the initial answer to avoid duplicated OBSERVATION
                    try:
                        from freeform_qa_generation.verification.conclusion_rules import canonicalize_answer
                        enhanced_answer = canonicalize_answer(enhanced_answer)
                    except Exception:
                        pass

                    # Append compact DSM/SVF/SEG grid evidence for water accumulation
                    try:
                        if category == 'water_accumulation':
                            from freeform_qa_generation.evidence.water_grid_stats import (
                                compute_water_grid_stats,
                                format_water_grid_evidence,
                            )
                            # Compute grid stats from available arrays in scene_data
                            grid_summary = compute_water_grid_stats(scene_data)
                            evidence_text = format_water_grid_evidence(grid_summary)
                            if evidence_text:
                                # Insert into <OBSERVATION> if not already present
                                from freeform_qa_generation.verification.conclusion_rules import extract_sections
                                sections_tmp = extract_sections(enhanced_answer)
                                obs_body = (sections_tmp.get('OBSERVATION') or '').strip()
                                con_body = (sections_tmp.get('CONCLUSION') or '').strip()
                                if obs_body and ('Grid evidence' not in obs_body):
                                    joined_obs = (obs_body + ' ' + evidence_text).strip()
                                    enhanced_answer = f"<OBSERVATION>{joined_obs}</OBSERVATION><CONCLUSION>{con_body}</CONCLUSION>"
                                    # Re-canonicalize to ensure clean structure
                                    try:
                                        enhanced_answer = canonicalize_answer(enhanced_answer)
                                    except Exception:
                                        pass
                    except Exception:
                        # Non-blocking; continue without evidence if any issue occurs
                        pass

                    # Prepare statistics context YAML (same as generation uses)
                    statistics_context = None
                    try:
                        from dataclasses import asdict, is_dataclass
                        import yaml as _yaml
                        stats_obj = enhanced_statistics
                        if stats_obj is None:
                            stats_obj = self._create_scene_statistics_from_data(scene_data, qa_info)
                        if stats_obj is not None:
                            if is_dataclass(stats_obj):
                                stats_dict = asdict(stats_obj)
                            else:
                                stats_dict = dict(getattr(stats_obj, '__dict__', {}))
                            statistics_context = _yaml.safe_dump(stats_dict, sort_keys=False, allow_unicode=True)
                    except Exception:
                        statistics_context = None

                    # Prepare compact 3x3 grid analysis metadata for traceability (if available)
                    grid_metadata = None
                    try:
                        stats_obj_for_grid = enhanced_statistics
                        if stats_obj_for_grid and hasattr(stats_obj_for_grid, 'grid_analysis') and stats_obj_for_grid.grid_analysis:
                            grid = stats_obj_for_grid.grid_analysis

                            def _pos_name(pos):
                                row_names = ["Top", "Middle", "Bottom"]
                                col_names = ["Left", "Center", "Right"]
                                try:
                                    return f"{row_names[pos[0]]}-{col_names[pos[1]]}"
                                except Exception:
                                    return str(pos)

                            # Build minimal, serialization-safe dictionary
                            try:
                                grid_metadata = {
                                    'grid_svf_values': getattr(grid, 'grid_svf_values', None),
                                    'grid_height_values': getattr(grid, 'grid_height_values', None),
                                }
                                # Include hydrology summary with explicit selection reason if available
                                try:
                                    hyd_scores = getattr(grid, 'grid_water_accumulation_score', None)
                                    hyd_best = getattr(grid, 'best_water_accumulation_position', None)
                                    if hyd_scores is not None or hyd_best is not None:
                                        grid_metadata['hydrology'] = {
                                            'grid_water_accumulation_score': hyd_scores,
                                            'best_water_accumulation_position': {
                                                'coordinates': hyd_best,
                                                'name': _pos_name(hyd_best) if hyd_best is not None else None,
                                                'reason': 'Selected by hydrology score: 0.6*(-zscore(low-percentile elevation)) + 0.2*(-zscore(mean_slope)) + 0.2*(zscore(water_ratio)).'
                                            } if hyd_best is not None else None
                                        }
                                except Exception:
                                    pass
                                # Best/worst and extrema positions if present
                                # Add brief one-line reasons for how each metric is determined
                                reason_map = {
                                    'best_svf_position': 'Selected as the grid cell with the highest SVF (openness).',
                                    'worst_svf_position': 'Selected as the grid cell with the lowest SVF (openness).',
                                    'best_solar_position': 'Selected as the grid cell with the highest solar potential (from SVF and land cover).',
                                    'worst_solar_position': 'Selected as the grid cell with the lowest solar potential (from SVF and land cover).',
                                    'best_development_position': 'Highest development score from moderate-to-high SVF (≥0.3 with ≥0.7 capped at max), developable land cover (agricultural/grassland/others), and relatively flat terrain (height near 50 m).',
                                    'worst_development_position': 'Selected as the grid cell with the lowest development potential (from SVF, land cover, and terrain).',
                                    'best_scenic_position': 'Selected as the grid cell with the highest scenic score (from openness and visual diversity).',
                                    'worst_scenic_position': 'Selected as the grid cell with the lowest scenic score (from openness and visual diversity).',
                                    'highest_height_position': 'Selected as the grid cell with the highest mean elevation.',
                                    'lowest_height_position': 'Selected as the grid cell with the lowest mean elevation.'
                                }
                                for key_attr, value_matrix_attr, label in [
                                    ('best_svf_position', 'grid_svf_values', 'best_svf_position'),
                                    ('worst_svf_position', 'grid_svf_values', 'worst_svf_position'),
                                    ('best_solar_position', 'grid_solar_potential', 'best_solar_position'),
                                    ('worst_solar_position', 'grid_solar_potential', 'worst_solar_position'),
                                    ('best_development_position', 'grid_development_potential', 'best_development_position'),
                                    ('worst_development_position', 'grid_development_potential', 'worst_development_position'),
                                    ('best_scenic_position', 'grid_scenic_score', 'best_scenic_position'),
                                    ('worst_scenic_position', 'grid_scenic_score', 'worst_scenic_position'),
                                    ('highest_height_position', 'grid_height_values', 'highest_height_position'),
                                    ('lowest_height_position', 'grid_height_values', 'lowest_height_position'),
                                ]:
                                    try:
                                        coord = getattr(grid, key_attr)
                                        matrix = getattr(grid, value_matrix_attr, None)
                                        value = None
                                        if coord is not None and matrix is not None:
                                            value = matrix[coord[0]][coord[1]]
                                        grid_metadata[label] = {
                                            'coordinates': coord,
                                            'name': _pos_name(coord) if coord is not None else None,
                                            'value': value,
                                        }
                                        # Attach concise reason explaining selection logic
                                        reason_text = reason_map.get(label)
                                        if reason_text:
                                            grid_metadata[label]['reason'] = reason_text
                                    except Exception:
                                        # Keep robustness; skip if any field missing
                                        continue
                            except Exception:
                                grid_metadata = None
                    except Exception:
                        grid_metadata = None

                    # Candidate generation: base + one extra variant from legacy generator (if available)
                    candidates = []
                    if isinstance(enhanced_answer, str) and enhanced_answer.strip():
                        candidates.append(("base", enhanced_answer))
                    # Add a second candidate by reusing legacy generator with slight temperature variation
                    try:
                        gen_stats = enhanced_statistics
                        if gen_stats is None:
                            gen_stats = self._create_scene_statistics_from_data(scene_data, qa_info)
                        if self.gpt4_generator and gen_stats is not None:
                            # Preserve prior temp if set
                            prior_temp = getattr(self.gpt4_generator, "default_temperature", None)
                            try:
                                # Candidate v0: deterministic
                                self.gpt4_generator.default_temperature = 0.0
                                v0 = self.gpt4_generator.generate_answer(
                                    question=qa_info['question'],
                                    statistics=gen_stats,
                                    scene_id=scene_data.get('id', 'unknown'),
                                    analysis_type=qa_info.get('analysis_type', category),
                                    image_data=panel_img,
                                    question_context={'variation': 'v0'}
                                )
                                if isinstance(v0, str) and v0.strip():
                                    try:
                                        from freeform_qa_generation.verification.conclusion_rules import canonicalize_answer
                                        v0 = canonicalize_answer(v0)
                                    except Exception:
                                        pass
                                    candidates.append(("v0", v0))
                                # Candidate v1: slight diversity
                                self.gpt4_generator.default_temperature = 0.2
                                v1 = self.gpt4_generator.generate_answer(
                                    question=qa_info['question'],
                                    statistics=gen_stats,
                                    scene_id=scene_data.get('id', 'unknown'),
                                    analysis_type=qa_info.get('analysis_type', category),
                                    image_data=panel_img,
                                    question_context={'variation': 'v1'}
                                )
                                if isinstance(v1, str) and v1.strip():
                                    try:
                                        from freeform_qa_generation.verification.conclusion_rules import canonicalize_answer
                                        v1 = canonicalize_answer(v1)
                                    except Exception:
                                        pass
                                    candidates.append(("v1", v1))
                            finally:
                                if prior_temp is None:
                                    try:
                                        delattr(self.gpt4_generator, "default_temperature")
                                    except Exception:
                                        pass
                                else:
                                    self.gpt4_generator.default_temperature = prior_temp
                    except Exception:
                        pass

                    # Verify all candidates and pick the best one
                    best_label = None
                    best_answer = None
                    best_verdict = None
                    best_score = None

                    for label, cand in candidates:
                        verdict = verify_answer(cand, panel_image_data_url, enable_llm=True, statistics_context=statistics_context)
                        # Score: prioritize pass, then fewer issues
                        passed = 1 if verdict.get('passed', False) else 0
                        rules = verdict.get('rules') or {}
                        violations = rules.get('violations') or {}
                        # count total violation items length (lists only)
                        v_count = 0
                        for _, v in violations.items():
                            if isinstance(v, list):
                                v_count += len(v)
                        llm = verdict.get('llm') or {}
                        llm_issues = llm.get('detected_issues') or []
                        li_count = len(llm_issues)

                        # Grid consistency penalty (important for single-sector constraint)
                        grid_info = rules.get('grid_feedback') or violations.get('grid_consistency') or {}
                        grid_status = str((grid_info or {}).get('status', 'unknown')).lower()
                        grid_penalty = 0
                        if grid_status in ('multiple', 'multiple_without_stats'):
                            grid_penalty = 150
                        elif grid_status in ('mismatch', 'missing'):
                            grid_penalty = 80
                        # unknown/match -> no penalty

                        # Compass direction penalty (prefer grid naming over compass words)
                        compass_v = violations.get('compass_terms') or {}
                        compass_found = compass_v.get('found_compass') or []
                        compass_penalty = 60 if compass_found else 0

                        # Higher score is better: pass=1 adds big weight; fewer issues preferred; penalize grid issues strongly
                        score = (passed * 1000) - (v_count * 10) - li_count - grid_penalty - compass_penalty
                        if best_score is None or score > best_score:
                            best_score = score
                            best_label = label
                            best_answer = cand
                            best_verdict = verdict

                    # Fall back to original enhanced_answer if no candidate assembled
                    if best_answer is None:
                        best_answer = enhanced_answer
                        best_verdict = verify_answer(best_answer, panel_image_data_url, enable_llm=True, statistics_context=statistics_context)

                    enhanced_answer = best_answer
                    verification_result = best_verdict

                    # Capture verification data before creating clean QA pair (using chosen answer)
                    self.verification_exporter.capture_qa_generation_data(
                        scene_id=scene_data['id'],
                        category=category,
                        metrics=qa_info.get('metrics', {}),
                        question=qa_info['question'],
                        answer=enhanced_answer,
                        prompt_data=None,  # Will be enhanced later with actual prompt data
                        scene_stats=enhanced_statistics,
                        data_sources=data_sources
                    )

                    # If verification failed, try a single guided regeneration using feedback
                    if not verification_result.get('passed', False):
                        try:
                            sections = extract_sections(enhanced_answer)
                            original_observation = sections.get('OBSERVATION', '').strip()
                            # Summarize issues for the model
                            v = verification_result.get('violations', {})
                            issues_list = []
                            for k in ['new_numbers_in_conclusion', 'unsupported_terms']:
                                vals = v.get(k, []) or []
                                if vals:
                                    issues_list.append(f"{k}: {', '.join(str(x) for x in vals)}")
                            llm_block = (verification_result.get('llm') or {})
                            # Backward-compat: detected_issues (list[str])
                            llm_issues_legacy = llm_block.get('detected_issues', []) or []
                            if llm_issues_legacy:
                                issues_list.append("llm_issues: " + "; ".join(llm_issues_legacy))
                            # New schema: issues (list[object]) with description
                            if isinstance(llm_block.get('issues'), list) and llm_block.get('issues'):
                                descs = []
                                for it in llm_block['issues']:
                                    try:
                                        d = str(it.get('description', '')).strip()
                                        if d:
                                            descs.append(d)
                                    except Exception:
                                        continue
                                if descs:
                                    issues_list.append("llm_issues: " + "; ".join(descs))
                            issues_text = "\n- " + "\n- ".join(issues_list) if issues_list else ""

                            # Decide whether to allow OBSERVATION minimal fix based on verifier feedback
                            obs_counts = llm_block.get('observation_issue_counts') or {}
                            obs_need_fix = bool(obs_counts.get('critical', 0) > 0 or obs_counts.get('minor', 0) > 0 or llm_block.get('observation_fix'))

                            if obs_need_fix:
                                suggestion = str(llm_block.get('observation_fix') or '').strip()
                                revision_constraints = (
                                    "\n\nREVISION REQUEST:"\
                                    "\n- MINIMALLY REVISE the <OBSERVATION> to correctly reflect the provided statistics (fix misreads/mislabeled context only)."\
                                    "\n- Any numbers/labels in <OBSERVATION> must come from the statistics; do not invent new ones."\
                                    "\n- Then regenerate the <CONCLUSION> with NO new numbers, supported by <OBSERVATION> and not contradicted by RGB/SVF/DSM/SEG."\
                                    "\n- Avoid terms not evidenced in the SEG panel (e.g., vegetation/water if not visible)."\
                                    "\n- If you mention a location (e.g., top-right), ensure it matches the image layout."\
                                    "\n- Output full answer with updated <OBSERVATION> and <CONCLUSION>."\
                                    f"\n\nSUGGESTED_OBSERVATION_FIX (optional, apply only if accurate): {suggestion}\n"\
                                    f"\nVERIFICATION FEEDBACK:{issues_text}\n"
                                )
                                # Provide current observation as input context to revise
                                revised_question = qa_info['question'] + revision_constraints + f"\n\nCURRENT OBSERVATION TO REVISE:\n<OBSERVATION>{original_observation}</OBSERVATION>\n"
                            else:
                                # Keep OBSERVATION as-is; regenerate only CONCLUSION
                                revision_constraints = (
                                    "\n\nREVISION REQUEST:"\
                                    "\n- Keep the OBSERVATION EXACTLY as provided below."\
                                    "\n- Regenerate ONLY the <CONCLUSION> so that it contains NO new numbers and claims are supported by the RGB/SVF/DSM/SEG panels."\
                                    "\n- Remove or avoid any terms not evidenced in the SEG panel (e.g., vegetation/water if not visible)."\
                                    "\n- If you mention a location (e.g., top-right), ensure it matches the image layout."\
                                    "\n- Output full answer with <OBSERVATION> (unchanged) and new <CONCLUSION>."\
                                    f"\n\nVERIFICATION FEEDBACK:{issues_text}\n"
                                )
                                revised_question = qa_info['question'] + revision_constraints + f"\n\n<OBSERVATION>{original_observation}</OBSERVATION>\n"

                            # Choose statistics object for regeneration
                            regen_stats = enhanced_statistics
                            if regen_stats is None:
                                regen_stats = self._create_scene_statistics_from_data(scene_data, qa_info)

                            regenerated = None
                            if self.gpt4_generator and regen_stats is not None:
                                regenerated = self.gpt4_generator.generate_answer(
                                    question=revised_question,
                                    statistics=regen_stats,
                                    scene_id=scene_data.get('id', 'unknown'),
                                    analysis_type=qa_info.get('analysis_type', category),
                                    image_data=panel_img,
                                    question_context={'revision': True, 'feedback_issues': issues_list}
                                )
                            elif self.enhanced_gpt4_generator and enhanced_statistics is not None:
                                regenerated = self.enhanced_gpt4_generator._generate_gpt_answer_for_question(
                                    revised_question, enhanced_statistics
                                )

                            if regenerated:
                                try:
                                    from freeform_qa_generation.verification.conclusion_rules import canonicalize_answer
                                except Exception:
                                    canonicalize_answer = None
                                new_sections = extract_sections(regenerated)
                                new_conclusion = new_sections.get('CONCLUSION', '').strip() or regenerated.strip()
                                # If OBSERVATION was allowed to change, take it from regenerated; otherwise keep original
                                if obs_need_fix:
                                    new_observation = new_sections.get('OBSERVATION', '').strip() or original_observation
                                else:
                                    new_observation = original_observation
                                repaired_answer = f"<OBSERVATION>{new_observation}</OBSERVATION><CONCLUSION>{new_conclusion}</CONCLUSION>"
                                if canonicalize_answer is not None:
                                    repaired_answer = canonicalize_answer(repaired_answer)
                                second_verdict = verify_answer(repaired_answer, panel_image_data_url, enable_llm=True, statistics_context=statistics_context)
                                if second_verdict.get('passed', False):
                                    enhanced_answer = repaired_answer
                                    generation_method = generation_method + '+regenerated'
                                    verification_result = second_verdict
                        except Exception:
                            # If anything goes wrong, fallback to original verification result
                            pass

                    # Create clean QA pair without heavy metadata
                    format_instruction = (
                        "Please structure your response using only <OBSERVATION> and <CONCLUSION>.\n\n"
                        "<OBSERVATION>Report concrete, image-grounded facts and statistics only (e.g., SVF values, elevation in meters, land-cover %). No interpretation.</OBSERVATION>\n"
                        "<CONCLUSION>Answer directly and concisely. Avoid new numbers here; synthesize only from <OBSERVATION> and what the RGB/SVF/DSM/SEG images support. Do not introduce facts that are not visible or stated in <OBSERVATION>.</CONCLUSION>"
                    )
                    question_text_with_instruction = f"{qa_info['question']}\n\n{format_instruction}"
                    qa_pair = {
                        'scene_id': scene_data['id'],
                        'question_id': self.question_id_counter,
                        'text': question_text_with_instruction,
                        'answer': enhanced_answer,
                        'category': category,
                        'difficulty': 'medium',  # Default difficulty for freeform questions
                        'generation_method': f"freeform_{generation_method}",
                        'timestamp': datetime.now().isoformat(),
                        'dataset_type': dataset_type,  # Use the actual dataset type (train/test)
                        'image': self._convert_image_path(scene_data)
                    }
                    
                    # Store detailed metadata separately for YAML export (using special key)
                    qa_pair['__detailed_metadata__'] = {
                        'analysis_type': qa_info.get('analysis_type', category),
                        'metrics': qa_info.get('metrics', {}),
                        'context': context,
                        'enhanced_stats_used': enhanced_statistics is not None or generation_method == 'enhanced_fallback',
                        'model_used': self.model,
                        'area': scene_data.get('area', 'unknown'),
                        'data_focus': ['svf', 'dsm', 'seg', 'rgb'],
                        'quality_score': 0.85,
                        'bias_prevention_applied': True,
                        'verification': verification_result,
                        # Attach grid analysis for traceability if available
                        'grid_analysis': grid_metadata
                    }
                    
                    # Route by verification result
                    if verification_result.get('passed', False):
                        qa_pairs.append(qa_pair)
                    else:
                        # If failed, still record into a side file for debugging
                        try:
                            failed_path = os.path.join(self.metrics_dir, 'failed_verification.jsonl')
                            regen_log_path = os.path.join(self.metrics_dir, 'regeneration_log.jsonl')
                            os.makedirs(self.metrics_dir, exist_ok=True)
                            import json as _json
                            # Save failed item
                            with open(failed_path, 'a', encoding='utf-8') as _f:
                                clean_failed = {k: v for k, v in qa_pair.items() if k != '__detailed_metadata__'}
                                clean_failed['verification'] = verification_result
                                _json.dump(clean_failed, _f, ensure_ascii=False)
                                _f.write('\n')
                            if not self._printed_failed_verification_path:
                                print(f" Failed verification items saved to: {failed_path}")
                                self._printed_failed_verification_path = True
                            # Save regeneration diff if exists
                            try:
                                if 'regenerated' in generation_method:
                                    _diff = {
                                        'scene_id': scene_data['id'],
                                        'category': category,
                                        'question_id': qa_pair.get('question_id'),
                                        'original_answer': answer,
                                        'regenerated_answer': enhanced_answer,
                                        'verification': verification_result
                                    }
                                    with open(regen_log_path, 'a', encoding='utf-8') as _rf:
                                        _json.dump(_diff, _rf, ensure_ascii=False)
                                        _rf.write('\n')
                                    if not self._printed_regeneration_log_path:
                                        print(f" Regeneration diff log saved to: {regen_log_path}")
                                        self._printed_regeneration_log_path = True
                            except Exception:
                                pass
                        except Exception:
                            pass
                        if self.debug:
                            print(f"  QA skipped due to verification failure: {qa_info['question'][:60]}...")
                        # Do not increase question_id_counter when skipping
                        continue
                    self.question_id_counter += 1 # Increment counter for next question
                    
                    if self.debug:
                        print(f" Generated {category} QA for scene {scene_data['id']}")
                        
                except Exception as e:
                    print(f" Failed to generate {category} QA for scene {scene_data['id']}: {e}")
                    if self.debug:
                        import traceback
                        print(f"   Detailed error: {traceback.format_exc()}")
                    continue
                    
        except Exception as e:
            print(f" Failed to initialize analyzer for scene {scene_data['id']}: {e}")
            if self.debug:
                import traceback
                print(f"   Detailed error: {traceback.format_exc()}")
            return []
        
        return qa_pairs
    
    def _create_scene_statistics_from_data(self, scene_data: Dict[str, Any], qa_info: Dict[str, Any]) -> Optional['SceneStatistics']:
        """Create SceneStatistics object from available scene data and QA info"""
        from scene_statistics import SceneStatistics
        
        # Extract metrics from qa_info
        metrics = qa_info.get('metrics', {})
        
        # Get basic SVF statistics
        sky_vis = metrics.get('sky_visibility', {})
        svf_mean = sky_vis.get('mean', 0.5)
        svf_std = sky_vis.get('std', 0.2)
        svf_min = sky_vis.get('min', 0.1)
        svf_max = sky_vis.get('max', 0.9)
        svf_range = svf_max - svf_min
        svf_quartiles = [svf_min + svf_range * 0.25, svf_mean, svf_min + svf_range * 0.75]
        
        # Get terrain statistics
        terrain = metrics.get('terrain', {})
        height_mean = terrain.get('mean_height', 50.0)
        height_std = terrain.get('height_std', 15.0)
        height_range = terrain.get('elevation_range', 50.0)
        height_min = height_mean - height_range / 2
        height_max = height_mean + height_range / 2
        height_quartiles = [height_min + height_range * 0.25, height_mean, height_min + height_range * 0.75]
        
        # Get land cover statistics
        land_use = metrics.get('land_use', {})
        if not land_use:
            land_use = metrics.get('landcover', {})

        # Normalize land cover to fractions with safe default
        land_cover_ratios = {}
        total_land = sum(land_use.values()) if land_use else 0
        if total_land <= 0:
            # Safe default to avoid returning None and losing statistics_context
            land_cover_ratios = {'others': 1.0}
            total_land = 1.0
        else:
            for k, v in land_use.items():
                land_cover_ratios[k] = v / total_land
        
        # Calculate derived statistics
        vegetation_types = ['forest', 'grassland', 'agricultural']
        vegetation_ratio = sum(land_cover_ratios.get(t, 0) for t in vegetation_types)
        vegetation_ratio_natural = sum(land_cover_ratios.get(t, 0) for t in ['forest', 'grassland'])
        built_ratio = sum(land_cover_ratios.get(t, 0) for t in ['residential', 'buildings', 'commercial'])
        water_ratio = land_cover_ratios.get('water', 0)
        
        # Calculate spatial heterogeneity (Shannon diversity)
        spatial_heterogeneity = 0
        for ratio in land_cover_ratios.values():
            if ratio > 0:
                spatial_heterogeneity += -ratio * np.log(ratio)
        
        # Create SceneStatistics object
        stats = SceneStatistics(
            # SVF Statistics
            svf_mean=svf_mean,
            svf_std=svf_std,
            svf_min=svf_min,
            svf_max=svf_max,
            svf_range=svf_range,
            svf_quartiles=svf_quartiles,
            
            # Height Statistics
            height_mean=height_mean,
            height_std=height_std,
            height_min=height_min,
            height_max=height_max,
            height_range=height_range,
            height_quartiles=height_quartiles,
            
            # Land Cover
            land_cover_ratios=land_cover_ratios,
            vegetation_ratio=vegetation_ratio,
            built_ratio=built_ratio,
            water_ratio=water_ratio,
            vegetation_ratio_natural=vegetation_ratio_natural,
            vegetation_ratio_total=vegetation_ratio,
            
            # Spatial Characteristics
            spatial_heterogeneity=spatial_heterogeneity,
            grid_svf_variance=0.1,
            directional_asymmetry=0.05,
            
            # RGB/Color Statistics (defaults)
            rgb_mean=[120, 130, 115],
            rgb_std=[25, 30, 20],
            brightness=125.0,
            color_diversity=0.7,
            green_dominance=0.3,
            moderate_green_ratio=0.25,
            hue_entropy=1.8,
            saturation_mean=0.6,
            value_mean=0.5,
            green_vegetation_score=vegetation_ratio_natural,
            color_harmony_score=0.65,
            
            # Derived Metrics
            openness_score=svf_mean,
            urbanization_level=min(5, max(1, int(built_ratio * 5))),
            landscape_type='mixed_development' if 0.3 < built_ratio < 0.7 else 'urban' if built_ratio > 0.7 else 'natural',
            scenic_quality=min(1.0, max(0.0, vegetation_ratio + svf_mean * 0.3)),
            
            # Optional Grid Analysis
            grid_analysis=None
        )
        
        return stats
    
    def generate_qa_dataset_realtime(self, scenes: List[Dict[str, Any]], 
                                   target_questions: int = 50,
                                   output_path: str = None,
                                   dataset_type: str = "train",
                                   start_question_id: int = 1) -> List[Dict[str, Any]]:
        """Generate QA dataset with real-time saving and enhanced error handling"""
        
        # Set starting question ID
        self.question_id_counter = start_question_id
        
        all_qa_pairs = []
        questions_per_scene = len(self.categories)  # 4 questions per scene
        required_scenes = min(len(scenes), (target_questions + questions_per_scene - 1) // questions_per_scene)
        
        # Debug logging for question count control
        print(f" DEBUG: Question generation parameters:")
        print(f"   Target questions: {target_questions}")
        print(f"   Questions per scene: {questions_per_scene}")
        print(f"   Available scenes: {len(scenes)}")
        print(f"   Required scenes: {required_scenes}")
        print(f"   Expected output: {min(target_questions, required_scenes * questions_per_scene)} questions")
        
        # Enhanced validation and preparation
        if not scenes:
            print(" No scenes provided for QA generation")
            return []
        
        if target_questions <= 0:
            print(" Invalid target_questions: must be greater than 0")
            return []
        
        # Initialize performance tracking
        processing_stats = {
            'start_time': time.time(),
            'category_successes': {cat: 0 for cat in self.categories},
            'category_failures': {cat: 0 for cat in self.categories},
            'scenes_with_enhanced_stats': 0,
            'total_api_calls': 0,
            'fallback_answers': 0
        }
        
        print(f" Target: {target_questions} questions")
        print(f"  Categories per scene: {questions_per_scene}")
        print(f"  Scenes to process: {required_scenes}")
        print(f" Expected output: ~{required_scenes * questions_per_scene} questions")
        print(f" Enhanced GPT-4 generator: {'Available' if self.enhanced_gpt4_generator else 'Not Available'}")
        print(f" Real-time saving: {'Enabled' if output_path else 'Disabled'}")
        
        # Process scenes with proper target question control
        processed_scenes = 0
        skipped_scenes = 0
        
        print(f"\n Starting scene processing (max {required_scenes} scenes)...")
        
        for i, scene in enumerate(scenes[:required_scenes]):
            try:
                print(f"\n Processing scene {processed_scenes + 1}/{required_scenes}: {scene['id']}")
                
                # Early termination check before processing scene
                if len(all_qa_pairs) >= target_questions:
                    print(f" Target questions ({target_questions}) already reached with {len(all_qa_pairs)} questions. Stopping.")
                    break
                
                # Generate QA pairs for this scene
                scene_qa_pairs = self.generate_qa_from_scene(scene, dataset_type)
                
                if scene_qa_pairs:
                    # Check if adding these questions would exceed target
                    questions_to_add = len(scene_qa_pairs)
                    current_count = len(all_qa_pairs)
                    
                    if current_count + questions_to_add > target_questions:
                        # Truncate to exact target
                        questions_needed = target_questions - current_count
                        scene_qa_pairs = scene_qa_pairs[:questions_needed]
                        print(f" Truncating scene output to {len(scene_qa_pairs)} questions to meet target")
                    
                    all_qa_pairs.extend(scene_qa_pairs)
                    processed_scenes += 1
                    
                    print(f" Added {len(scene_qa_pairs)} QA pairs from scene {scene['id']} (Total: {len(all_qa_pairs)}/{target_questions})")
                    
                    # Real-time saving
                    if output_path:
                        self._save_realtime(scene_qa_pairs, output_path)
                    
                    # Check if we've reached the exact target
                    if len(all_qa_pairs) >= target_questions:
                        print(f" Exact target reached: {len(all_qa_pairs)} questions. Stopping scene processing.")
                        break
                else:
                    print(f"  No QA pairs generated for scene {scene['id']}")
                    skipped_scenes += 1
                    
            except Exception as e:
                print(f" Error processing scene {scene['id']}: {e}")
                skipped_scenes += 1
                continue
        
        # Final verification and truncation (should not be needed with improved logic)
        if len(all_qa_pairs) > target_questions:
            print(f"  Final truncation: {len(all_qa_pairs)} -> {target_questions} questions")
            all_qa_pairs = all_qa_pairs[:target_questions]
        elif len(all_qa_pairs) < target_questions:
            print(f"  Generated fewer questions than target: {len(all_qa_pairs)}/{target_questions}")
        
        # Finalize processing statistics
        processing_stats['end_time'] = time.time()
        processing_stats['total_processing_time'] = processing_stats['end_time'] - processing_stats['start_time']
        processing_stats['processed_scenes'] = processed_scenes
        processing_stats['skipped_scenes'] = skipped_scenes
        processing_stats['questions_generated'] = len(all_qa_pairs)
        
        print(f"\n Final dataset statistics:")
        print(f"   Total QA pairs: {len(all_qa_pairs)}")
        print(f"   Processed scenes: {processed_scenes}")
        print(f"   Skipped scenes: {skipped_scenes}")
        
        # Category distribution
        category_counts = {}
        for qa in all_qa_pairs:
            category = qa.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        print(f" Category distribution:")
        for category, count in sorted(category_counts.items()):
            print(f"   {category}: {count} questions")
        
        # Save detailed metrics and metadata to YAML files
        if output_path and all_qa_pairs:
            try:
                print(f"\n Saving detailed metrics and metadata...")
                self._save_metrics_yaml(all_qa_pairs, output_path)
                self._save_generation_log_yaml(processing_stats, output_path)
                
                # Export verification files
                print(f" Exporting verification files...")
                verification_files = self.verification_exporter.export_verification_files(
                    filename_prefix=f"freeform_{dataset_type}_{self.model.replace('-', '_')}"
                )
                print(f" Metadata and verification files saved successfully")
                
                # Print verification file paths for user reference
                if verification_files:
                    print(f"\n Verification files created:")
                    for file_type, file_path in verification_files.items():
                        print(f"   {file_type}: {os.path.basename(file_path)}")
                        
            except Exception as e:
                print(f"  Failed to save metadata files: {e}")
                if self.debug:
                    import traceback
                    print(f"   Detailed error: {traceback.format_exc()}")
        
        return all_qa_pairs
    
    def _save_realtime(self, qa_pairs: List[Dict[str, Any]], output_path: str):
        """Save QA pairs in real-time with enhanced error handling (clean format)"""
        if not qa_pairs:
            return
        
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Check if this is the first write to the file
            file_exists = os.path.exists(output_path)
            mode = 'a' if file_exists else 'w'
            
            with open(output_path, mode, encoding='utf-8') as f:
                for qa in qa_pairs:
                    # Validate QA pair before saving
                    if self._validate_qa_pair(qa):
                        # Create clean QA pair without heavy metadata
                        clean_qa = self._create_clean_qa_format(qa)
                        json.dump(clean_qa, f, ensure_ascii=False)
                        f.write('\n')
                    elif self.debug:
                        print(f"     Skipped invalid QA pair: {qa.get('question_id', 'unknown')}")
        except IOError as e:
            print(f" File I/O error saving real-time data: {e}")
            raise
        except Exception as e:
            print(f"  Unexpected error saving real-time data: {e}")
            if self.debug:
                import traceback
                print(f"   Detailed error: {traceback.format_exc()}")
            raise
    
    def _create_clean_qa_format(self, qa: Dict[str, Any]) -> Dict[str, Any]:
        """Create clean QA format by removing heavy metadata"""
        clean_qa = {
            'scene_id': qa.get('scene_id'),
            'question_id': qa.get('question_id'),
            'text': qa.get('text'),
            'answer': qa.get('answer'),
            'category': qa.get('category'),
            'difficulty': qa.get('difficulty', 'medium'),
            'generation_method': qa.get('generation_method'),
            'dataset_type': qa.get('dataset_type'),
            'image': qa.get('image'),
            'timestamp': qa.get('timestamp')
        }
        # Remove detailed metadata if present
        clean_qa.pop('__detailed_metadata__', None)
        return clean_qa
    
    def _save_metrics_yaml(self, qa_pairs: List[Dict[str, Any]], output_path: str):
        """Save detailed metrics and metadata to YAML file"""
        if not qa_pairs:
            return
        
        # Generate metrics file path
        base_path = Path(output_path).stem
        output_dir = os.path.dirname(output_path)
        metrics_path = os.path.join(output_dir, f"{base_path}_metrics.yaml")
        
        try:
            # Collect all metrics and metadata
            metrics_data = {
                'generation_summary': {
                    'timestamp': datetime.now().isoformat(),
                    'total_questions': len(qa_pairs),
                    'generation_method': 'freeform_analysis',
                    'model_used': self.model,
                    'gpt4_enabled': self.use_gpt4,
                    'enhanced_gpt4_available': self.enhanced_gpt4_generator is not None
                },
                'category_distribution': {},
                'scene_analysis': {},
                'quality_metrics': [],
                'detailed_metadata': []
            }
            
            # Analyze category distribution
            for qa in qa_pairs:
                category = qa.get('category', 'unknown')
                metrics_data['category_distribution'][category] = metrics_data['category_distribution'].get(category, 0) + 1
            
            # Collect detailed metadata from each QA pair
            for qa in qa_pairs:
                # Extract detailed metadata if available
                detailed_meta = qa.get('__detailed_metadata__', {})
                if detailed_meta:
                    scene_id = qa.get('scene_id')
                    
                    # Scene analysis
                    if scene_id not in metrics_data['scene_analysis']:
                        metrics_data['scene_analysis'][scene_id] = {
                            'area': detailed_meta.get('area'),
                            'metrics': detailed_meta.get('metrics', {}),
                            'enhanced_stats_used': detailed_meta.get('enhanced_stats_used', False),
                            'questions_generated': 0
                        }
                    metrics_data['scene_analysis'][scene_id]['questions_generated'] += 1
                    
                    # Quality metrics
                    quality_info = {
                        'question_id': qa.get('question_id'),
                        'quality_score': detailed_meta.get('quality_score', 0.85),
                        'bias_prevention_applied': detailed_meta.get('bias_prevention_applied', True),
                        'category': qa.get('category'),
                        'generation_method': qa.get('generation_method')
                    }
                    metrics_data['quality_metrics'].append(quality_info)
                    
                    # Detailed metadata (without heavy data)
                    clean_metadata = {
                        'question_id': qa.get('question_id'),
                        'scene_id': scene_id,
                        'category': qa.get('category'),
                        'analysis_type': detailed_meta.get('analysis_type'),
                        'model_used': detailed_meta.get('model_used'),
                        'timestamp': qa.get('timestamp')
                    }
                    metrics_data['detailed_metadata'].append(clean_metadata)

                    # Attach compact 3x3 grid analysis to scene_analysis if available
                    grid_meta = detailed_meta.get('grid_analysis')
                    if grid_meta:
                        try:
                            if 'grid_analysis' not in metrics_data['scene_analysis'][scene_id]:
                                metrics_data['scene_analysis'][scene_id]['grid_analysis'] = grid_meta
                        except Exception:
                            pass
            
            # Save YAML file
            with open(metrics_path, 'w', encoding='utf-8') as f:
                yaml.dump(metrics_data, f, default_flow_style=False, allow_unicode=True, indent=2)
                
            print(f" Metrics saved to: {metrics_path}")
            
        except Exception as e:
            print(f"  Failed to save metrics YAML: {e}")
            if self.debug:
                import traceback
                print(f"   Detailed error: {traceback.format_exc()}")
    
    def _save_generation_log_yaml(self, processing_stats: Dict[str, Any], output_path: str):
        """Save generation process log to YAML file"""
        base_path = Path(output_path).stem
        output_dir = os.path.dirname(output_path)
        log_path = os.path.join(output_dir, f"{base_path}_generation_log.yaml")
        
        try:
            log_data = {
                'generation_log': {
                    'timestamp': datetime.now().isoformat(),
                    'processing_statistics': processing_stats,
                    'generation_settings': {
                        'model': self.model,
                        'use_gpt4': self.use_gpt4,
                        'categories': self.categories,
                        'enhanced_generator_available': self.enhanced_gpt4_generator is not None
                    }
                }
            }
            
            with open(log_path, 'w', encoding='utf-8') as f:
                yaml.dump(log_data, f, default_flow_style=False, allow_unicode=True, indent=2)
                
            print(f" Generation log saved to: {log_path}")
            
        except Exception as e:
            print(f"  Failed to save generation log: {e}")
            if self.debug:
                import traceback
                print(f"   Detailed error: {traceback.format_exc()}")
    
    def _validate_scene_data(self, scene: Dict[str, Any]) -> bool:
        """Validate scene data before processing"""
        required_keys = ['svf', 'dsm', 'seg', 'id']
        
        for key in required_keys:
            if key not in scene:
                print(f"    Missing required key: {key}")
                return False
        
        # Check data types and shapes
        try:
            import numpy as np
            svf = scene['svf']
            dsm = scene['dsm']
            seg = scene['seg']
            
            if not all(isinstance(arr, np.ndarray) for arr in [svf, dsm, seg]):
                print(f"    Invalid data types - expected numpy arrays")
                return False
            
            # Check for minimum data size
            if any(arr.size < 100 for arr in [svf, dsm, seg]):
                print(f"    Data arrays too small (< 100 pixels)")
                return False
            
            # Check for reasonable value ranges
            if not (0 <= np.nanmin(svf) and np.nanmax(svf) <= 1):
                print(f"     SVF values outside expected range [0,1]")
                # Don't fail, just warn
            
            return True
            
        except Exception as e:
            print(f"    Scene validation error: {e}")
            return False
    
    def _validate_qa_pair(self, qa: Dict[str, Any]) -> bool:
        """Validate QA pair before saving"""
        required_fields = ['question_id', 'text', 'answer']
        
        for field in required_fields:
            if field not in qa or not qa[field]:
                return False
        
        # Check for reasonable answer length
        answer_length = len(qa['answer'])
        if answer_length < 10 or answer_length > 2000:
            return False
        
        return True
    
    def visualize_and_save_statistics(self, scenes: List[Dict[str, Any]], 
                                    output_dir: str) -> Dict[str, Any]:
        """Visualize scene statistics with enhanced analysis and save data"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for organized output
        stats_dir = os.path.join(output_dir, "statistics")
        viz_dir = os.path.join(output_dir, "visualizations")
        enhanced_dir = os.path.join(output_dir, "enhanced_analysis")
        
        for dir_path in [stats_dir, viz_dir, enhanced_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        successful_visualizations = 0
        successful_statistics = 0
        successful_enhanced = 0
        
        for i, scene in enumerate(scenes):
            try:
                scene_id = scene['id']
                
                # Create enhanced scene statistics if available
                enhanced_stats = None
                if self.enhanced_gpt4_generator:
                    try:
                        enhanced_stats = self.enhanced_gpt4_generator.extract_scene_statistics(scene)
                        if enhanced_stats:
                            # Save enhanced statistics
                            enhanced_file = os.path.join(enhanced_dir, f"scene_{i:03d}_{scene_id}_enhanced.json")
                            with open(enhanced_file, 'w', encoding='utf-8') as f:
                                json.dump(enhanced_stats.to_dict(), f, indent=2, ensure_ascii=False)
                            successful_enhanced += 1
                            if self.debug:
                                print(f" Enhanced statistics saved for scene {scene_id}")
                    except Exception as e:
                        if self.debug:
                            print(f"  Enhanced statistics failed for scene {scene_id}: {e}")
                
                # Create basic scene statistics as fallback
                scene_stats = SceneStatistics({
                    'svf': scene['svf'],
                    'dsm': scene['dsm'],
                    'seg': scene['seg'],
                    'rgb': scene['rgb']
                })
                
                # Save basic statistics as JSON
                stats_file = os.path.join(stats_dir, f"scene_{i:03d}_{scene_id}_basic_statistics.json")
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump(scene_stats.to_dict(), f, indent=2, ensure_ascii=False)
                successful_statistics += 1
                
                # Visualization (if available)
                try:
                    import matplotlib.pyplot as plt
                    
                    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
                    fig.suptitle(f'Scene {scene_id} - Multi-modal Data', fontsize=16)
                    
                    # RGB
                    if scene['rgb'] is not None:
                        axes[0, 0].imshow(scene['rgb'])
                        axes[0, 0].set_title('RGB Image')
                        axes[0, 0].axis('off')
                    
                    # SVF
                    im1 = axes[0, 1].imshow(scene['svf'], cmap='viridis', vmin=0, vmax=1)
                    axes[0, 1].set_title('Sky View Factor')
                    axes[0, 1].axis('off')
                    plt.colorbar(im1, ax=axes[0, 1])
                    
                    # DSM
                    im2 = axes[1, 0].imshow(scene['dsm'], cmap='terrain')
                    axes[1, 0].set_title('Digital Surface Model')
                    axes[1, 0].axis('off')
                    plt.colorbar(im2, ax=axes[1, 0])
                    
                    # Segmentation
                    im3 = axes[1, 1].imshow(scene['seg'], cmap='tab20')
                    axes[1, 1].set_title('Land Cover Segmentation')
                    axes[1, 1].axis('off')
                    plt.colorbar(im3, ax=axes[1, 1])
                    
                    # Save enhanced visualization
                    viz_file = os.path.join(viz_dir, f"scene_{i:03d}_{scene_id}_multimodal.png")
                    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    # Additional analysis visualization if enhanced stats available
                    if enhanced_stats and self.debug:
                        try:
                            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                            fig.suptitle(f'Enhanced Analysis - Scene {scene_id}', fontsize=14)
                            
                            # Statistics comparison
                            basic_openness = getattr(scene_stats, 'openness_score', 0)
                            enhanced_openness = getattr(enhanced_stats, 'openness_score', 0)
                            
                            categories = ['Basic\nOpenness', 'Enhanced\nOpenness', 'Vegetation\nScore', 'Color\nHarmony']
                            values = [
                                basic_openness,
                                enhanced_openness,
                                getattr(enhanced_stats, 'vegetation_ratio', 0),
                                getattr(enhanced_stats, 'color_harmony_score', 0)
                            ]
                            
                            axes[0].bar(categories, values, color=['skyblue', 'lightgreen', 'forestgreen', 'gold'])
                            axes[0].set_title('Analysis Comparison')
                            axes[0].set_ylabel('Score (0-1)')
                            axes[0].set_ylim(0, 1)
                            
                            # Grid analysis if available
                            if hasattr(enhanced_stats, 'grid_analysis') and enhanced_stats.grid_analysis:
                                grid_data = enhanced_stats.grid_analysis.get('svf_grid', np.zeros((3, 3)))
                                im = axes[1].imshow(grid_data, cmap='viridis', vmin=0, vmax=1)
                                axes[1].set_title('3x3 SVF Grid Analysis')
                                plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
                            else:
                                axes[1].text(0.5, 0.5, 'Grid Analysis\nNot Available', 
                                           ha='center', va='center', transform=axes[1].transAxes)
                                axes[1].set_title('Grid Analysis')
                            
                            enhanced_viz_file = os.path.join(viz_dir, f"scene_{i:03d}_{scene_id}_enhanced_analysis.png")
                            plt.savefig(enhanced_viz_file, dpi=150, bbox_inches='tight')
                            plt.close()
                            
                        except Exception as viz_e:
                            if self.debug:
                                print(f"  Enhanced visualization failed for scene {scene_id}: {viz_e}")
                    
                    successful_visualizations += 1
                    
                except ImportError:
                    if self.debug:
                        print("  Matplotlib not available for visualization")
                except Exception as e:
                    if self.debug:
                        print(f"  Visualization failed for scene {scene_id}: {e}")
                
                if self.debug:
                    print(f" Saved analysis for scene {scene_id}")
                    
            except Exception as e:
                print(f" Failed to analyze scene {scene.get('id', f'scene_{i}')}: {e}")
                if self.debug:
                    import traceback
                    print(f"   Detailed error: {traceback.format_exc()}")
                continue
        
        summary = {
            'total_scenes': len(scenes),
            'successful_visualizations': successful_visualizations,
            'successful_statistics': successful_statistics,
            'successful_enhanced': successful_enhanced,
            'output_directory': output_dir,
            'subdirectories': {
                'basic_statistics': stats_dir,
                'visualizations': viz_dir,
                'enhanced_analysis': enhanced_dir
            },
            'enhanced_stats_available': self.enhanced_gpt4_generator is not None
        }
        
        return summary


def main_with_real_data(image_dir: str, svf_dir: str,
                       target_questions: int = 50, max_scenes: int = None,
                       mode: str = "both", output_dir: str = "calc_svf/outputs",
                       model: str = "gpt-4o-mini", use_gpt4: bool = True):
    """
    Main function for real data processing (GeoNRW dataset)
    
    Args:
        image_dir: Directory containing RGB, SEG, DSM images
        svf_dir: Directory containing SVF files
        target_questions: Target number of questions
        max_scenes: Maximum number of scenes to process
        mode: Processing mode ("train", "test", "both", "visualize")
        output_dir: Output directory
        model: GPT model to use
        use_gpt4: Whether to use GPT-4 for answer generation
    """
    print(f"\n Free-form Analysis QA Generation System")
    print(f"{'='*80}")
    print(f" Using model: {model}")
    print(f" GPT-4 enabled: {use_gpt4}")
    
    # Environment variable check
    if use_gpt4 and not os.getenv('OPENAI_API_KEY'):
        print("  OPENAI_API_KEY not set. Running in statistical fallback mode.")
        print("   Set with: export OPENAI_API_KEY='your-api-key-here'")
        use_gpt4 = False
    
    # Initialize generator
    today = datetime.now().strftime('%m%d')
    generator = FreeformQAGenerator(
        debug=False,
        use_gpt4=use_gpt4,
        model=model,
        output_dir=f'{output_dir}/{today}'
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Processing modes
    results = {}
    modes_to_process = []
    
    # Global question ID counter for continuous numbering across train/test
    global_question_id = 1
    
    if mode == "train":
        modes_to_process.append("train")
    elif mode == "test":
        modes_to_process.append("test")
    elif mode == "both":
        modes_to_process = ["train", "test"]
    elif mode == "visualize":
        modes_to_process = ["train", "test"]
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'test', 'both', or 'visualize'")
    
    for current_mode in modes_to_process:
        print(f"\n{'='*80}")
        if mode == "visualize":
            print(f"   VISUALIZE MODE - {current_mode.upper()} dataset")
        else:
            print(f"  {current_mode.upper()} dataset generation")
        print(f"{'='*80}")
        
        # Mode-specific SVF directory
        if current_mode == "train":
            mode_svf_dir = os.path.join(svf_dir, "skyview_umep_train")
        else:  # test
            mode_svf_dir = os.path.join(svf_dir, "skyview_umep_test")
        
        # Directory existence check
        if not os.path.exists(mode_svf_dir):
            print(f"  {current_mode} SVF directory not found: {mode_svf_dir}")
            print(f"   Using specified directory: {svf_dir}")
            mode_svf_dir = svf_dir
        
        print(f" Loading {current_mode} scenes from directories:")
        print(f"   Image Dir (RGB/SEG/DSM): {image_dir}")
        print(f"   SVF Dir: {mode_svf_dir}")
        
        # Mode-specific question count adjustment
        mode_target_questions = target_questions
        if mode == "both":
            if current_mode == "train":
                mode_target_questions = int(target_questions * 0.7)  # 70% for train
            else:  # test
                mode_target_questions = int(target_questions * 0.3)  # 30% for test
        
        print(f" Processing parameters for {current_mode}:")
        print(f"   Target questions (mode-specific): {mode_target_questions}")
        print(f"   Max scenes limit: {max_scenes if max_scenes else 'No limit'}")
        
        # Load scene data
        scenes = load_scenes_from_svf_directory(
            mode_svf_dir, image_dir,
            max_scenes=max_scenes, debug=False
        )
        
        if not scenes:
            print(f"  No scenes found for {current_mode}")
            continue
        
        print(f" Successfully loaded {len(scenes)} {current_mode} scenes")
        
        # Generate output filename compatible with existing pattern
        current_date = datetime.now().strftime('%Y%m%d')
        model_suffix = model.replace("-", "_").replace(".", "_")
        method_suffix = "freeform_gpt4" if use_gpt4 else "freeform_statistical"
        output_filename = f"free_qas_{current_date}_{current_mode}_{mode_target_questions}q_{model_suffix}.jsonl"
        output_path = os.path.join(output_dir, output_filename)
        if os.path.exists(output_path):
            import shutil
            backup_path = output_path.replace(".jsonl", "_backup.jsonl")
            shutil.copy(output_path, backup_path)
            os.remove(output_path)  # Remove original to prevent appending
            print(f" Existing file backed up to: {os.path.basename(backup_path)}")
        
        print(f" Output file: {output_path}")
        
        # Handle visualize mode
        if mode == "visualize":
            viz_output_dir = os.path.join(output_dir, f"freeform_visualization_output_{current_mode}_{mode_target_questions}q_{current_date}")
            
            if os.path.exists(viz_output_dir):
                import shutil
                shutil.rmtree(viz_output_dir)
                print(f"  Existing visualization directory removed: {viz_output_dir}")
            
            summary = generator.visualize_and_save_statistics(scenes, viz_output_dir)
            results[current_mode] = {
                'summary': summary,
                'output_dir': viz_output_dir,
                'scene_count': len(scenes)
            }
            continue
        
        # Generate QA dataset
        dataset = generator.generate_qa_dataset_realtime(
            scenes,
            target_questions=mode_target_questions,
            output_path=output_path,
            dataset_type=current_mode,
            start_question_id=global_question_id
        )
        
        # Update global question ID counter for next mode
        global_question_id += len(dataset)
        
        # Display results summary
        print(f"\n{'='*80}")
        print(f"  {current_mode.upper()} Dataset Generation Summary")
        print(f"{'='*80}")
        print(f" Total QA pairs: {len(dataset)}")
        print(f" Saved to: {output_path}")
        
        # Category distribution
        category_counts = {}
        for qa in dataset:
            category = qa.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        print(f" Category distribution:")
        for category, count in sorted(category_counts.items()):
            print(f"   {category}: {count} questions")
        
        # Save results
        results[current_mode] = {
            'dataset': dataset,
            'output_path': output_path,
            'scene_count': len(scenes),
            'qa_count': len(dataset)
        }
    
    # Final summary
    print(f"\n{'='*80}")
    if mode == "visualize":
        print("   VISUALIZATION SUMMARY ")
        print(f"{'='*80}")
        
        total_scene_count = 0
        for mode_name, result in results.items():
            total_scene_count += result['scene_count']
            output_path = result.get('output_dir', 'unknown')
            print(f"{mode_name.upper():5s}: {result['scene_count']:3d} scenes -> {os.path.basename(output_path)}")
            if result.get('summary'):
                summary = result['summary']
                print(f"          Successful visualizations: {summary['successful_visualizations']}")
                print(f"          Successful statistics: {summary['successful_statistics']}")
        
        print(f"{'TOTAL':5s}: {total_scene_count:3d} scenes")
    else:
        print("   FINAL GENERATION SUMMARY ")
        print(f"{'='*80}")
        
        total_qa_count = 0
        for mode_name, result in results.items():
            qa_count = result.get('qa_count', 0)
            total_qa_count += qa_count
            output_path = result.get('output_path', 'unknown')
            print(f"{mode_name.upper():5s}: {qa_count:4d} questions ({result['scene_count']:3d} scenes) -> {os.path.basename(output_path)}")
        
        print(f"{'TOTAL':5s}: {total_qa_count:4d} questions")
    
    print(f"\n Output directory: {output_dir}")
    print(f"{'='*80}")
    
    return results


def main_with_sample_data(model: str = "gpt-4o-mini", use_gpt4: bool = True):
    """Main function for sample data testing"""
    print(f"\n Free-form Analysis QA Generation System (Sample Data)")
    print(f"{'='*80}")
    print(f" Using model: {model}")
    print(f" GPT-4 enabled: {use_gpt4}")
    
    # Environment variable check
    if use_gpt4 and not os.getenv('OPENAI_API_KEY'):
        print("  OPENAI_API_KEY not set. Running in statistical fallback mode.")
        use_gpt4 = False
    
    # Initialize generator
    generator = FreeformQAGenerator(
        debug=False,
        use_gpt4=use_gpt4,
        model=model
    )
    
    # Generate sample scenes
    scenes = [create_sample_scene(f"sample_{i:03d}") for i in range(3)]
    print(f" Generated {len(scenes)} sample scenes")
    
    # Generate output filename
    current_date = datetime.now().strftime('%Y%m%d')
    model_suffix = model.replace("-", "_").replace(".", "_")
    method_suffix = "freeform_gpt4" if use_gpt4 else "freeform_statistical"
    target_questions = 12  # 3 scenes * 4 categories
    output_path = f"calc_svf/outputs/free_qa_{current_date}_{model_suffix}_{target_questions}q.jsonl"
    
    # Remove existing file
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"  Existing sample output file removed: {output_path}")
    
    print(f" Output file: {output_path}")
    
    # Generate QA dataset
    dataset = generator.generate_qa_dataset_realtime(
        scenes,
        target_questions=target_questions,
        output_path=output_path
    )
    
    # Display results
    print(f"\n{'='*80}")
    print("   SAMPLE DATA GENERATION COMPLETE ")
    print(f"{'='*80}")
    print(f" Total QA pairs: {len(dataset)}")
    print(f" Saved to: {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Free-form Analysis QA Generation System")
    parser.add_argument("--mode", type=str, choices=["train", "test", "both", "visualize"], default="both",
                        help="Processing mode: 'train', 'test', 'both', or 'visualize' (default: both)")
    parser.add_argument("--target_questions", type=int, default=50,
                        help="Target number of questions (default: 50)")
    parser.add_argument("--max_scenes", type=int, default=None,
                        help="Maximum number of scenes (default: no limit)")
    parser.add_argument("--image_dir", type=str, default="../../SynRS3D/GeoNRW_dsm",
                        help="Image directory path (default: ../SynRS3D/GeoNRW_dsm)")
    parser.add_argument("--svf_dir", type=str, default="../../SynRS3D/GeoNRW_dsm/svf",
                        help="SVF directory path (default: ../SynRS3D/GeoNRW_dsm/svf)")
    parser.add_argument("--output_dir", type=str, default="free_qa/",
                        help="Output directory path (default: calc_svf/outputs)")
    parser.add_argument("--use_sample", action="store_true",
                        help="Use sample data instead of real data")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="GPT model name (default: gpt-4o-mini). Examples: gpt-4.1-mini, gpt-4o, gpt-5")
    parser.add_argument("--disable_gpt4", action="store_true",
                        help="Disable GPT-4 and use statistical fallback only")
    parser.add_argument("--enable_verification", action="store_true",
                        help="Enable verification file export (metrics and prompts for manual checking)")
    
    parser.epilog = """
Usage Examples:
  # Free-form QA generation with GPT-4 (default)
  python calc_svf/statistics/freeform_main.py --mode both --target_questions 50 --max_scenes 20
  
  # Statistical fallback mode (no GPT-4)
  python calc_svf/statistics/freeform_main.py --disable_gpt4 --target_questions 20
  
  # Visualization mode
  python calc_svf/statistics/freeform_main.py --mode visualize --max_scenes 5 --target_questions 20
  
  # Sample data testing
  python calc_svf/statistics/freeform_main.py --use_sample

Free-form Categories:
  - urban_development_application: Urban planning and development analysis
  - renewable_energy_installation: Solar and wind energy potential
  - landscape_analysis: Comprehensive landscape assessment
  - water_accumulation: Water flow and accumulation analysis

Output Format:
  JSONL file with each line containing:
  {
    "question_id": "scene_id_category",
    "text": "Question text...",
    "answer": "Detailed analysis answer...",
    "image": "path/to/image",
    "category": "category_name",
    "generation_method": "gpt4|statistical",
    "scene_id": "scene_identifier",
    "area": "area_name",
    "timestamp": "ISO timestamp",
    "metrics": {...},
    "context": [...]
  }
"""
    
    args = parser.parse_args()
    
    use_gpt4 = not args.disable_gpt4
    
    print(f"\n Free-form Analysis QA Generation System")
    print(f"{'='*80}")
    print(f" Mode: {args.mode}")
    print(f" Target Questions: {args.target_questions}")
    if args.max_scenes:
        print(f" Max Scenes: {args.max_scenes}")
    print(f" Model: {args.model}")
    print(f" GPT-4 Enabled: {use_gpt4}")
    print(f"{'='*80}")
    
    if args.use_sample:
        print(" Running with SAMPLE DATA")
        main_with_sample_data(args.model, use_gpt4)
    else:
        print("  Running with REAL DATA")
        
        # Directory existence check
        dirs_to_check = [args.image_dir]
        
        # Check required directories based on mode
        svf_base_dir = args.svf_dir
        if args.mode in ["train", "both", "visualize"]:
            dirs_to_check.append(os.path.join(svf_base_dir, "skyview_umep_train"))
        if args.mode in ["test", "both", "visualize"]:
            dirs_to_check.append(os.path.join(svf_base_dir, "skyview_umep_test"))
        
        missing_dirs = [d for d in dirs_to_check if not os.path.exists(d)]
        
        if missing_dirs:
            print(f" Missing directories: {missing_dirs}")
            print("   Please check paths. Running with sample data instead.")
            print(f"   Required directory structure:")
            print(f"     {args.image_dir}")
            print(f"     {os.path.join(svf_base_dir, 'skyview_umep_train')}")
            print(f"     {os.path.join(svf_base_dir, 'skyview_umep_test')}")
            main_with_sample_data(args.model, use_gpt4)
        else:
            print(" All directories exist. Running with real data.")
            
            # Execute with real data
            results = main_with_real_data(
                image_dir=args.image_dir,
                svf_dir=svf_base_dir,
                target_questions=args.target_questions,
                max_scenes=args.max_scenes,
                mode=args.mode,
                output_dir=args.output_dir,
                model=args.model,
                use_gpt4=use_gpt4
            )
            
            if results:
                print("\n Execution completed!")
                for mode_name, result in results.items():
                    if 'qa_count' in result:
                        print(f"  {mode_name.upper()}: {result['qa_count']} questions -> {os.path.basename(result['output_path'])}")
                    elif 'scene_count' in result:
                        print(f"  {mode_name.upper()}: {result['scene_count']} scenes -> {os.path.basename(result.get('output_dir', 'unknown'))}")
            else:
                print("  Execution failed") 