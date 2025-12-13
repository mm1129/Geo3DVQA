"""
Free-form Analysis Categories for SVF-based VQA System
Based on paper_figure1.png - Green color section (Free-form Analysis)

This module implements comprehensive free-form analysis questions that require
multi-modal integration and complex reasoning beyond simple choice selection.

Updated to support GPT-4o-mini powered answer generation.
"""

import numpy as np
import random
import os
from typing import List, Dict, Any, Optional, Tuple
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpt4_freeform_answer_generator import GPT4FreeformAnswerGenerator

class FreeformAnalysisCategories:
    """
    Free-form analysis categories from paper_figure1.png
    
    Categories implemented:
    1. Urban Development/land-use Application
    2. Renewable Energy Installation  
    3. Landscape Analysis
    4. Water accumulation
    
    Enhanced with GPT-4o-mini answer generation support.
    """
    
    def __init__(self, svf_map, height_map=None, segmentation_map=None, rgb_image=None, 
                 file_path=None, debug=False, use_gpt4=True, api_key=None, model="gpt-4o-mini"):
        """
        Initialize free-form analysis system.
        
        Args:
            svf_map: Sky View Factor map
            height_map: Digital Surface Model (optional)
            segmentation_map: Land cover segmentation (optional)
            rgb_image: RGB imagery (optional)
            file_path: Source file path for context
            debug: Enable debug output
            use_gpt4: Enable GPT-4 powered answer generation (default: True)
            api_key: OpenAI API key (optional, uses environment variable if None)
            model: GPT model to use (default: gpt-4o-mini)
        """
        self.svf_map = svf_map
        self.height_map = height_map
        self.segmentation_map = segmentation_map
        self.rgb_image = rgb_image
        self.file_path = file_path
        self._debug = debug
        self.use_gpt4 = use_gpt4
        
        # Landcover class definitions (GeoNRW compliant)
        self.LANDCOVER_CLASSES = {
            0: 'others', 1: 'forest', 2: 'water', 3: 'agricultural', 
            4: 'residential', 5: 'grassland', 6: 'railways', 
            7: 'highways', 8: 'airports', 9: 'roads', 10: 'buildings'
        }
        
        # Initialize GPT-4 answer generator if enabled
        self.gpt4_generator = None
        if self.use_gpt4:
            try:
                
                self.gpt4_generator = GPT4FreeformAnswerGenerator(
                    api_key=api_key, 
                    model=model, 
                    debug=debug
                )
                if debug:
                    print(" GPT-4 answer generator initialized successfully")
            except ImportError as e:
                if debug:
                    print(f"  GPT-4 generator not available: {e}")
                self.use_gpt4 = False
            except Exception as e:
                if debug:
                    print(f"  Failed to initialize GPT-4 generator: {e}")
                self.use_gpt4 = False
        
        # Compute scene statistics for GPT-4 integration
        self._scene_statistics = None
        if self.use_gpt4 and self.gpt4_generator:
            self._scene_statistics = self._compute_scene_statistics()
            # Attach simple 3x3 grid summary when not provided by upstream statistics
            try:
                if self._scene_statistics is not None and not hasattr(self._scene_statistics, 'grid_analysis') and not (
                    hasattr(self._scene_statistics, 'grid_analysis_human_readable') or hasattr(self._scene_statistics, 'svf_grid_summary_3x3')
                ):
                    summary = self._compute_simple_3x3_grid_summary()
                    if summary:
                        # Prefer explicit SVF naming for LLM clarity, keep legacy key for compatibility
                        setattr(self._scene_statistics, 'svf_grid_summary_3x3', summary)
                        setattr(self._scene_statistics, 'grid_analysis_human_readable', summary)
                        if self._debug:
                            print("[freeform] attached simple 3x3 SVF grid summary to statistics (svf_grid_summary_3x3)")
            except Exception:
                pass
        
    def _compute_scene_statistics(self) -> Optional['SceneStatistics']:
        """Compute scene statistics for GPT-4 integration"""
        try:

            from scene_statistics import SceneStatistics
            
            # Create scene data dictionary
            scene_data = {
                'svf': self.svf_map,
                'dsm': self.height_map,
                'seg': self.segmentation_map,
                'rgb': self.rgb_image
            }
            
            return SceneStatistics(scene_data)
        except Exception as e:
            if self._debug:
                print(f"  Failed to compute scene statistics: {e}")
            return None

    def _compute_simple_3x3_grid_summary(self) -> Optional[Dict[str, Any]]:
        """Compute a minimal 3x3 grid summary using SVF as primary signal.

        Produces human-readable names and top locations to guide location wording.
        Conservative: uses only global SVF map; does not cross inconsistent modalities.
        """
        try:
            import numpy as _np
            if self.svf_map is None or self.svf_map.ndim != 2:
                return None
            H, W = self.svf_map.shape
            if H < 3 or W < 3:
                return None
            # Define 3x3 boundaries
            hs = [0, H//3, 2*H//3, H]
            ws = [0, W//3, 2*W//3, W]
            names = [["Top-Left", "Top-Center", "Top-Right"],
                     ["Middle-Left", "Middle-Center", "Middle-Right"],
                     ["Bottom-Left", "Bottom-Center", "Bottom-Right"]]
            cells: List[Dict[str, Any]] = []
            for i in range(3):
                for j in range(3):
                    r0, r1 = hs[i], hs[i+1]
                    c0, c1 = ws[j], ws[j+1]
                    tile = self.svf_map[r0:r1, c0:c1]
                    valid = tile[~_np.isnan(tile) & (tile > 0)]
                    mean_svf = float(_np.mean(valid)) if valid.size else 0.0
                    cells.append({
                        "name": names[i][j],
                        "svf_mean": mean_svf,
                        "row": i,
                        "col": j,
                    })
            # Pick top-2 by svf_mean as example optimal locations
            ranked = sorted(cells, key=lambda x: x["svf_mean"], reverse=True)
            optimal = [{"name": r["name"], "svf_mean": r["svf_mean"]} for r in ranked[:2]]
            return {
                "grid": cells,
                "optimal_locations": optimal,
                "note": "SVF-based simple grid summary for wording guidance"
            }
        except Exception:
            return None

    def urban_development_application(self) -> Dict[str, Any]:
        """
        Urban Development/land-use Application Analysis with GPT-4 enhancement
        """
        if self.segmentation_map is None:
            return self._fallback_analysis("urban_development", "segmentation data")
            
        # Calculate development metrics
        analysis = self._calculate_urban_metrics()
        if analysis is None:
            return None  # Skip this question if no valid data
        
        question = """Q: Analyze the potential of this area for urban development. 
Please explain how to make this area better in terms of scenic quality, safety, and human-natural coexistence.

Consider the following factors:
- Sky visibility and openness for human comfort
- Existing land use distribution and compatibility  
- Building density and urban fabric
- Infrastructure accessibility and connectivity
- Environmental sustainability aspects

The total length of the answer should be 30-40 words.

Please structure your response using only <OBSERVATION> and <CONCLUSION>:
<OBSERVATION>Statistical data and image-grounded observations only (SVF values, elevation in meters, land cover %). No interpretation.</OBSERVATION>
<CONCLUSION>Base the decision strictly on <OBSERVATION> and evidence visible in the RGB/SVF/DSM/SEG panels.</CONCLUSION>."""


        # Generate answer using GPT-4 if available, otherwise use fallback
        if self.use_gpt4 and self.gpt4_generator and self._scene_statistics:
            panel_img = self._build_quadrant_visual()
            if self._debug:
                try:
                    import numpy as _np
                    if isinstance(panel_img, _np.ndarray):
                        print(f"[freeform] quadrant panel created: shape={panel_img.shape}, dtype={panel_img.dtype}")
                    else:
                        print("[freeform] quadrant panel not created; falling back to rgb image")
                except Exception:
                    pass
            answer = self.gpt4_generator.generate_answer(
                question=question,
                statistics=self._scene_statistics,
                scene_id=self.file_path or "unknown",
                analysis_type="urban_development",
                image_data=panel_img,
                question_context={'analysis': analysis}
            )
        else:
            answer = self._generate_urban_development_recommendation(analysis)
        return {
            "question": question,
            "answer": answer,
            "analysis_type": "urban_development_application",
            "metrics": analysis,
            "context": ['urban_development', analysis],
            "generation_method": "gpt4" if (self.use_gpt4 and self.gpt4_generator) else "statistical"
        }
    
    def renewable_energy_installation(self) -> Dict[str, Any]:
        """
        Renewable Energy Installation Analysis with GPT-4 enhancement
        """
        if self.svf_map is None:
            return self._fallback_analysis("renewable_energy", "SVF data")
            
        # Calculate energy potential metrics
        analysis = self._calculate_energy_potential()
        if analysis is None:
            return None  # Skip this question if no valid data
        
        question = """Q: Analyze the potential of this area for solar panel and wind power generation installation.

Evaluate:
- Solar irradiance potential based on sky visibility
- Wind exposure and turbulence factors
- Available land area for installation
- Environmental and visual impact considerations
- Grid connectivity and infrastructure requirements

The total length of the answer should be 30-40 words.
Please structure your response using only <OBSERVATION> and <CONCLUSION>:
<OBSERVATION>Statistical data and image-grounded observations only (SVF values, elevation in meters, land cover %). No interpretation.</OBSERVATION>
<CONCLUSION>Base the decision strictly on <OBSERVATION> and evidence visible in the RGB/SVF/DSM/SEG panels.</CONCLUSION>."""


        # Generate answer using GPT-4 if available
        if self.use_gpt4 and self.gpt4_generator and self._scene_statistics:
            panel_img = self._build_quadrant_visual()
            if self._debug:
                try:
                    import numpy as _np
                    if isinstance(panel_img, _np.ndarray):
                        print(f"[freeform] quadrant panel created: shape={panel_img.shape}, dtype={panel_img.dtype}")
                    else:
                        print("[freeform] quadrant panel not created; falling back to rgb image")
                except Exception:
                    pass
            answer = self.gpt4_generator.generate_answer(
                question=question,
                statistics=self._scene_statistics,
                scene_id=self.file_path or "unknown",
                analysis_type="renewable_energy",
                image_data=panel_img,
                question_context={'analysis': analysis}
            )
        else:
            answer = self._generate_energy_installation_recommendation(analysis)
        return {
            "question": question,
            "answer": answer,
            "analysis_type": "renewable_energy_installation", 
            "metrics": analysis,
            "context": ['renewable_energy', analysis],
            "generation_method": "gpt4" if (self.use_gpt4 and self.gpt4_generator) else "statistical"
        }
    
    def landscape_analysis(self) -> Dict[str, Any]:
        """
        Comprehensive Landscape Analysis with GPT-4 enhancement
        """
        analysis = self._calculate_landscape_metrics()
        if analysis is None:
            return None  # Skip this question if no valid data
        
        question = """Q: Analyze overall landscape of this area, in terms of sky visibility, terrain, and landcover types.

Provide assessment on:
- Natural and artificial landscape balance
- Topographical characteristics and terrain variation
- Ecological connectivity and habitat potential  
- Visual landscape quality and scenic value
- Sky openness and spatial characteristics
- Biodiversity and environmental health indicators

The total length of the answer should be 30-40 words.

Please structure your response using only <OBSERVATION> and <CONCLUSION>:
<OBSERVATION>Statistical data and image-grounded observations only (SVF values, elevation in meters, land cover %). No interpretation.</OBSERVATION>
<CONCLUSION>Base the decision strictly on <OBSERVATION> and evidence visible in the RGB/SVF/DSM/SEG panels.</CONCLUSION>."""


        # Generate answer using GPT-4 if available
        if self.use_gpt4 and self.gpt4_generator and self._scene_statistics:
            panel_img = self._build_quadrant_visual()
            if self._debug:
                try:
                    import numpy as _np
                    if isinstance(panel_img, _np.ndarray):
                        print(f"[freeform] quadrant panel created: shape={panel_img.shape}, dtype={panel_img.dtype}")
                    else:
                        print("[freeform] quadrant panel not created; falling back to rgb image")
                except Exception:
                    pass
            answer = self.gpt4_generator.generate_answer(
                question=question,
                statistics=self._scene_statistics,
                scene_id=self.file_path or "unknown",
                analysis_type="landscape_analysis",
                image_data=panel_img,
                question_context={'analysis': analysis}
            )
        else:
            answer = self._generate_landscape_analysis(analysis)
        return {
            "question": question,
            "answer": answer,
            "analysis_type": "landscape_analysis",
            "metrics": analysis,
            "context": ['landscape_analysis', analysis],
            "generation_method": "gpt4" if (self.use_gpt4 and self.gpt4_generator) else "statistical"
        }
    
    def water_accumulation(self) -> Dict[str, Any]:
        """
        Water Accumulation Analysis with GPT-4 enhancement
        """
        if self.height_map is None:
            return self._fallback_analysis("water_accumulation", "height data")
            
        analysis = self._calculate_water_accumulation()
        if analysis is None:
            return None  # Skip this question if no valid data
        
        question = """Q: Which place will mostly accumulate water?

Analyze:
- Topographical flow patterns and natural drainage
- Low-lying areas most susceptible to water collection
- Surface permeability and infiltration capacity
- Potential flood risks and water management needs
- Drainage infrastructure requirements

The total length of the answer should be 30-40 words.

Please structure your response using only <OBSERVATION> and <CONCLUSION>:
<OBSERVATION>Statistical data and image-grounded observations only (SVF values, elevation in meters, land cover %). No interpretation.</OBSERVATION>
<CONCLUSION>Base the decision strictly on <OBSERVATION> and evidence visible in the RGB/SVF/DSM/SEG panels.</CONCLUSION>."""


        # Generate answer using GPT-4 if available
        if self.use_gpt4 and self.gpt4_generator and self._scene_statistics:
            panel_img = self._build_quadrant_visual()
            if self._debug:
                try:
                    import numpy as _np
                    if isinstance(panel_img, _np.ndarray):
                        print(f"[freeform] quadrant panel created: shape={panel_img.shape}, dtype={panel_img.dtype}")
                    else:
                        print("[freeform] quadrant panel not created; falling back to rgb image")
                except Exception:
                    pass
            answer = self.gpt4_generator.generate_answer(
                question=question,
                statistics=self._scene_statistics,
                scene_id=self.file_path or "unknown",
                analysis_type="water_accumulation",
                image_data=panel_img,
                question_context={'analysis': analysis}
            )
        else:
            answer = self._generate_water_accumulation_analysis(analysis)
        return {
            "question": question,
            "answer": answer,
            "analysis_type": "water_accumulation",
            "metrics": analysis,
            "context": ['water_accumulation', analysis],
            "generation_method": "gpt4" if (self.use_gpt4 and self.gpt4_generator) else "statistical"
        }
    
    def _calculate_urban_metrics(self) -> Dict:
        """Calculate urban development metrics with enhanced SVF-TSM-Segmentation integration."""
        metrics = {}
        
        # Enhanced SVF analysis for human comfort and spatial quality
        if self.svf_map is not None:
            valid_svf = self.svf_map[~np.isnan(self.svf_map) & (self.svf_map > 0)]
            
            if len(valid_svf) > 0:
                # Basic SVF statistics
                metrics['sky_visibility'] = {
                    'mean': float(np.mean(valid_svf)),
                    'std': float(np.std(valid_svf)),
                    'min': float(np.min(valid_svf)),
                    'max': float(np.max(valid_svf)),
                    'comfort_ratio': float(np.sum((valid_svf > 0.3) & (valid_svf < 0.8)) / len(valid_svf))
                }
                
                # SVF-based spatial quality metrics
                metrics['spatial_quality'] = {
                    'openness_score': float(np.mean(valid_svf)),
                    'spatial_diversity': float(np.std(valid_svf)),
                    'enclosed_areas_ratio': float(np.sum(valid_svf < 0.3) / len(valid_svf)),
                    'very_open_areas_ratio': float(np.sum(valid_svf > 0.8) / len(valid_svf))
                }
            else:
                # Return None to indicate no valid data - this will cause QA generation to be skipped
                return None
            
            # Sky visibility distribution analysis
            if len(valid_svf) > 0:
                svf_quartiles = np.percentile(valid_svf, [25, 50, 75])
                metrics['svf_distribution'] = {
                    'q1': float(svf_quartiles[0]),
                    'median': float(svf_quartiles[1]),
                    'q3': float(svf_quartiles[2]),
                    'iqr': float(svf_quartiles[2] - svf_quartiles[0])
                }
            else:
                return None
        
        # Enhanced land use analysis with SVF integration
        if self.segmentation_map is not None:
            unique_classes, counts = np.unique(self.segmentation_map, return_counts=True)
            total_pixels = self.segmentation_map.size
            
            metrics['land_use'] = {}
            for cls, count in zip(unique_classes, counts):
                if cls in self.LANDCOVER_CLASSES:
                    percentage = (count / total_pixels) * 100
                    metrics['land_use'][self.LANDCOVER_CLASSES[cls]] = float(percentage)
            
            # Development suitability with SVF consideration
            developed_classes = [4, 6, 7, 8, 10]  # residential, railways, roads, commercial, buildings
            developed_pixels = sum(counts[np.isin(unique_classes, developed_classes)])
            metrics['development_ratio'] = float(developed_pixels / total_pixels)
            
            # Natural-artificial balance for sustainability assessment
            natural_classes = [1, 2, 3, 5]  # forest, water, agricultural, grassland
            natural_pixels = sum(counts[np.isin(unique_classes, natural_classes)])
            metrics['sustainability_balance'] = {
                'natural_ratio': float(natural_pixels / total_pixels),
                'developed_ratio': float(developed_pixels / total_pixels),
                'balance_score': float(min(natural_pixels, developed_pixels) / max(natural_pixels, developed_pixels) if max(natural_pixels, developed_pixels) > 0 else 0)
            }
        
        # Enhanced terrain analysis with SVF correlation
        if self.height_map is not None:
            valid_height = self.height_map[~np.isnan(self.height_map)]
            if len(valid_height) == 0:
                return None
            
            metrics['terrain'] = {
                'mean_height': float(np.mean(valid_height)),
                'height_std': float(np.std(valid_height)),
                'elevation_range': float(np.max(valid_height) - np.min(valid_height)),
                'terrain_roughness': float(np.std(np.gradient(valid_height))),
                'slope_variation': float(np.std(np.gradient(valid_height)))
            }
            
            # Height-based development potential
            height_percentiles = np.percentile(valid_height, [25, 75])
            metrics['height_suitability'] = {
                'suitable_terrain_ratio': float(np.sum((valid_height >= height_percentiles[0]) & (valid_height <= height_percentiles[1])) / len(valid_height)),
                'extreme_terrain_ratio': float(np.sum((valid_height < height_percentiles[0]) | (valid_height > height_percentiles[1])) / len(valid_height))
            }
        
        # SVF-Terrain correlation analysis
        if self.svf_map is not None and self.height_map is not None:
            valid_mask = ~np.isnan(self.svf_map) & ~np.isnan(self.height_map) & (self.svf_map > 0)
            if np.sum(valid_mask) > 10:
                svf_valid = self.svf_map[valid_mask]
                height_valid = self.height_map[valid_mask]
                correlation = np.corrcoef(svf_valid, height_valid)[0, 1]
                metrics['svf_terrain_correlation'] = {
                    'correlation': float(correlation if not np.isnan(correlation) else 0),
                    'relationship_strength': 'strong' if abs(correlation) > 0.5 else 'moderate' if abs(correlation) > 0.3 else 'weak'
                }
        
        # Integrated urban density score (BCR + FAR + SVF impact)
        if self.segmentation_map is not None and self.height_map is not None and self.svf_map is not None:
            # Building Coverage Ratio
            building_pixels = np.sum(np.isin(self.segmentation_map, [4, 10]))
            bcr = building_pixels / self.segmentation_map.size
            
            # Floor Area Ratio approximation
            building_mask = np.isin(self.segmentation_map, [4, 10])
            if np.sum(building_mask) > 0:
                avg_building_height = np.mean(self.height_map[building_mask & ~np.isnan(self.height_map)])
                far = bcr * (avg_building_height / 3.0)  # Assuming 3m per floor
            else:
                far = 0
            
            # SVF impact on density perception
            valid_svf = self.svf_map[~np.isnan(self.svf_map) & (self.svf_map > 0)]
            svf_impact = 1 - np.mean(valid_svf)  # Higher density -> lower SVF
            
            metrics['integrated_density'] = {
                'bcr': float(bcr),
                'far': float(far),
                'svf_impact': float(svf_impact),
                'integrated_score': float((bcr * 0.4 + far * 0.4 + svf_impact * 0.2))
            }
            
        return metrics
    
    def _calculate_energy_potential(self) -> Dict:
        """Calculate renewable energy potential metrics with enhanced precision."""
        metrics = {}
        
        # Enhanced solar potential from SVF with terrain consideration
        if self.svf_map is not None:
            valid_svf = self.svf_map[~np.isnan(self.svf_map) & (self.svf_map > 0)]
            if len(valid_svf) == 0:
                return None
            
            # Basic solar access metrics
            high_solar = valid_svf > 0.7
            excellent_solar = valid_svf > 0.8
            moderate_solar = (valid_svf > 0.5) & (valid_svf <= 0.7)
            
            metrics['solar_potential'] = {
                'mean_svf': float(np.mean(valid_svf)),
                'high_solar_ratio': float(np.sum(high_solar) / len(valid_svf)),
                'excellent_solar_ratio': float(np.sum(excellent_solar) / len(valid_svf)),
                'moderate_solar_ratio': float(np.sum(moderate_solar) / len(valid_svf)),
                'optimal_areas_percentage': float(np.sum(valid_svf > 0.8) / len(valid_svf) * 100)
            }
            
            # Solar irradiance estimation based on SVF
            clear_sky_irradiance = 1000  # W/m² typical clear sky
            estimated_irradiance = valid_svf * clear_sky_irradiance
            
            metrics['solar_irradiance'] = {
                'mean_irradiance': float(np.mean(estimated_irradiance)),
                'max_irradiance': float(np.max(estimated_irradiance)),
                'irradiance_std': float(np.std(estimated_irradiance)),
                'high_irradiance_ratio': float(np.sum(estimated_irradiance > 700) / len(estimated_irradiance))
            }
        
        # Wind potential analysis with terrain consideration
        if self.height_map is not None and self.svf_map is not None:
            valid_height = self.height_map[~np.isnan(self.height_map)]
            
            # Wind exposure analysis
            elevation_gradient = np.gradient(self.height_map)
            terrain_roughness = np.std(elevation_gradient)
            
            # Higher elevations and open areas (high SVF) generally have better wind exposure
            valid_mask = ~np.isnan(self.svf_map) & ~np.isnan(self.height_map) & (self.svf_map > 0)
            if np.sum(valid_mask) > 0:
                svf_valid = self.svf_map[valid_mask]
                height_valid = self.height_map[valid_mask]
                
                # Wind exposure score combining height and openness
                wind_exposure = (height_valid - np.min(height_valid)) / (np.max(height_valid) - np.min(height_valid)) * 0.6 + svf_valid * 0.4
                
                metrics['wind_potential'] = {
                    'mean_wind_exposure': float(np.mean(wind_exposure)),
                    'terrain_roughness': float(terrain_roughness),
                    'high_wind_ratio': float(np.sum(wind_exposure > 0.7) / len(wind_exposure)),
                    'optimal_wind_areas': float(np.sum(wind_exposure > 0.8) / len(wind_exposure) * 100)
                }
        
        # Land suitability for installations
        if self.segmentation_map is not None and self.svf_map is not None:
            # Suitable land classes for energy installations
            suitable_classes = [3, 5]  # agricultural, grassland
            unsuitable_classes = [1, 2, 4, 6, 7, 8, 9, 10]  # forest, water, residential, infrastructure, buildings
            
            total_pixels = self.segmentation_map.size
            suitable_pixels = np.sum(np.isin(self.segmentation_map, suitable_classes))
            unsuitable_pixels = np.sum(np.isin(self.segmentation_map, unsuitable_classes))
            
            # Calculate SVF in suitable areas
            suitable_mask = np.isin(self.segmentation_map, suitable_classes)
            valid_svf_suitable = self.svf_map[suitable_mask & ~np.isnan(self.svf_map) & (self.svf_map > 0)]
            
            if len(valid_svf_suitable) > 0:
                metrics['land_suitability'] = {
                    'suitable_land_ratio': float(suitable_pixels / total_pixels),
                    'unsuitable_land_ratio': float(unsuitable_pixels / total_pixels),
                    'suitable_area_svf': float(np.mean(valid_svf_suitable)),
                    'high_potential_suitable_ratio': float(np.sum(valid_svf_suitable > 0.7) / len(valid_svf_suitable))
                }
        
        # Grid connectivity assessment
        if self.segmentation_map is not None:
            # Proximity to infrastructure (roads, commercial, residential)
            infrastructure_classes = [4, 6, 7, 8]  # residential, railways, roads, commercial
            infrastructure_pixels = np.sum(np.isin(self.segmentation_map, infrastructure_classes))
            
            metrics['grid_connectivity'] = {
                'infrastructure_ratio': float(infrastructure_pixels / self.segmentation_map.size),
                'connectivity_score': float(min(infrastructure_pixels / self.segmentation_map.size * 10, 1.0))  # Normalized to 1.0
            }
        
        # Integrated renewable energy score
        if 'solar_potential' in metrics and 'land_suitability' in metrics:
            solar_score = metrics['solar_potential']['mean_svf'] * 0.4
            land_score = metrics['land_suitability']['suitable_land_ratio'] * 0.3
            wind_score = metrics.get('wind_potential', {}).get('mean_wind_exposure', 0) * 0.2
            connectivity_score = metrics.get('grid_connectivity', {}).get('connectivity_score', 0) * 0.1
            
            metrics['integrated_energy_potential'] = {
                'solar_component': float(solar_score),
                'land_component': float(land_score),
                'wind_component': float(wind_score),
                'connectivity_component': float(connectivity_score),
                'total_score': float(solar_score + land_score + wind_score + connectivity_score)
            }
        
        return metrics
    
    def _calculate_landscape_metrics(self) -> Dict:
        """Calculate comprehensive landscape metrics with multimodal consistency evaluation."""
        metrics = {}
        
        # Enhanced sky visibility analysis
        if self.svf_map is not None:
            valid_svf = self.svf_map[~np.isnan(self.svf_map) & (self.svf_map > 0)]
            if len(valid_svf) == 0:
                return None
            
            # Advanced SVF statistics
            svf_percentiles = np.percentile(valid_svf, [10, 25, 50, 75, 90])
            metrics['sky_visibility'] = {
                'mean': float(np.mean(valid_svf)),
                'std': float(np.std(valid_svf)),
                'min': float(np.min(valid_svf)),
                'max': float(np.max(valid_svf)),
                'spatial_openness': float(np.sum(valid_svf > 0.5) / len(valid_svf)),
                'very_open_ratio': float(np.sum(valid_svf > 0.8) / len(valid_svf)),
                'enclosed_ratio': float(np.sum(valid_svf < 0.3) / len(valid_svf)),
                'percentiles': {
                    'p10': float(svf_percentiles[0]),
                    'p25': float(svf_percentiles[1]),
                    'p50': float(svf_percentiles[2]),
                    'p75': float(svf_percentiles[3]),
                    'p90': float(svf_percentiles[4])
                }
            }
        
        # Enhanced terrain analysis
        if self.height_map is not None:
            valid_height = self.height_map[~np.isnan(self.height_map)]
            if len(valid_height) == 0:
                return None
            
            # Terrain complexity metrics
            height_gradient = np.gradient(self.height_map)
            terrain_complexity = np.std(height_gradient)
            
            # Elevation zones analysis
            height_percentiles = np.percentile(valid_height, [25, 50, 75])
            low_elevation = np.sum(valid_height < height_percentiles[0])
            mid_elevation = np.sum((valid_height >= height_percentiles[0]) & (valid_height < height_percentiles[2]))
            high_elevation = np.sum(valid_height >= height_percentiles[2])
            
            metrics['terrain'] = {
                'elevation_range': float(np.max(valid_height) - np.min(valid_height)),
                'mean_elevation': float(np.mean(valid_height)),
                'elevation_std': float(np.std(valid_height)),
                'terrain_roughness': float(np.std(valid_height)),
                'terrain_complexity': float(terrain_complexity),
                'elevation_zones': {
                    'low_ratio': float(low_elevation / len(valid_height)),
                    'mid_ratio': float(mid_elevation / len(valid_height)),
                    'high_ratio': float(high_elevation / len(valid_height))
                }
            }
        
        # Enhanced land cover diversity analysis
        if self.segmentation_map is not None:
            unique_classes, counts = np.unique(self.segmentation_map, return_counts=True)
            total_pixels = self.segmentation_map.size
            
            metrics['landcover'] = {}
            for cls, count in zip(unique_classes, counts):
                if cls in self.LANDCOVER_CLASSES:
                    percentage = (count / total_pixels) * 100
                    metrics['landcover'][self.LANDCOVER_CLASSES[cls]] = float(percentage)
            
            # Enhanced diversity indices
            proportions = counts / total_pixels
            simpson_diversity = 1 - np.sum(proportions ** 2)
            shannon_diversity = -np.sum(proportions * np.log(proportions + 1e-10))
            
            metrics['landcover_diversity'] = {
                'simpson_index': float(simpson_diversity),
                'shannon_index': float(shannon_diversity),
                'richness': float(len(unique_classes))
            }
            
            # Natural vs artificial balance with ecological connectivity
            natural_classes = [1, 2, 3, 5]  # forest, water, agricultural, grassland
            artificial_classes = [4, 6, 7, 8, 9,10]  # residential, railways, roads, commercial, buildings
            
            natural_pixels = sum(counts[np.isin(unique_classes, natural_classes)])
            artificial_pixels = sum(counts[np.isin(unique_classes, artificial_classes)])
            
            # Ecological connectivity assessment
            water_pixels = sum(counts[np.isin(unique_classes, [2])])  # water bodies
            forest_pixels = sum(counts[np.isin(unique_classes, [1])])  # forest
            
            metrics['natural_artificial_balance'] = {
                'natural_ratio': float(natural_pixels / total_pixels),
                'artificial_ratio': float(artificial_pixels / total_pixels),
                'balance_score': float(min(natural_pixels, artificial_pixels) / max(natural_pixels, artificial_pixels) if max(natural_pixels, artificial_pixels) > 0 else 0),
                'water_connectivity': float(water_pixels / total_pixels),
                'forest_connectivity': float(forest_pixels / total_pixels)
            }
        
        # Multimodal consistency evaluation
        if self.svf_map is not None and self.height_map is not None and self.segmentation_map is not None:
            consistency_metrics = self._calculate_multimodal_consistency()
            metrics['multimodal_consistency'] = consistency_metrics
        
        # Visual landscape quality assessment
        if self.rgb_image is not None:
            visual_metrics = self._calculate_visual_quality()
            metrics['visual_quality'] = visual_metrics
        
        # Integrated landscape quality score
        if 'sky_visibility' in metrics and 'terrain' in metrics and 'landcover_diversity' in metrics:
            # Weighted combination of landscape quality factors
            openness_score = metrics['sky_visibility']['spatial_openness'] * 0.25
            diversity_score = min(metrics['landcover_diversity']['shannon_index'] / 2.0, 1.0) * 0.25
            terrain_score = min(metrics['terrain']['elevation_range'] / 100.0, 1.0) * 0.20
            natural_score = metrics['natural_artificial_balance']['natural_ratio'] * 0.20
            balance_score = metrics['natural_artificial_balance']['balance_score'] * 0.10
            
            metrics['integrated_landscape_quality'] = {
                'openness_component': float(openness_score),
                'diversity_component': float(diversity_score),
                'terrain_component': float(terrain_score),
                'natural_component': float(natural_score),
                'balance_component': float(balance_score),
                'total_score': float(openness_score + diversity_score + terrain_score + natural_score + balance_score)
            }
        
        return metrics
    
    def _calculate_multimodal_consistency(self) -> Dict:
        """Calculate consistency between SVF, height, and segmentation data."""
        consistency = {}
        
        # SVF-Height consistency
        if self.svf_map is not None and self.height_map is not None:
            valid_mask = ~np.isnan(self.svf_map) & ~np.isnan(self.height_map) & (self.svf_map > 0)
            if np.sum(valid_mask) > 10:
                svf_valid = self.svf_map[valid_mask]
                height_valid = self.height_map[valid_mask]
                
                # Higher buildings should generally have lower SVF
                correlation = np.corrcoef(svf_valid, height_valid)[0, 1]
                expected_negative = correlation < 0
                
                consistency['svf_height'] = {
                    'correlation': float(correlation if not np.isnan(correlation) else 0),
                    'expected_relationship': bool(expected_negative),
                    'consistency_score': float(abs(correlation) if expected_negative else 0)
                }
        
        # SVF-Segmentation consistency
        if self.svf_map is not None and self.segmentation_map is not None:
            building_mask = np.isin(self.segmentation_map, [4, 10])  # residential, buildings
            open_mask = np.isin(self.segmentation_map, [2, 3, 5])  # water, agricultural, grassland
            
            if np.sum(building_mask) > 0 and np.sum(open_mask) > 0:
                svf_buildings = self.svf_map[building_mask & ~np.isnan(self.svf_map)]
                svf_open = self.svf_map[open_mask & ~np.isnan(self.svf_map)]
                
                if len(svf_buildings) > 0 and len(svf_open) > 0:
                    mean_svf_buildings = np.mean(svf_buildings)
                    mean_svf_open = np.mean(svf_open)
                    
                    # Open areas should have higher SVF than built areas
                    expected_relationship = mean_svf_open > mean_svf_buildings
                    svf_difference = mean_svf_open - mean_svf_buildings
                    
                    consistency['svf_segmentation'] = {
                        'mean_svf_buildings': float(mean_svf_buildings),
                        'mean_svf_open': float(mean_svf_open),
                        'svf_difference': float(svf_difference),
                        'expected_relationship': bool(expected_relationship),
                        'consistency_score': float(svf_difference if expected_relationship else 0)
                    }
        
        # Height-Segmentation consistency
        if self.height_map is not None and self.segmentation_map is not None:
            building_mask = np.isin(self.segmentation_map, [4, 10])  # residential, buildings
            natural_mask = np.isin(self.segmentation_map, [1, 2, 3, 5])  # forest, water, agricultural, grassland
            
            if np.sum(building_mask) > 0 and np.sum(natural_mask) > 0:
                height_buildings = self.height_map[building_mask & ~np.isnan(self.height_map)]
                height_natural = self.height_map[natural_mask & ~np.isnan(self.height_map)]
                
                if len(height_buildings) > 0 and len(height_natural) > 0:
                    mean_height_buildings = np.mean(height_buildings)
                    mean_height_natural = np.mean(height_natural)
                    
                    # Buildings should generally be taller than natural areas
                    expected_relationship = mean_height_buildings > mean_height_natural
                    height_difference = mean_height_buildings - mean_height_natural
                    
                    consistency['height_segmentation'] = {
                        'mean_height_buildings': float(mean_height_buildings),
                        'mean_height_natural': float(mean_height_natural),
                        'height_difference': float(height_difference),
                        'expected_relationship': bool(expected_relationship),
                        'consistency_score': float(height_difference if expected_relationship else 0)
                    }
        
        return consistency
    
    def _calculate_visual_quality(self) -> Dict:
        """Calculate visual quality metrics from RGB image."""
        visual = {}
        
        if self.rgb_image is not None and self.rgb_image.ndim == 3 and self.rgb_image.shape[2] >= 3:
            # Color harmony analysis
            rgb_mean = np.mean(self.rgb_image, axis=(0, 1))
            rgb_std = np.std(self.rgb_image, axis=(0, 1))
            
            # Green dominance (indicator of natural elements)
            green_dominance = rgb_mean[1] / (np.sum(rgb_mean) + 1e-6)
            
            # Color variance (indicator of visual complexity)
            color_variance = np.mean(rgb_std)
            
            # Brightness distribution
            brightness = np.mean(self.rgb_image, axis=2)
            brightness_hist, _ = np.histogram(brightness, bins=10)
            brightness_diversity = 1 - np.sum((brightness_hist / np.sum(brightness_hist)) ** 2)
            
            visual = {
                'green_dominance': float(green_dominance),
                'color_variance': float(color_variance),
                'brightness_diversity': float(brightness_diversity),
                'mean_brightness': float(np.mean(brightness)),
                'rgb_balance': {
                    'red': float(rgb_mean[0]),
                    'green': float(rgb_mean[1]),
                    'blue': float(rgb_mean[2])
                }
            }
        
        return visual

    def _build_quadrant_visual(self) -> Optional[np.ndarray]:
        """Compose a 2x2 panel image (RGB, SVF, DSM, SEG) for multimodal GPT input.
        Returns an RGB numpy array (H*2, W*2, 3) or None if inputs are missing.
        """
        try:
            import numpy as _np
            if self.svf_map is None or self.height_map is None or self.segmentation_map is None:
                # Fallback to original RGB if available
                return self.rgb_image if isinstance(self.rgb_image, _np.ndarray) else None
            # Normalize helpers
            def _norm01(arr):
                a = arr.astype(float)
                a = a[~_np.isnan(a)] if _np.isnan(arr).any() else a
                amin = float(_np.min(a)) if a.size else 0.0
                amax = float(_np.max(a)) if a.size else 1.0
                if amax - amin < 1e-6:
                    return _np.zeros_like(arr, dtype=_np.float32)
                out = (arr - amin) / (amax - amin)
                out = _np.clip(out, 0, 1)
                return out
            def _to_rgb_from_gray(arr, cmap):
                from matplotlib import cm
                colored = cm.get_cmap(cmap)(_np.nan_to_num(arr, nan=0.0))[:, :, :3]
                return (colored * 255).astype(_np.uint8)
            # Prepare each panel
            rgb = self.rgb_image
            if rgb is None or rgb.ndim != 3 or rgb.shape[2] < 3:
                # synthesize RGB from SVF as fallback
                rgb = (_norm01(self.svf_map) * 255).astype(_np.uint8)
                rgb = _np.stack([rgb, rgb, rgb], axis=-1)
            svf_rgb = _to_rgb_from_gray(_norm01(self.svf_map), 'viridis')
            dsm_rgb = _to_rgb_from_gray(_norm01(self.height_map), 'terrain')
            # Simple palette for segmentation
            seg = self.segmentation_map.astype(_np.int32)
            unique_vals = _np.unique(seg)
            palette = _np.array([
                [0,0,0],[0,100,0],[0,0,255],[255,255,0],[128,128,128],
                [144,238,144],[165,42,42],[192,192,192],[255,165,0],[128,128,0],[255,0,0]
            ], dtype=_np.uint8)
            max_idx = int(unique_vals.max()) if unique_vals.size else 10
            lut = _np.zeros((max(11, max_idx+1), 3), dtype=_np.uint8)
            lut[:palette.shape[0], :] = palette
            seg_rgb = lut[_np.clip(seg, 0, lut.shape[0]-1)]
            # Resize panels to match
            H = min(x.shape[0] for x in [rgb, svf_rgb, dsm_rgb, seg_rgb])
            W = min(x.shape[1] for x in [rgb, svf_rgb, dsm_rgb, seg_rgb])
            rgb = rgb[:H, :W, :3]
            svf_rgb = svf_rgb[:H, :W, :3]
            dsm_rgb = dsm_rgb[:H, :W, :3]
            seg_rgb = seg_rgb[:H, :W, :3]
            # Compose 2x2 grid
            top = _np.concatenate([rgb, svf_rgb], axis=1)
            bottom = _np.concatenate([dsm_rgb, seg_rgb], axis=1)
            panel = _np.concatenate([top, bottom], axis=0)
            return panel
        except Exception:
            return self.rgb_image

    def _generate_urban_development_recommendation(self, metrics: Dict) -> str:
        """Generate urban development recommendation based on metrics."""
        recommendations = []
        
        # Sky visibility assessment
        if 'sky_visibility' in metrics:
            comfort_ratio = metrics['sky_visibility']['comfort_ratio']
            if comfort_ratio > 0.6:
                recommendations.append("The area has good sky visibility with comfortable openness levels for human habitation.")
            else:
                recommendations.append("Sky visibility could be improved through strategic building placement and height restrictions.")
        
        if not recommendations:
            recommendations.append("Limited data available for comprehensive urban development analysis.")
        
        return " ".join(recommendations)
    
    def _generate_energy_installation_recommendation(self, metrics: Dict) -> str:
        """Generate renewable energy installation recommendation."""
        recommendations = []
        
        # Solar potential
        if 'solar_potential' in metrics:
            high_solar_ratio = metrics['solar_potential']['high_solar_ratio']
            if high_solar_ratio > 0.4:
                recommendations.append("Excellent solar potential with high sky visibility. Recommend solar panel installation.")
            else:
                recommendations.append("Moderate solar potential. Consider selective solar installation in optimal areas.")
        
        if not recommendations:
            recommendations.append("Renewable energy potential assessment requires additional data.")
        
        return " ".join(recommendations)
    
    def _generate_landscape_analysis(self, metrics: Dict) -> str:
        """Generate comprehensive landscape analysis."""
        analysis_points = []
        
        # Sky characteristics
        if 'sky_visibility' in metrics:
            openness = metrics['sky_visibility']['mean']
            if openness > 0.6:
                analysis_points.append("High sky openness creates a sense of spatial freedom and visual comfort.")
            else:
                analysis_points.append("Moderate sky visibility with enclosed spatial character.")
        
        if not analysis_points:
            analysis_points.append("Landscape analysis requires comprehensive multi-modal data.")
        
        return " ".join(analysis_points)
    
    def _generate_water_accumulation_analysis(self, metrics: Dict) -> str:
        """Generate water accumulation analysis."""
        analysis_points = []
        analysis_points.append("The open-space square faces the most risks because there has the lowest terrain and the ground is cement that prevents water penetration.")
        
        return " ".join(analysis_points)
    
    def _calculate_spatial_integration_metrics(self) -> Dict:
        """Calculate spatial integration metrics combining SVF, terrain, and segmentation."""
        metrics = {}
        
        if self.svf_map is not None and self.height_map is not None and self.segmentation_map is not None:
            # Spatial heterogeneity analysis
            valid_mask = ~np.isnan(self.svf_map) & ~np.isnan(self.height_map) & (self.svf_map > 0)
            
            if np.sum(valid_mask) > 100:  # Sufficient data for analysis
                svf_valid = self.svf_map[valid_mask]
                height_valid = self.height_map[valid_mask]
                seg_valid = self.segmentation_map[valid_mask]
                
                # Calculate spatial autocorrelation (Moran's I approximation)
                # This measures how similar nearby values are
                svf_variance = np.var(svf_valid)
                height_variance = np.var(height_valid)
                
                metrics['spatial_heterogeneity'] = {
                    'svf_variance': float(svf_variance),
                    'height_variance': float(height_variance),
                    'combined_variance': float(svf_variance + height_variance)
                }
                
                # Edge effect analysis
                # Calculate how much the landscape changes at boundaries
                h, w = self.svf_map.shape
                edge_pixels = set()
                
                # Add border pixels
                for i in range(h):
                    edge_pixels.add((i, 0))
                    edge_pixels.add((i, w-1))
                for j in range(w):
                    edge_pixels.add((0, j))
                    edge_pixels.add((h-1, j))
                
                # Calculate edge characteristics
                edge_svf_values = []
                edge_height_values = []
                
                for i, j in edge_pixels:
                    if not np.isnan(self.svf_map[i, j]) and self.svf_map[i, j] > 0:
                        edge_svf_values.append(self.svf_map[i, j])
                    if not np.isnan(self.height_map[i, j]):
                        edge_height_values.append(self.height_map[i, j])
                
                if edge_svf_values and edge_height_values:
                    center_svf = np.mean(svf_valid)
                    center_height = np.mean(height_valid)
                    
                    edge_svf_mean = np.mean(edge_svf_values)
                    edge_height_mean = np.mean(edge_height_values)
                    
                    metrics['edge_effects'] = {
                        'edge_svf_mean': float(edge_svf_mean),
                        'center_svf_mean': float(center_svf),
                        'svf_edge_center_diff': float(edge_svf_mean - center_svf),
                        'edge_height_mean': float(edge_height_mean),
                        'center_height_mean': float(center_height),
                        'height_edge_center_diff': float(edge_height_mean - center_height)
                    }
                
                # Land use transition analysis
                unique_classes = np.unique(seg_valid)
                if len(unique_classes) > 1:
                    # Calculate transition complexity
                    transition_count = 0
                    total_neighbors = 0
                    
                    for i in range(1, h-1):
                        for j in range(1, w-1):
                            if valid_mask[i, j]:
                                current_class = self.segmentation_map[i, j]
                                neighbors = [
                                    self.segmentation_map[i-1, j],
                                    self.segmentation_map[i+1, j],
                                    self.segmentation_map[i, j-1],
                                    self.segmentation_map[i, j+1]
                                ]
                                
                                for neighbor in neighbors:
                                    total_neighbors += 1
                                    if neighbor != current_class:
                                        transition_count += 1
                    
                    transition_ratio = transition_count / max(total_neighbors, 1)
                    
                    metrics['land_use_transitions'] = {
                        'transition_ratio': float(transition_ratio),
                        'landscape_complexity': float(transition_ratio * len(unique_classes))
                    }
        
        return metrics
    
    def _calculate_sustainability_balance(self) -> Dict:
        """Calculate sustainability balance metrics."""
        metrics = {}
        
        if self.svf_map is not None and self.segmentation_map is not None:
            # Environmental sustainability indicators
            natural_classes = [1, 2, 3, 5]  # forest, water, agricultural, grassland
            artificial_classes = [4, 6, 7, 8, 9, 10]  # residential, railways, roads, commercial, buildings
            
            total_pixels = self.segmentation_map.size
            natural_pixels = np.sum(np.isin(self.segmentation_map, natural_classes))
            artificial_pixels = np.sum(np.isin(self.segmentation_map, artificial_classes))
            
            # Natural coverage in high SVF areas
            high_svf_mask = self.svf_map > 0.6
            natural_high_svf = np.sum(np.isin(self.segmentation_map, natural_classes) & high_svf_mask)
            
            # Artificial coverage in low SVF areas
            low_svf_mask = self.svf_map < 0.4
            artificial_low_svf = np.sum(np.isin(self.segmentation_map, artificial_classes) & low_svf_mask)
            
            # Carbon sequestration potential (forest areas)
            forest_pixels = np.sum(np.isin(self.segmentation_map, [1]))
            carbon_sequestration_potential = forest_pixels / total_pixels
            
            # Biodiversity support potential
            water_pixels = np.sum(np.isin(self.segmentation_map, [2]))
            biodiversity_support = (forest_pixels + water_pixels) / total_pixels
            
            metrics['environmental_sustainability'] = {
                'natural_coverage_ratio': float(natural_pixels / total_pixels),
                'artificial_coverage_ratio': float(artificial_pixels / total_pixels),
                'natural_high_svf_ratio': float(natural_high_svf / max(natural_pixels, 1)),
                'artificial_low_svf_ratio': float(artificial_low_svf / max(artificial_pixels, 1)),
                'carbon_sequestration_potential': float(carbon_sequestration_potential),
                'biodiversity_support_potential': float(biodiversity_support)
            }
            
            # Sustainability score (0-1)
            sustainability_score = (
                (natural_pixels / total_pixels) * 0.3 +
                (natural_high_svf / max(natural_pixels, 1)) * 0.2 +
                carbon_sequestration_potential * 0.2 +
                biodiversity_support * 0.2 +
                (1 - artificial_low_svf / max(artificial_pixels, 1)) * 0.1
            )
            
            metrics['sustainability_score'] = float(sustainability_score)
        
        return metrics
    
    def _calculate_water_accumulation(self) -> Dict:
        """Calculate water accumulation potential metrics with enhanced terrain flow analysis."""
        metrics = {}
        
        if self.height_map is not None:
            valid_height = self.height_map[~np.isnan(self.height_map)]
            
            # Basic elevation statistics
            height_percentiles = np.percentile(valid_height, [10, 20, 50, 80, 90])
            low_areas_mask = self.height_map <= height_percentiles[1]  # Bottom 20%
            very_low_areas_mask = self.height_map <= height_percentiles[0]  # Bottom 10%
            
            # Terrain flow analysis using gradient
            if self.height_map.shape[0] > 1 and self.height_map.shape[1] > 1:
                gradient_y, gradient_x = np.gradient(self.height_map)
                slope_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
                
                # Flow accumulation zones (low slope areas)
                flat_areas_mask = slope_magnitude < np.percentile(slope_magnitude[~np.isnan(slope_magnitude)], 25)
                steep_areas_mask = slope_magnitude > np.percentile(slope_magnitude[~np.isnan(slope_magnitude)], 75)
                
                # Potential depression areas (low elevation + low slope)
                depression_areas = low_areas_mask & flat_areas_mask
                
                metrics['terrain_flow'] = {
                    'mean_slope': float(np.mean(slope_magnitude[~np.isnan(slope_magnitude)])),
                    'flat_areas_ratio': float(np.sum(flat_areas_mask) / self.height_map.size),
                    'steep_areas_ratio': float(np.sum(steep_areas_mask) / self.height_map.size),
                    'depression_areas_ratio': float(np.sum(depression_areas) / self.height_map.size)
                }
            
            metrics['water_accumulation'] = {
                'low_lying_percentage': float(np.sum(low_areas_mask) / self.height_map.size * 100),
                'very_low_lying_percentage': float(np.sum(very_low_areas_mask) / self.height_map.size * 100),
                'elevation_range': float(np.max(valid_height) - np.min(valid_height)),
                'lowest_elevation': float(np.min(valid_height)),
                'elevation_std': float(np.std(valid_height)),
                'height_percentiles': {
                    'p10': float(height_percentiles[0]),
                    'p20': float(height_percentiles[1]),
                    'p50': float(height_percentiles[2]),
                    'p80': float(height_percentiles[3]),
                    'p90': float(height_percentiles[4])
                }
            }
        
        # Surface permeability analysis based on land cover
        if self.segmentation_map is not None:
            total_pixels = self.segmentation_map.size
            
            # Permeable surfaces (allow water infiltration)
            permeable_classes = [1, 3, 5]  # forest, agricultural, grassland
            permeable_pixels = np.sum(np.isin(self.segmentation_map, permeable_classes))
            
            # Impermeable surfaces (prevent water infiltration)
            impermeable_classes = [4, 6, 7, 8, 9, 10]  # residential, railways, roads, commercial, buildings
            impermeable_pixels = np.sum(np.isin(self.segmentation_map, impermeable_classes))
            
            # Water bodies (existing water)
            water_pixels = np.sum(np.isin(self.segmentation_map, [2]))
            
            metrics['surface_permeability'] = {
                'permeable_ratio': float(permeable_pixels / total_pixels),
                'impermeable_ratio': float(impermeable_pixels / total_pixels),
                'water_body_ratio': float(water_pixels / total_pixels),
                'infiltration_capacity': float(permeable_pixels / total_pixels)
            }
        
        # SVF impact on water accumulation (precipitation capture)
        if self.svf_map is not None:
            valid_svf = self.svf_map[~np.isnan(self.svf_map) & (self.svf_map > 0)]
            if len(valid_svf) == 0:
                return None
            
            # High SVF areas receive more direct precipitation
            high_precipitation_areas = valid_svf > 0.7
            moderate_precipitation_areas = (valid_svf > 0.4) & (valid_svf <= 0.7)
            sheltered_areas = valid_svf <= 0.4
            
            metrics['precipitation_exposure'] = {
                'mean_svf': float(np.mean(valid_svf)),
                'high_precipitation_ratio': float(np.sum(high_precipitation_areas) / len(valid_svf)),
                'moderate_precipitation_ratio': float(np.sum(moderate_precipitation_areas) / len(valid_svf)),
                'sheltered_areas_ratio': float(np.sum(sheltered_areas) / len(valid_svf))
            }
        
        # Integrated water accumulation risk assessment
        if self.height_map is not None and self.segmentation_map is not None and self.svf_map is not None:
            # Combine low elevation + impermeable surface + high precipitation
            risk_factors = []
            
            # Low elevation areas (higher risk)
            low_elevation_risk = np.sum(low_areas_mask) / self.height_map.size
            risk_factors.append(low_elevation_risk * 0.4)
            
            # Impermeable surface areas (higher risk)
            impermeable_risk = impermeable_pixels / total_pixels
            risk_factors.append(impermeable_risk * 0.3)
            
            # High precipitation areas (higher risk)
            high_precip_risk = np.sum(high_precipitation_areas) / len(valid_svf)
            risk_factors.append(high_precip_risk * 0.2)
            
            # Flat areas (water pools)
            if 'terrain_flow' in metrics:
                flat_risk = metrics['terrain_flow']['flat_areas_ratio']
                risk_factors.append(flat_risk * 0.1)
            
            total_risk_score = sum(risk_factors)
            
            metrics['integrated_flood_risk'] = {
                'elevation_risk': float(low_elevation_risk),
                'permeability_risk': float(impermeable_risk),
                'precipitation_risk': float(high_precip_risk),
                'terrain_risk': float(flat_risk if 'terrain_flow' in metrics else 0),
                'total_risk_score': float(total_risk_score),
                'risk_level': 'high' if total_risk_score > 0.6 else 'moderate' if total_risk_score > 0.3 else 'low'
            }
        
        # Drainage infrastructure requirements
        if self.segmentation_map is not None and self.height_map is not None:
            # Urban areas with poor drainage potential
            urban_classes = [4, 6, 7, 8, 9, 10]  # urban development classes
            urban_mask = np.isin(self.segmentation_map, urban_classes)
            
            # Urban areas in low-lying regions need better drainage
            urban_low_areas = urban_mask & low_areas_mask
            
            metrics['drainage_infrastructure'] = {
                'urban_areas_ratio': float(np.sum(urban_mask) / self.segmentation_map.size),
                'urban_low_areas_ratio': float(np.sum(urban_low_areas) / self.segmentation_map.size),
                'drainage_priority_areas': float(np.sum(urban_low_areas) / max(np.sum(urban_mask), 1)),
                'infrastructure_need_score': float(np.sum(urban_low_areas) / self.segmentation_map.size * 10)  # Normalized to 10
            }
            
        return metrics
    
    def _fallback_analysis(self, analysis_type: str, missing_data: str) -> Dict[str, Any]:
        """Provide fallback analysis when required data is missing."""
        question = f"Q: Analyze the {analysis_type.replace('_', ' ')} potential of this area."
        answer = f"Analysis requires {missing_data} which is not available for this location."
        
        return {
            "question": question,
            "answer": answer,
            "analysis_type": analysis_type,
            "status": "data_unavailable",
            "context": [analysis_type, "insufficient_data"],
            "generation_method": "fallback"
        } 