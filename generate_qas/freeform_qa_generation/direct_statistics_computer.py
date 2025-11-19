"""
Direct Statistical Computation Module for Enhanced Freeform QA Generation

This module performs direct statistical calculations for best locations, key metrics,
and derives comprehensive scene statistics that are integrated with GPT for enhanced
answer generation. Based on svf_questions_rgb_estimated.py approach.

Features:
- GPU-accelerated statistical computations when available
- Best location detection using argmax/argmin operations
- Multi-metric scoring with weighted combinations
- Region-based analysis for spatial comparisons
- Coordinate-based answer formatting
- Integration with existing freeform QA categories
"""

import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import warnings

# GPU acceleration (optional)
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class DirectStatistics:
    """Container for directly computed statistics"""
    # Best locations
    best_sky_visibility_location: Tuple[int, int]
    best_sky_visibility_score: float
    best_solar_location: Tuple[int, int] 
    best_solar_score: float
    best_development_location: Tuple[int, int]
    best_development_score: float
    
    # Key metrics (3 most important)
    openness_index: float  # Overall sky visibility
    development_potential: float  # Suitability for development
    terrain_complexity: float  # Elevation variation
    
    # Supporting statistics
    elevation_range: Tuple[float, float]  # (min, max)
    mean_elevation: float
    dominant_land_cover: str
    land_cover_diversity: float
    water_presence: bool
    vegetation_ratio: float
    
    # Location descriptions
    best_locations_description: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def get_key_metrics(self) -> Dict[str, float]:
        """Get the 3 key metrics for JSON output"""
        return {
            "openness_index": self.openness_index,
            "development_potential": self.development_potential, 
            "terrain_complexity": self.terrain_complexity
        }


class DirectStatisticsComputer:
    """
    Direct statistical computation engine for freeform QA generation.
    Computes answers using statistical operations rather than LLM inference.
    """
    
    def __init__(self, svf_map: np.ndarray, height_map: np.ndarray, 
                 segmentation_map: np.ndarray, rgb_image: np.ndarray = None,
                 debug: bool = False):
        """
        Initialize with scene data
        
        Args:
            svf_map: Sky View Factor map (0-1 values)
            height_map: Digital Surface Model / height data
            segmentation_map: Land cover segmentation
            rgb_image: RGB image (optional)
            debug: Enable debug output
        """
        self.svf_map = svf_map
        self.height_map = height_map
        self.segmentation_map = segmentation_map
        self.rgb_image = rgb_image
        self.debug = debug
        
        # Validate inputs
        self._validate_inputs()
        
        # Land cover class mapping (adjust based on your segmentation classes)
        self.land_cover_classes = {
            0: 'background',
            1: 'buildings',
            2: 'roads', 
            3: 'vegetation',
            4: 'water',
            5: 'agriculture',
            6: 'other'
        }
    
    def _validate_inputs(self):
        """Validate input data"""
        if self.svf_map.shape != self.height_map.shape:
            raise ValueError("SVF and height maps must have same dimensions")
        if self.svf_map.shape != self.segmentation_map.shape:
            raise ValueError("All input maps must have same dimensions")
        
        # Check for reasonable SVF values
        if not (0 <= np.nanmin(self.svf_map) and np.nanmax(self.svf_map) <= 1):
            if self.debug:
                print(f"  SVF values outside [0,1]: min={np.nanmin(self.svf_map):.3f}, max={np.nanmax(self.svf_map):.3f}")
    
    def gpu_calculate_stats(self, data: np.ndarray) -> Dict[str, float]:
        """GPU-accelerated statistical calculations"""
        if GPU_AVAILABLE and data.size > 10000:  # Use GPU for larger arrays
            try:
                data_gpu = cp.asarray(data)
                valid_mask = ~cp.isnan(data_gpu)
                valid_data = data_gpu[valid_mask]
                
                if len(valid_data) == 0:
                    return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
                
                return {
                    'mean': float(cp.mean(valid_data)),
                    'std': float(cp.std(valid_data)),
                    'min': float(cp.min(valid_data)),
                    'max': float(cp.max(valid_data))
                }
            except Exception as e:
                if self.debug:
                    print(f"GPU calculation failed, falling back to CPU: {e}")
        
        # CPU fallback
        valid_mask = ~np.isnan(data)
        valid_data = data[valid_mask]
        
        if len(valid_data) == 0:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        return {
            'mean': float(np.mean(valid_data)),
            'std': float(np.std(valid_data)),
            'min': float(np.min(valid_data)),
            'max': float(np.max(valid_data))
        }
    
    def find_best_sky_visibility_location(self) -> Tuple[Tuple[int, int], float]:
        """
        Find best location for sky visibility using direct statistical computation.
        Based on SVF values with building penalty similar to svf_questions_rgb_estimated.py
        """
        h, w = self.svf_map.shape
        window_size = min(32, h // 4, w // 4)  # Adaptive window size
        
        best_score = -1
        best_location = (0, 0)
        
        # Grid search for best location
        step = max(1, window_size // 4)
        
        for y in range(0, h - window_size, step):
            for x in range(0, w - window_size, step):
                # Extract window
                svf_window = self.svf_map[y:y+window_size, x:x+window_size]
                seg_window = self.segmentation_map[y:y+window_size, x:x+window_size]
                
                # Calculate SVF score component
                valid_svf = svf_window[~np.isnan(svf_window)]
                if len(valid_svf) == 0:
                    continue
                
                svf_score = np.mean(valid_svf) * 0.7
                
                # Building penalty calculation
                building_pixels = np.sum(seg_window == 1)  #  class 1 is buildings
                building_ratio = building_pixels / (window_size * window_size)
                building_penalty = building_ratio * 0.3
                
                # Edge penalty (avoid edges)
                edge_penalty = 0.0
                if y < window_size or x < window_size or y > h - 2*window_size or x > w - 2*window_size:
                    edge_penalty = 0.1
                
                total_score = svf_score - building_penalty - edge_penalty
                
                if total_score > best_score:
                    best_score = total_score
                    best_location = (y + window_size//2, x + window_size//2)
        
        return best_location, best_score
    
    def find_best_solar_location(self) -> Tuple[Tuple[int, int], float]:
        """Find best location for solar installation using combined metrics"""
        h, w = self.svf_map.shape
        
        # Create combined score map
        # High SVF + low building density + reasonable elevation
        svf_normalized = (self.svf_map - np.nanmin(self.svf_map)) / (np.nanmax(self.svf_map) - np.nanmin(self.svf_map) + 1e-8)
        
        # Building penalty map
        building_mask = (self.segmentation_map == 1).astype(float)
        from scipy.ndimage import uniform_filter
        building_density = uniform_filter(building_mask, size=16, mode='constant')
        
        # Combine scores
        solar_score_map = svf_normalized * 0.6 - building_density * 0.4
        
        # Find best location
        if GPU_AVAILABLE and solar_score_map.size > 10000:
            try:
                score_gpu = cp.asarray(solar_score_map)
                valid_mask = ~cp.isnan(score_gpu)
                if cp.sum(valid_mask) > 0:
                    best_idx = cp.argmax(score_gpu * valid_mask)
                    best_location = tuple(cp.unravel_index(best_idx, score_gpu.shape))
                    best_score = float(score_gpu[best_location])
                else:
                    best_location = (h//2, w//2)
                    best_score = 0.0
            except Exception:
                # CPU fallback
                valid_mask = ~np.isnan(solar_score_map)
                if np.sum(valid_mask) > 0:
                    best_idx = np.argmax(solar_score_map * valid_mask)
                    best_location = np.unravel_index(best_idx, solar_score_map.shape)
                    best_score = solar_score_map[best_location]
                else:
                    best_location = (h//2, w//2)
                    best_score = 0.0
        else:
            # CPU computation
            valid_mask = ~np.isnan(solar_score_map)
            if np.sum(valid_mask) > 0:
                best_idx = np.argmax(solar_score_map * valid_mask)
                best_location = np.unravel_index(best_idx, solar_score_map.shape)
                best_score = solar_score_map[best_location]
            else:
                best_location = (h//2, w//2)
                best_score = 0.0
        
        return best_location, float(best_score)
    
    def find_best_development_location(self) -> Tuple[Tuple[int, int], float]:
        """Find best location for development using terrain and accessibility"""
        h, w = self.height_map.shape
        
        # Calculate terrain slopes
        gy, gx = np.gradient(self.height_map)
        slope = np.sqrt(gx**2 + gy**2)
        
        # Normalize slope (lower is better for development)
        slope_normalized = 1.0 - (slope - np.nanmin(slope)) / (np.nanmax(slope) - np.nanmin(slope) + 1e-8)
        
        # Building proximity (closer to existing buildings is better)
        building_mask = (self.segmentation_map == 1).astype(float)
        if np.sum(building_mask) > 0:
            from scipy.ndimage import distance_transform_edt
            building_distance = distance_transform_edt(1 - building_mask)
            # Normalize and invert (closer is better)
            building_proximity = 1.0 - (building_distance - np.min(building_distance)) / (np.max(building_distance) - np.min(building_distance) + 1e-8)
        else:
            building_proximity = np.ones_like(slope_normalized) * 0.5
        
        # Avoid water areas
        water_penalty = (self.segmentation_map == 4).astype(float)
        
        # Combined development score
        development_score = (slope_normalized * 0.4 + 
                           building_proximity * 0.3 + 
                           self.svf_map * 0.2 - 
                           water_penalty * 0.5)
        
        # Find best location
        valid_mask = ~np.isnan(development_score)
        if np.sum(valid_mask) > 0:
            best_idx = np.argmax(development_score * valid_mask)
            best_location = np.unravel_index(best_idx, development_score.shape)
            best_score = development_score[best_location]
        else:
            best_location = (h//2, w//2)
            best_score = 0.0
        
        return best_location, float(best_score)
    
    def compute_key_metrics(self) -> Tuple[float, float, float]:
        """Compute 3 key metrics for the scene"""
        
        # 1. Openness Index (based on SVF)
        svf_stats = self.gpu_calculate_stats(self.svf_map)
        openness_index = svf_stats['mean']
        
        # 2. Development Potential (terrain + accessibility)
        gy, gx = np.gradient(self.height_map)
        slope = np.sqrt(gx**2 + gy**2)
        slope_stats = self.gpu_calculate_stats(slope)
        
        # Lower slope variance = better for development
        development_potential = max(0.0, 1.0 - (slope_stats['std'] / (slope_stats['mean'] + 1e-8)))
        development_potential = min(1.0, development_potential)
        
        # 3. Terrain Complexity (elevation variation)
        height_stats = self.gpu_calculate_stats(self.height_map)
        terrain_complexity = height_stats['std'] / (height_stats['mean'] + 1e-8)
        terrain_complexity = min(1.0, terrain_complexity / 10.0)  # Normalize to 0-1
        
        return openness_index, development_potential, terrain_complexity
    
    def analyze_land_cover(self) -> Dict[str, Any]:
        """Analyze land cover distribution"""
        unique_classes, counts = np.unique(self.segmentation_map, return_counts=True)
        total_pixels = self.segmentation_map.size
        
        class_ratios = {}
        for cls, count in zip(unique_classes, counts):
            class_name = self.land_cover_classes.get(cls, f'class_{cls}')
            class_ratios[class_name] = count / total_pixels
        
        # Find dominant land cover
        dominant_class = max(class_ratios.items(), key=lambda x: x[1])
        
        # Calculate diversity (Shannon entropy)
        diversity = 0.0
        for ratio in class_ratios.values():
            if ratio > 0:
                diversity += -ratio * np.log(ratio)
        
        # Check for water presence
        water_present = class_ratios.get('water', 0) > 0.01  # More than 1%
        
        # Vegetation ratio
        vegetation_ratio = class_ratios.get('vegetation', 0) + class_ratios.get('agriculture', 0)
        
        return {
            'dominant_land_cover': dominant_class[0],
            'diversity': diversity,
            'water_present': water_present,
            'vegetation_ratio': vegetation_ratio,
            'class_ratios': class_ratios
        }
    
    def generate_location_descriptions(self, best_locations: Dict[str, Tuple[int, int]]) -> Dict[str, str]:
        """Generate human-readable descriptions of best locations"""
        h, w = self.svf_map.shape
        descriptions = {}
        
        for purpose, (y, x) in best_locations.items():
            # Convert to relative position
            rel_y = y / h
            rel_x = x / w
            
            # Determine quadrant
            if rel_y < 0.33:
                v_pos = "top"
            elif rel_y < 0.67:
                v_pos = "middle"
            else:
                v_pos = "bottom"
            
            if rel_x < 0.33:
                h_pos = "left"
            elif rel_x < 0.67:
                h_pos = "center"
            else:
                h_pos = "right"
            
            position = f"{v_pos}-{h_pos}" if h_pos != "center" else v_pos
            if position == "middle-center":
                position = "center"
            
            descriptions[purpose] = position
        
        return descriptions
    
    def compute_comprehensive_statistics(self) -> DirectStatistics:
        """
        Compute comprehensive scene statistics using direct calculations.
        This is the main method that integrates all statistical computations.
        """
        if self.debug:
            print(" Computing direct statistics...")
        
        # Find best locations
        sky_loc, sky_score = self.find_best_sky_visibility_location()
        solar_loc, solar_score = self.find_best_solar_location()
        dev_loc, dev_score = self.find_best_development_location()
        
        # Compute key metrics
        openness, development, complexity = self.compute_key_metrics()
        
        # Analyze terrain
        height_stats = self.gpu_calculate_stats(self.height_map)
        elevation_range = (height_stats['min'], height_stats['max'])
        mean_elevation = height_stats['mean']
        
        # Analyze land cover
        land_cover_analysis = self.analyze_land_cover()
        
        # Generate location descriptions
        best_locations = {
            'sky_visibility': sky_loc,
            'solar_potential': solar_loc,
            'development': dev_loc
        }
        location_descriptions = self.generate_location_descriptions(best_locations)
        
        # Create comprehensive statistics object
        stats = DirectStatistics(
            best_sky_visibility_location=sky_loc,
            best_sky_visibility_score=sky_score,
            best_solar_location=solar_loc,
            best_solar_score=solar_score,
            best_development_location=dev_loc,
            best_development_score=dev_score,
            openness_index=openness,
            development_potential=development,
            terrain_complexity=complexity,
            elevation_range=elevation_range,
            mean_elevation=mean_elevation,
            dominant_land_cover=land_cover_analysis['dominant_land_cover'],
            land_cover_diversity=land_cover_analysis['diversity'],
            water_presence=land_cover_analysis['water_present'],
            vegetation_ratio=land_cover_analysis['vegetation_ratio'],
            best_locations_description=location_descriptions
        )
        
        if self.debug:
            print(f" Computed statistics: openness={openness:.3f}, development={development:.3f}, complexity={complexity:.3f}")
            print(f" Best locations: sky={location_descriptions['sky_visibility']}, solar={location_descriptions['solar_potential']}, dev={location_descriptions['development']}")
        
        return stats


def compute_scene_statistics(scene_data: Dict[str, Any], debug: bool = False) -> DirectStatistics:
    """
    Convenience function to compute statistics from scene data.
    
    Args:
        scene_data: Dictionary containing 'svf', 'dsm'/'height_map', 'seg', 'rgb' keys
        debug: Enable debug output
        
    Returns:
        DirectStatistics object with computed metrics
    """
    # Extract data arrays
    svf_map = scene_data['svf']
    height_map = scene_data.get('dsm', scene_data.get('height_map'))
    segmentation_map = scene_data['seg']
    rgb_image = scene_data.get('rgb')
    
    if height_map is None:
        raise ValueError("Scene data must contain 'dsm' or 'height_map' key")
    
    # Create computer and compute statistics
    computer = DirectStatisticsComputer(
        svf_map=svf_map,
        height_map=height_map,
        segmentation_map=segmentation_map,
        rgb_image=rgb_image,
        debug=debug
    )
    
    return computer.compute_comprehensive_statistics()


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    print(" Testing Direct Statistics Computer...")
    
    # Generate synthetic test data
    h, w = 128, 128
    np.random.seed(42)
    
    # Create synthetic SVF map
    svf_map = np.random.beta(2, 2, (h, w))  # Values between 0-1
    
    # Create synthetic height map
    height_map = np.random.normal(100, 20, (h, w))
    
    # Create synthetic segmentation
    segmentation_map = np.random.choice([0, 1, 2, 3, 4], size=(h, w), p=[0.1, 0.2, 0.15, 0.4, 0.15])
    
    # Test computation
    scene_data = {
        'svf': svf_map,
        'dsm': height_map,
        'seg': segmentation_map
    }
    
    stats = compute_scene_statistics(scene_data, debug=True)
    
    print(f"\n Test Results:")
    print(f"Key Metrics: {stats.get_key_metrics()}")
    print(f"Best Locations: {stats.best_locations_description}")
    print(f"Land Cover: {stats.dominant_land_cover} (diversity: {stats.land_cover_diversity:.3f})")
    print(f"Elevation: {stats.mean_elevation:.1f}m (range: {stats.elevation_range[1]-stats.elevation_range[0]:.1f}m)")
    
    print(" Direct Statistics Computer test completed successfully!")