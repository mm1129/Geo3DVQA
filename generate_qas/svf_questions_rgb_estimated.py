import os
import json
import numpy as np
import random
import os
import math
import requests
import json
from utils import tqdm_safe_print, write_json_line, select_choices_with_diversity, select_choices_prioritizing_correct_gap, add_short_instruction, bias_free_shuffle
from utils.error_handler import ErrorHandler, ErrorType
from validation.question_validators import validate_question_quality, log_skip_decision, log_validation_passed
import warnings

try:
    from numba import jit, prange
    os.environ['NUMBA_DISABLE_JIT'] = '0'
    NUMBA_AVAILABLE = True
except ImportError:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def prange(*args, **kwargs):
        return range(*args, **kwargs)
    
    NUMBA_AVAILABLE = False

try:
    import cupy as cp
    GPU_AVAILABLE = True
    tqdm_safe_print(" GPU processing (CuPy) is available")
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False
    tqdm_safe_print("GPU processing unavailable (using NumPy)")

warnings.filterwarnings('ignore', category=UserWarning)

# Import pixel-level implementations from svf_questions_re.py
try:
    from svf_questions_re import ConstructSVFQuestion
    PIXEL_LEVEL_AVAILABLE = True
except ImportError:
    PIXEL_LEVEL_AVAILABLE = False
    tqdm_safe_print("Warning: svf_questions_re.py not available. Pixel-level questions will be skipped.")

# Import hard category implementations
try:
    from svf_questions_hard import SVFHardQuestionMixin
    HARD_CATEGORIES_AVAILABLE = True
    tqdm_safe_print(" Hard categories loaded successfully")
except ImportError as e:
    HARD_CATEGORIES_AVAILABLE = False
    tqdm_safe_print(f" Warning: svf_questions_hard.py not available: {e}. Hard categories will be skipped.")
    
    # Create dummy mixin class for safe inheritance
    class SVFHardQuestionMixin:
        def _init_hard_templates(self, mode="train"):
            pass

def gpu_calculate_svf_stats(svf_map):
    """GPU-accelerated SVF statistics calculation"""
    if GPU_AVAILABLE:
        svf_gpu = cp.asarray(svf_map)
        return {
            'mean': float(cp.mean(svf_gpu)),
            'std': float(cp.std(svf_gpu)),
            'min': float(cp.min(svf_gpu)),
            'max': float(cp.max(svf_gpu))
        }
    else:
        return {
            'mean': float(np.mean(svf_map)),
            'std': float(np.std(svf_map)),
            'min': float(np.min(svf_map)),
            'max': float(np.max(svf_map))
        }

def gpu_find_extremes(svf_map, find_max=True):
    """GPU-accelerated extreme value search"""
    if GPU_AVAILABLE:
        svf_gpu = cp.asarray(svf_map)
        if find_max:
            idx = cp.argmax(svf_gpu)
        else:
            idx = cp.argmin(svf_gpu)
        return cp.unravel_index(idx, svf_gpu.shape)
    else:
        if find_max:
            idx = np.argmax(svf_map)
        else:
            idx = np.argmin(svf_map)
        return np.unravel_index(idx, svf_map.shape)

try:
    @jit(nopython=True, parallel=True)
    def _fast_distance_matrix(coords):
        """Optimized distance matrix calculation"""
        n = coords.shape[0]
        distances = np.zeros((n, n))
        for i in prange(n):
            for j in prange(i+1, n):
                dx = coords[i, 0] - coords[j, 0]
                dy = coords[i, 1] - coords[j, 1]
                dist = (dx*dx + dy*dy) ** 0.5
                distances[i, j] = dist
                distances[j, i] = dist
        return distances

    @jit(nopython=True)
    def _fast_stats_calculation(values):
        """Optimized statistics calculation"""
        n = len(values)
        if n == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        range_val = max_val - min_val
        
        return mean_val, std_val, min_val, max_val, range_val

except:
    def _fast_distance_matrix(coords):
        """Fallback: NumPy vectorized distance calculation"""
        diff = coords[:, np.newaxis] - coords[np.newaxis, :]
        return np.sqrt(np.sum(diff**2, axis=2))
    
    def _fast_stats_calculation(values):
        """Fallback: Standard statistics calculation"""
        if len(values) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        return np.mean(values), np.std(values), np.min(values), np.max(values), np.max(values) - np.min(values)

if not NUMBA_AVAILABLE:
    tqdm_safe_print("Info: Using optimized NumPy operations for high performance.")

def _format_coordinate_answer(label, bbox):
    """
    Format coordinate-based answer for FT-mode
    
    Args:
        label: Region label (e.g., 'A', 'B', 'C', 'D')
        bbox: Bounding box in format [xmin, ymin, xmax, ymax] (percentages)
        
    Returns:
        Formatted answer string (e.g., 'Region A: [24%, 8%, 48%, 32%]')
    """
    xmin, ymin, xmax, ymax = bbox
    return f"Region {label}: [{xmin:.0f}%, {ymin:.0f}%, {xmax:.0f}%, {ymax:.0f}%]"

def assign_balanced_labels(regions, target_region):
    """
    Label distribution function to statistically balance correct answer positions
    
    Args:
        regions: List of regions
        target_region: Correct answer region
        
    Returns:
        List of regions with redistributed labels
    """
    labels = ['A', 'B', 'C', 'D']
    available_labels = labels[:len(regions)]
    
    for _ in range(5):
        random.shuffle(available_labels)
    
    random_factor = random.random()
    position_weights = [1.0 + 0.3 * random.random() for _ in available_labels]
    weighted_choice = random.choices(available_labels, weights=position_weights, k=1)[0]
    
    final_choice = random.choice([available_labels[0], weighted_choice])
    
    target_region['label'] = final_choice
    
    remaining_labels = [l for l in available_labels if l != final_choice]
    
    for _ in range(3):
        random.shuffle(remaining_labels)
    
    other_regions = [r for r in regions if r != target_region]
    for i, region in enumerate(other_regions):
        if i < len(remaining_labels):
            region['label'] = remaining_labels[i]
    
    return regions

class ConstructSVFQuestionRGB(SVFHardQuestionMixin):
    def __init__(self, estimated_svf_map, estimated_height_map=None, estimated_segmentation_map=None, rgb_image=None, file_path=None, cnt=0, debug=False, use_gpt_rephrase=False, openai_api_key=None, hard_ratio=0.0, mode="train", coordinate_answers=False, balanced_categories=False):
        """
        Initialize the SVF Question generator with RGB support.
        
        Parameters:
        -----------
        estimated_svf_map : np.ndarray
            Estimated Sky View Factor map
        estimated_height_map : np.ndarray, optional
            Estimated height/DSM map
        estimated_segmentation_map : np.ndarray, optional
            Estimated segmentation map
        rgb_image : np.ndarray, optional
            RGB image
        file_path : str, optional
            File path for unique seeding
        cnt : int, optional
            Counter for unique seeding
        debug : bool, optional
            Enable debug output
        use_gpt_rephrase : bool, optional
            Enable GPT rephrasing
        openai_api_key : str, optional
            OpenAI API key for GPT rephrasing
        hard_ratio : float, optional
            Ratio of hard questions (0.0 to 1.0)
        mode : str, optional
            Processing mode ('train' or 'test') for template selection
        coordinate_answers : bool, optional
            If True, generate coordinate-based answers (e.g., 'Region A: [24%, 8%, 48%, 32%]') 
            instead of choice-based answers (e.g., 'Region A'). Default is False.
        """
        self.svf_map = estimated_svf_map
        self.height_map = estimated_height_map
        self.segmentation_map = estimated_segmentation_map
        self.rgb_image = rgb_image
        self.questions = []
        self.answers = []
        self.canonical_questions = []
        self.file_path = self._process_file_path(file_path)
        self.cnt = cnt
        self._debug = debug
        self.use_gpt_rephrase = use_gpt_rephrase
        self.openai_api_key = openai_api_key
        self.hard_ratio = hard_ratio
        self.category_timing = {}
        self.total_generation_time = 0.0
        self.mode = mode
        self.coordinate_answers = coordinate_answers
        self.balanced_categories = balanced_categories
        self.error_handler = ErrorHandler(debug=self._debug)
        
        if HARD_CATEGORIES_AVAILABLE:
            self._init_hard_templates(mode)
        
        self._cache = {
            'valid_mask': None,
            'svf_stats': None,
            'candidate_points': {},
            'distance_matrix': {},  # LRUキャッシュで自動管理
            'segmentation_stats': None
        }
        self._cache_max_size = 50

        self.question_templates = {
            'sky_visibility': [
                "Which location has the highest sky visibility? (Where can you see the most sky?)",
                "Where is the view of the sky most unobstructed?",
                "Which spot provides the clearest skyward view?",
                "Which location provides the highest upward view?",
                "Where can you see the largest portion of the sky?",
                "Which spot has the most open sky view?",
                "Where is sky access most unrestricted?"
            ],
            'building_density': [
                "Which area has the highest urban density (highest concentration of buildings and urban structures)?",
                "Where is the building concentration highest?",
                "Which region shows the most dense urban development?",
                "Which area has the most crowded urban layout?",
                "Where are buildings most tightly packed?",
                "Which location has the densest urban fabric?"
            ],
            'spatial_openness': [  # Renamed from openness_assessment
                "Which location has the most open space with good sky visibility?",
                "Where is the area most spatially open and unobstructed?",
                "Which region shows the highest spatial openness characteristics?",
                "Where can you find the most open and spacious environment?",
                "Which area demonstrates maximum openness with minimal obstructions?",
                "Where is the most expansive open space located?"
            ],
            'visibility_range': [
                "Which location offers the highest visibility range? (Where can you see the furthest?)",
                "Where can you see the furthest distance?",
                "Which spot provides the highest long-distance view?",
                "Which area offers the most extensive view?",
                "Where is the view range most expansive?",
                "Which location has the most comprehensive sight lines?",
                "Where can you observe the widest area?"
            ],
            'sun_exposure': [  # 図表: "Sun Exposure" (exact name from figure)
                "Which location receives the most sunlight? (Which location looks brightest or most open to the sky?)",
                "Where is solar exposure highest?",
                "Which area gets the maximum sun exposure?",
                "Where does the sun reach most directly?",
                "Which location has the strongest solar access?",
                "Where is direct sunlight most abundant?"
            ],
            'grid_extremes': [  # Renamed from svf_extreme_grid
                "Which grid cell contains the {extreme_type} SVF value in the entire area?",
                "In which grid section can you find the {extreme_type} sky view factor?",
                "Which area of the grid shows the {extreme_type} sky visibility?",
                "Where in the grid is the {extreme_type} SVF measurement located?",
                "Which grid cell has the {extreme_type} sky exposure value?"
            ],
            'svf_average_region': [
                # paper_figure1.pngの表現を追加
                "In which region is the highest average SVF found?",
                "Which region has the highest average SVF value?",
                "Which region shows the highest mean sky view factor?",
                "Which region has higher average sunlight exposure?",
                "Where is the mean SVF value highest across the region?",
                "In which region is average sky exposure highest?",
                "Where do you find the highest regional SVF mean?",
                "Which zone has the highest average sky view factor?",
            ],
            'regional_svf_variability': [  # Renamed from svf_region_analysis
                "Which region shows the {analysis_type}?",
                "Where in the area can you identify the {analysis_type}?", 
                "Which zone demonstrates the {analysis_type}?",
                "In which region is the {analysis_type} found?",
                "Where do you observe the {analysis_type}?"
            ],
            'best_landcover_balance': [  # 図表: "Best landcover balance" (exact name from figure)
                "Which region offers the most scenic landcover composition with optimal natural-artificial balance?",
                "Where can you find the most visually appealing green coverage while maintaining landscape diversity?",
                "Which area demonstrates the most aesthetically balanced vegetation distribution across different environments?", 
                "Where is the visual scenic quality highest with comprehensive green coverage and diverse land use patterns?",
                "Which region shows the most harmonious natural element distribution from a landscape perspective?",
                "Where do you find the most scenic vegetation density while preserving visual landscape variety?"
            ],
            'hard_pixel': [
                "What is the exact SVF value at pixel coordinates ({x}, {y})?",
                "Calculate the precise Sky View Factor at point ({x}, {y}).",
                "Determine the exact SVF measurement at location ({x}, {y}).",
                "What is the precise sky visibility value at coordinates ({x}, {y})?",
                "Find the exact SVF value at the specified pixel location ({x}, {y})."
            ],
            'hard_grid_5×5': [
                "Which cell in the 5×5 grid has the highest average SVF value?",
                "In the 5×5 grid analysis, which cell shows the most sky visibility?",
                "Which 5×5 grid cell has the maximum solar exposure?",
                "Among the 5×5 grid cells, which has the highest SVF measurement?",
                "Which cell in the 5×5 grid shows the greatest sky openness?"
            ],
            'hard_ranking': [
                "Rank these three regions by their SVF values from highest to lowest: {region1}, {region2}, {region3}.",
                "Order these areas by sky visibility from most to least open: {region1}, {region2}, {region3}.",
                "Arrange these regions by their openness levels from highest to lowest: {region1}, {region2}, {region3}.",
                "Sort these three areas by SVF values in descending order: {region1}, {region2}, {region3}.",
                "List these regions from most to least sky-visible: {region1}, {region2}, {region3}."
            ]
        }

        # Add test-mode specific question templates
        if self.mode == "test":
            self.question_templates['sky_visibility'].append("Which area offers the best overhead openness?")
            self.question_templates['building_density'].append("Which zone exhibits the heaviest structural concentration?")
            self.question_templates['spatial_openness'].append("Which spot demonstrates maximum spatial freedom?")
            self.question_templates['visibility_range'].append("Which point commands the broadest visual field?")
            self.question_templates['sun_exposure'].append("Which position experiences peak daylight access?")
            self.question_templates['svf_average_region'].append("Which territory displays the peak mean sky exposure?")
            self.question_templates['regional_svf_variability'].append("Which territorial division exhibits the {analysis_type}?")
            self.question_templates['best_landcover_balance'].append("Which region has the highest vegetation coverage with balanced natural-artificial distribution?")

        # Update question templates with hard category templates if available
        if HARD_CATEGORIES_AVAILABLE:
            self._init_hard_templates(mode)
            self.question_templates.update(self.HARD_QUESTION_TEMPLATES)

        self.gpt_prompts = {
            'casual': "Rephrase this question in a casual, conversational tone: {question}",
            'technical': "Rephrase this question using more technical terminology: {question}",
            'perspective': "Rephrase this question from a {context} perspective: {question}",
            'simple': "Simplify this question while keeping the same meaning: {question}",
            'detailed': "Expand this question with more specific details: {question}"
        }

        self.perspective_contexts = [
            "urban planning", "environmental assessment", "safety evaluation", 
            "architectural design", "real estate development", "landscape design"
        ]

        self.CATEGORY_WEIGHTS = {
            
            # 最低成績 (20%以下) - 2倍重み（3倍→2倍に軽減）
            'grid_5×5': 2.0,                        # 11.40% (図表: "Grid 5×5")
            'best_landcover_balance': 2.0,          # 23.31% (図表: "Best landcover balance")
            
            # 低成績 (20-40%) - 1.5倍重み（2倍→1.5倍に軽減）
            'visibility_range': 1.5,                # 31.23% (図表: "Visibility Range")
            'building_density': 1.5,                # 36.24% (図表: "Building Density")
            'precise_svf': 1.5,                     # 40.75% (図表: "Precise SVF")
            'spatial_openness': 1.5,                # 40.93% (図表: "Spatial Openness")
            
            # 中成績 (40-60%) - 1.2倍重み（1.5倍→1.2倍に軽減）
            'sky_visibility': 1.2,                  # 46.18% (図表: "Sky Visibility")
            'regional_svf_variability': 1.2,        # 51.64% (図表: "Regional SVF variability")
            
            # 高成績 (60%以上) - 標準重み
            'region_ranking': 1.0,                  # 93.51% (図表: "Region Ranking")
            'regional_highest_svf': 1.0,            # 100.00% (図表: "Regional Highest SVF")
            'sun_exposure': 1.0,                    # (図表: "Sun Exposure")
        }

        self.QUESTION_TYPES = {
            'sky_visibility': [self.CATEGORY_WEIGHTS.get('sky_visibility', 1.0) * 0.6, self.skyVisibility, 0],  # 図表: "Sky Visibility" (SVF + Segmentation) - Grid版と分散
            'spatial_openness': [self.CATEGORY_WEIGHTS.get('spatial_openness', 1.0), self.opennessAssessment, 0],  # 図表: "Spatial Openness" (renamed from openness_assessment)
            'visibility_range': [self.CATEGORY_WEIGHTS.get('visibility_range', 1.0), self.visibilityRange, 0],  # 図表: "Visibility Range" (SVF + Height)
            'building_density': [self.CATEGORY_WEIGHTS.get('building_density', 1.0) * 0.6, self.urbanDensity, 1],  # 図表: "Building Density" (renamed from urban_density) - Grid版と分散
            'sun_exposure': [self.CATEGORY_WEIGHTS.get('sun_exposure', 1.0), self.sunExposure, 0],  # Renamed from hard_sun_exposure (図表: "Sun Exposure")
        }

        # Add hard categories if available and methods exist
        if HARD_CATEGORIES_AVAILABLE:
            hard_question_types = {}
            
            # Check each hard method exists before adding
            hard_methods = [  
                ('precise_svf', 'hardPixel', 0),  # Renamed from hard_pixel (図表: "Precise SVF")
                ('region_ranking', 'hardRanking', 0),  # Renamed from hard_ranking (図表: "Region Ranking")
            ]
            
            for category, method_name, requirement in hard_methods:
                if hasattr(self, method_name):
                    hard_question_types[category] = [
                        self.CATEGORY_WEIGHTS.get(category, 1.0), 
                        getattr(self, method_name), 
                        requirement
                    ]
                    if self._debug:
                        tqdm_safe_print(f" Added hard category: {category}")
                else:
                    if self._debug:
                        tqdm_safe_print(f" Skipped hard category: {category} (method {method_name} not found)")
            
            self.QUESTION_TYPES.update(hard_question_types)

        # Filter out question types if necessary maps are missing
        if self.height_map is None:
            self.QUESTION_TYPES = {k: v for k, v in self.QUESTION_TYPES.items() if k not in ['building_density', 'spatial_openness', 'visibility_range', 'scenic_quality']}

        if self.segmentation_map is None:
            self.QUESTION_TYPES = {k: v for k, v in self.QUESTION_TYPES.items() if k not in ['building_density', 'sky_visibility', 'scenic_quality', 'best_landcover_balance']}

        if self.rgb_image is None:
            self.QUESTION_TYPES = {k: v for k, v in self.QUESTION_TYPES.items() if k not in ['scenic_quality', 'best_landcover_balance']}

        self.seeds = {}
        self.seed_base = None

        self._apply_adaptive_weights()
        self._initialize_image_compatibility_system()
        
        self.MIN_COORDINATE_DISTANCE_PERCENT = 15.0
        self.MIN_SCORE_GAP_STANDARD = 0.15
        self.MIN_SCORE_GAP_CONSERVATIVE = 0.20

        if self._debug:
            tqdm_safe_print("=== Category Weights Applied ===")
            for category, weight in self.CATEGORY_WEIGHTS.items():
                if category in self.QUESTION_TYPES:
                    performance = ""
                    if weight == 5.0:
                        performance = " (Very Poor: 20%以下)"
                    elif weight == 3.0:
                        performance = " (Poor: 20-40%)"
                    elif weight == 2.0:
                        performance = " (Average: 40-60%)"
                    elif weight == 1.0:
                        performance = " (Good: 60%以上)"
                    tqdm_safe_print(f"  {category}: weight={weight}{performance}")
            tqdm_safe_print("================================")

        if self.segmentation_map is not None:
            self.QUESTION_TYPES.update({
                'landcover_type': [1.0, self.landcoverType, 1],
                'land_use': [1.0, self.landUse, 1],
            })
        
        if self.height_map is not None:
            self.QUESTION_TYPES.update({
                'height_average': [1.0, self.heightAverage, 1],
                'highest_region': [1.0, self.highestRegion, 1],
            })

        self.question_templates.update({
            'landcover_type': [
                "Which are land-use types are there in this image?",
                "What types of land cover can you identify in this area?",
                "Which land classification categories are present?",
                "What different terrain types are visible?",
                "Which surface cover types can be distinguished?"
            ],
            'land_use': [
                "What are the top 3 primary land uses in this area?",
                "Which 3 land utilization patterns are most dominant in this area?",
                "What are the top 3 functional land categories present in this area?",
                "Which 3 land development types can be most identified in this area?",
                "What are the 3 most evident land allocation purposes in this area?"
            ],
            
            'height_average': [
                "What is the average height value for the region {region}?",
                "Calculate the mean elevation within the area {region}.",
                "Determine the average height measurement for region {region}.",
                "What is the regional average height value at {region}?",
                "Find the mean elevation value within the specified region {region}.",
            ],
            'height_std': [
                "What height patterns are observable?",
                "What height variation characteristics can be identified?",
                "What type of elevation distribution is present?",
                "What height variability patterns are evident?",
                "What terrain variation patterns can be observed?",
            ],
            'highest_region': [
                "Which region has the highest average elevation?",
                "Which area shows the greatest average height?", 
                "Where can you find the highest mean elevation?",
                "Which region represents the most elevated terrain on average?",
                "Which area has the highest overall elevation?"
            ],
            
            'building_density_grid': [
                "Which grid cell has the highest urban density?",
                "In the 3×3 grid, where is building concentration highest?",
                "Which grid area shows the greatest urban development?",
                "Which section of the grid has the most buildings?",
                "Where in the grid is urban density most intense?",
                "Which grid zone shows the highest building coverage?",
                "Which grid cell represents the most urbanized area?"
            ],
            'sky_visibility_grid': [
                "Which grid cell has the best sky visibility?",
                "In the 3×3 grid, where is sky openness highest?",
                "Which grid area offers the clearest sky view?",
                "Which section has the most unobstructed sky?",
                "Where in the grid can you see the most sky?",
                "Which grid zone has the best celestial visibility?",
                "Which grid cell provides optimal sky exposure?"
            ],
            'visibility_range_grid': [
                "Which grid cell offers the best visibility range?",
                "In the 3×3 grid, where is the viewing distance greatest?",
                "Which grid area provides the longest sight lines?",
                "Which section offers the best vantage point?",
                "Where in the grid can you see the furthest?",
                "Which grid zone has the optimal viewing potential?",
                "Which grid cell maximizes visual reach?"
            ],
        })

    def _generate_coordinate_answer(self, regions, target_region):
        """
        Generate coordinate-based answer format for FT-mode
        
        Args:
            regions: List of regions with bounding boxes
            target_region: The correct target region
            
        Returns:
            Formatted coordinate answer string
        """
        # Use assign_balanced_labels to get the label
        labeled_regions = assign_balanced_labels(regions, target_region)
        
        # Find the target region and format the coordinate answer
        for region in labeled_regions:
            if region == target_region:
                bbox = region.get('bbox', [0, 0, 100, 100])  # Default fallback
                label = region.get('label', 'A')
                if self.coordinate_answers:
                    return _format_coordinate_answer(label, bbox)
                else:
                    return f"Region {label}"
        
        # Fallback
        return "Region A"

    def _get_cached_valid_mask(self):
        """Get cached valid mask"""
        if self._cache['valid_mask'] is None:
            self._cache['valid_mask'] = ~np.isnan(self.svf_map) & (self.svf_map > 0)
        return self._cache['valid_mask']
    
    def _get_cached_svf_stats(self):
        """Get cached SVF statistics"""
        if self._cache['svf_stats'] is None:
            valid_mask = self._get_cached_valid_mask()
            if np.sum(valid_mask) > 0:
                valid_values = self.svf_map[valid_mask]
                #  Numba高速統計計算
                mean_val, std_val, min_val, max_val, range_val = _fast_stats_calculation(valid_values)
                self._cache['svf_stats'] = {
                    'mean': mean_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val,
                    'range': range_val,
                    'percentiles': np.percentile(valid_values, [25, 50, 75])
                }
            else:
                self._cache['svf_stats'] = None
        return self._cache['svf_stats']
    
    def _get_cached_candidate_points(self, threshold_key):
        """Get cached candidate points"""
        if threshold_key not in self._cache['candidate_points']:
            valid_mask = self._get_cached_valid_mask()
            if threshold_key == 'high_svf':
                threshold = 0.7
                candidate_mask = valid_mask & (self.svf_map > threshold)
            elif threshold_key == 'medium_svf':
                threshold = 0.5
                candidate_mask = valid_mask & (self.svf_map > threshold)
            else:
                candidate_mask = valid_mask
            
            y_indices, x_indices = np.where(candidate_mask)
            self._cache['candidate_points'][threshold_key] = (y_indices, x_indices)
        
        return self._cache['candidate_points'][threshold_key]

    def _stratified_sampling(self, y_indices, x_indices, target_count):
        """
        Perform stratified sampling to ensure area diversity and reduce spatial bias.
        
        Args:
            y_indices: Y coordinates of candidate points
            x_indices: X coordinates of candidate points
            target_count: Target number of points to sample
            
        Returns:
            sample_indices: Indices of selected points
        """
        if len(y_indices) <= target_count:
            return np.arange(len(y_indices))
        
        # Get image dimensions
        h, w = self.svf_map.shape
        
        # Create spatial grid (divide image into 4×4 = 16 regions)
        grid_size = 4
        y_bins = np.linspace(0, h, grid_size + 1)
        x_bins = np.linspace(0, w, grid_size + 1)
        
        # Assign each point to a spatial bin
        y_bin_indices = np.digitize(y_indices, y_bins) - 1
        x_bin_indices = np.digitize(x_indices, x_bins) - 1
        
        # Ensure indices are within bounds
        y_bin_indices = np.clip(y_bin_indices, 0, grid_size - 1)
        x_bin_indices = np.clip(x_bin_indices, 0, grid_size - 1)
        
        # Create region labels
        region_labels = y_bin_indices * grid_size + x_bin_indices
        
        # Calculate points per region
        points_per_region = target_count // (grid_size * grid_size)
        remaining_points = target_count % (grid_size * grid_size)
        
        selected_indices = []
        
        # Sample from each region
        for region_id in range(grid_size * grid_size):
            region_mask = region_labels == region_id
            region_indices = np.where(region_mask)[0]
            
            if len(region_indices) == 0:
                continue
                
            # Determine number of points to sample from this region
            n_sample = points_per_region
            if remaining_points > 0:
                n_sample += 1
                remaining_points -= 1
            
            # Sample points from this region
            if len(region_indices) <= n_sample:
                selected_indices.extend(region_indices)
            else:
                # Use SVF values for weighted sampling within region
                region_svf_values = self.svf_map[y_indices[region_indices], x_indices[region_indices]]
                # Higher SVF values get slightly higher probability
                weights = region_svf_values + 0.1  # Add small constant to avoid zero weights
                weights = weights / np.sum(weights)
                
                sampled = np.random.choice(
                    region_indices, 
                    size=n_sample, 
                    replace=False, 
                    p=weights
                )
                selected_indices.extend(sampled)
        
        # If we still need more points, randomly sample from remaining
        if len(selected_indices) < target_count:
            remaining_mask = np.ones(len(y_indices), dtype=bool)
            remaining_mask[selected_indices] = False
            remaining_indices = np.where(remaining_mask)[0]
            
            if len(remaining_indices) > 0:
                additional_count = min(target_count - len(selected_indices), len(remaining_indices))
                additional_indices = np.random.choice(remaining_indices, size=additional_count, replace=False)
                selected_indices.extend(additional_indices)
        
        return np.array(selected_indices[:target_count])

    def _initialize_image_compatibility_system(self):
        """Initialize image compatibility system to assess category suitability."""
        self.image_compatibility = {}
        h, w = self.svf_map.shape
        
        valid_mask = self._get_cached_valid_mask()
        if np.sum(valid_mask) == 0:
            for category in self.QUESTION_TYPES.keys():
                self.image_compatibility[category] = {
                    'suitable': False,
                    'confidence': 0.0,
                    'reason': 'No valid SVF data'
                }
            return
            
        svf_stats = self._get_cached_svf_stats()
        svf_mean = svf_stats['mean']
        svf_std = svf_stats['std']
        svf_range = svf_stats['range']
        
        image_complexity = self._assess_image_complexity()
        self._evaluate_category_compatibility(svf_mean, svf_std, svf_range, image_complexity)
        
        if self._debug:
            tqdm_safe_print("=== Image Compatibility Assessment ===")
            for category, comp in self.image_compatibility.items():
                status = " SUITABLE" if comp['suitable'] else " SKIP"
                tqdm_safe_print(f"  {category}: {status} (conf: {comp['confidence']:.2f}) - {comp['reason']}")
    
    def _assess_image_complexity(self):
        """Assess image complexity from spatial diversity, value distribution, and segmentation."""
        complexity = {
            'spatial_diversity': 0.0,
            'value_distribution': 0.0, 
            'segmentation_complexity': 0.0,
            'overall': 0.0
        }
        
        h, w = self.svf_map.shape
        if h > 20 and w > 20:
            try:
                grad_x = np.abs(np.diff(self.svf_map, axis=1))
                grad_y = np.abs(np.diff(self.svf_map, axis=0))
                spatial_variation = np.nanmean(grad_x) + np.nanmean(grad_y)
                complexity['spatial_diversity'] = min(1.0, spatial_variation / 0.5)
            except:
                complexity['spatial_diversity'] = 0.5
        
        valid_mask = ~np.isnan(self.svf_map) & (self.svf_map > 0)
        if np.sum(valid_mask) > 100:
            try:
                valid_values = self.svf_map[valid_mask]
                hist, _ = np.histogram(valid_values, bins=20, density=True)
                hist = hist[hist > 0]
                entropy = -np.sum(hist * np.log(hist + 1e-10))
                complexity['value_distribution'] = min(1.0, entropy / 3.0)
            except:
                complexity['value_distribution'] = 0.5
        
        if self.segmentation_map is not None:
            try:
                unique_classes = len(np.unique(self.segmentation_map))
                complexity['segmentation_complexity'] = min(1.0, unique_classes / 15.0)
            except:
                complexity['segmentation_complexity'] = 0.5
        
        complexity['overall'] = np.mean([
            complexity['spatial_diversity'],
            complexity['value_distribution'], 
            complexity['segmentation_complexity']
        ])
        
        return complexity
    
    def _evaluate_category_compatibility(self, svf_mean, svf_std, svf_range, complexity):
        """Evaluate category compatibility with image characteristics."""
        
        rules = {
            'svf_comparison': {
                'min_range': 0.1,
                'min_std': 0.05,
                'complexity_factor': 0.3
            },
            'sky_visibility': {
                'min_range': 0.15,
                'min_high_values': 0.3,
                'complexity_factor': 0.2
            },
            'urban_density': {
                'requires_segmentation': True,
                'min_building_ratio': 0.1,
                'complexity_factor': 0.4
            },
            'natural_artificial_ratio': {
                'requires_segmentation': True,
                'requires_rgb': True,
                'min_diversity': 0.3,
                'complexity_factor': 0.5
            },
            'openness_assessment': {
                'min_range': 0.2,
                'min_open_areas': 0.2,
                'complexity_factor': 0.3
            },
            'visibility_range': {
                'min_range': 0.15,
                'min_std': 0.08,
                'complexity_factor': 0.4
            }
        }
        
        for category, rule in rules.items():
            if category not in self.QUESTION_TYPES:
                continue
                
            suitable = True
            confidence = 1.0
            reasons = []
            
            if rule.get('requires_segmentation', False) and self.segmentation_map is None:
                suitable = False
                confidence = 0.0
                reasons.append("No segmentation map")
            
            if rule.get('requires_rgb', False) and self.rgb_image is None:
                suitable = False
                confidence = 0.0
                reasons.append("No RGB image")
            
            if suitable and 'min_range' in rule and svf_range < rule['min_range']:
                confidence *= 0.5
                reasons.append(f"Low SVF range ({svf_range:.3f})")
            
            if suitable and 'min_std' in rule and svf_std < rule['min_std']:
                confidence *= 0.6
                reasons.append(f"Low SVF std ({svf_std:.3f})")
            
            complexity_threshold = rule.get('complexity_factor', 0.3)
            if complexity['overall'] < complexity_threshold:
                confidence *= 0.7
                reasons.append(f"Low complexity ({complexity['overall']:.2f})")
            
            if confidence < 0.3:
                suitable = False
            
            self.image_compatibility[category] = {
                'suitable': suitable,
                'confidence': confidence,
                'reason': '; '.join(reasons) if reasons else 'Good compatibility'
            }

    def _enhanced_shuffle_choices(self, combined_choices, shuffle_rounds=None):
        """
        位置バイアスを完全に除去する高品質シャッフル
        """
        if shuffle_rounds is None:
            shuffle_rounds = random.randint(3, 7)
        
        # 複数回のシャッフルでランダム性を向上
        for _ in range(shuffle_rounds):
            combined_choices = bias_free_shuffle(combined_choices)
        
        return combined_choices
    
    def _calculate_coordinate_distance(self, coord1, coord2):
        """Calculate distance between two coordinates"""
        if isinstance(coord1, tuple) and len(coord1) == 2 and isinstance(coord2, tuple) and len(coord2) == 2:
            dx = coord1[0] - coord2[0]
            dy = coord1[1] - coord2[1]
            return (dx*dx + dy*dy) ** 0.5  # math.sqrtより高速
        return 0.0
    
    def _calculate_distance_matrix_vectorized(self, coords):
        """Calculate distance matrix for coordinate list"""
        coords_key = tuple(tuple(c) for c in coords)
        if coords_key in self._cache['distance_matrix']:
            return self._cache['distance_matrix'][coords_key]
        
        coords_array = np.array(coords, dtype=np.float64)
        if len(coords_array) == 0:
            return np.array([])
        
        distances = _fast_distance_matrix(coords_array)
        
        #  メモリ効率：キャッシュサイズ管理
        if len(self._cache['distance_matrix']) >= self._cache_max_size:
            # 最も古いキャッシュエントリを削除（簡易LRU）
            oldest_key = next(iter(self._cache['distance_matrix']))
            del self._cache['distance_matrix'][oldest_key]
        
        self._cache['distance_matrix'][coords_key] = distances
        return distances
    
    def _filter_minimum_distance_coordinates(self, candidate_coords, min_distance_percent=None):
        """Filter coordinates by minimum distance"""
        if min_distance_percent is None:
            min_distance_percent = self.MIN_COORDINATE_DISTANCE_PERCENT
        
        if len(candidate_coords) <= 1:
            return candidate_coords
        
        percent_coords = np.array([
            self._get_relative_point(coord[0], coord[1]) if isinstance(coord, tuple) and len(coord) == 2 else coord
            for coord in candidate_coords
        ])
        
        distances = self._calculate_distance_matrix_vectorized(percent_coords)
        
        selected_indices = []
        for i in range(len(candidate_coords)):
            if len(selected_indices) == 0:
                selected_indices.append(i)
                continue
            
            # 既選択点との最小距離をチェック
            min_dist = np.min(distances[i, selected_indices])
            if min_dist >= min_distance_percent:
                selected_indices.append(i)
        
        return [candidate_coords[i] for i in selected_indices]
    
    def _get_adaptive_score_gap(self, category_name):
        """Get adaptive score gap based on category and image complexity"""
        base_gap = self.MIN_SCORE_GAP_STANDARD
        
        # Adjust based on image compatibility
        if hasattr(self, 'image_compatibility') and category_name in self.image_compatibility:
            compatibility = self.image_compatibility[category_name]
            if compatibility['confidence'] < 0.5:
                base_gap = self.MIN_SCORE_GAP_CONSERVATIVE
        
        return base_gap
    
    def _calculate_edge_penalty(self, y, x, weight=0.15):
        """
        Calculate penalty for points near image edges.
        Penalty increases quadratically with normalized distance from center.
        
        Args:
            y, x: Point coordinates
            weight: Penalty weight factor
        """
        h, w = self.svf_map.shape
        center_y, center_x = h / 2, w / 2

        # Normalized distance from center (0.0 - 1.0)
        # Normalized by diagonal length so corners reach ~1.0
        norm_dist = math.sqrt((x - center_x)**2 + (y - center_y)**2) / math.sqrt(center_x**2 + center_y**2)
        
        # Quadratic penalty with distance
        penalty = weight * (norm_dist ** 2)
        return penalty

    def _get_standard_coordinate_explanation(self):
        """
        Standard coordinate system explanation
        """
        return ("\nNote: Coordinates are shown as (x%, y%) where x% is the horizontal distance from left edge (0-100%) and y% is the vertical distance from top edge (0-100%). For example, (25%, 50%) means a point located 25% from the left side and 50% from the top of the image. SVF values at each point are compared using their exact values without rounding.")
    
    def _is_category_suitable(self, category_name):
        """
        Check if category is suitable for current image.
        Skips question generation if compatibility is poor.
        """
        if not hasattr(self, 'image_compatibility'):
            return True
        
        if category_name not in self.image_compatibility:
            return True
        
        return self.image_compatibility[category_name]['suitable']

    def _get_dynamic_ensure_max(self, category_name=None):
        """Always include maximum value in choices"""
        return True

    def _get_question_template(self, question_type, **kwargs):
        """
        Select a random question template
        """
        if question_type not in self.question_templates:
            return None
        
        templates = self.question_templates[question_type]
        selected_template = random.choice(templates)
        
        # Replace template parameters if provided
        if kwargs:
            try:
                selected_template = selected_template.format(**kwargs)
            except KeyError:
                pass
        
        return selected_template

    def _rephrase_with_gpt(self, question, category):
        """
        Rephrase question using GPT-4
        """
        if self._debug:
            tqdm_safe_print(f"[Debug] _rephrase_with_gpt called for category: {category}")
        
        if not self.use_gpt_rephrase or not self.openai_api_key:
            if self._debug:
                tqdm_safe_print("[Debug] GPT rephrasing skipped (not enabled or no API key).")
            return question, None, None
        
        try:
            # Randomly select rephrasing style
            style_options = ['casual', 'technical', 'simple', 'detailed']
            if random.random() < 0.3:
                style = 'perspective'
                context = random.choice(self.perspective_contexts)
                prompt = self.gpt_prompts[style].format(question=question, context=context)
            else:
                style = random.choice(style_options)
                prompt = self.gpt_prompts[style].format(question=question)
            
            headers = {
                'Authorization': f'Bearer {self.openai_api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': 'gpt-4o-mini',
                'messages': [
                    {
                        'role': 'system', 
                        'content': 'You are an expert in geospatial analysis and data visualization. Your task is to rephrase questions to be clear, concise, and unambiguous. The rephrased question must be objective and directly answerable from measurable data. Maintain the exact meaning of the original question. Do not introduce subjective language (e.g., "beautiful," "nice," "harmonious"). Preserve all technical terms (like "Sky View Factor," "SVF," "BCR") and coordinate formats (e.g., "(10%, 20%)") precisely.'
                    },
                    {
                        'role': 'user', 
                        'content': prompt
                    }
                ],
                'max_tokens': 150,
                'temperature': 0.7
            }
            
            try:
                response = requests.post(
                    'https://api.openai.com/v1/chat/completions',
                    headers=headers,
                    json=payload,
                    timeout=10
                )
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if self._debug:
                    print(f"[Debug] API connection error: {e}")
                return question, None, None
            
            if response.status_code == 200:
                result = response.json()
                rephrased = result['choices'][0]['message']['content'].strip()
                
                gpt_info = f"[GPT Rephrased (style: {style})]"
                if self._debug:
                    tqdm_safe_print(f"[GPT Rephrase] Original: {question}")
                    tqdm_safe_print(f"[GPT Rephrase] Style: {style}, Rephrased: {rephrased}")
                
                return rephrased, gpt_info, style
            else:
                if self._debug:
                    tqdm_safe_print(f"[GPT Rephrase] API Error: {response.status_code}")
                return question, None, None
                
        except Exception as e:
            if self._debug:
                tqdm_safe_print(f"[GPT Rephrase] Error: {e}")
            return question, None, None
    
    def _enhance_question_diversity(self, question_info, canonical_question):
        """
        質問の多様性を強化する統合メソッド
        """
        original_question = question_info.get('question', '')
        if self._debug:
            tqdm_safe_print(f"DEBUG: _enhance_question_diversity - original question length: {len(original_question)} chars")
            tqdm_safe_print(f"DEBUG: _enhance_question_diversity - first 100 chars: {original_question[:100]}...")
        
        # Step 2: GPT rephrasing (if enabled)
        enhanced_question, gpt_info, gpt_style = self._rephrase_with_gpt(original_question, canonical_question[0])
        
        if self._debug:
            tqdm_safe_print(f"DEBUG: _enhance_question_diversity - enhanced question length: {len(enhanced_question)} chars")
            tqdm_safe_print(f"DEBUG: _enhance_question_diversity - first 100 chars: {enhanced_question[:100]}...")
        
        # Step 3: Preserve question structure
        enhanced_question_dict = question_info.copy()
        enhanced_question_dict['question'] = enhanced_question
        
        if gpt_info and gpt_style:
            # Update category name (e.g., sun_exposure -> sun_exposure_gpt_casual)
            # Replace spaces in gpt_style with underscores (e.g., "perspective urban planning")
            new_category = f"{canonical_question[0]}_gpt_{gpt_style.replace(' ', '_')}"
            canonical_question[0] = new_category

            if 'debug_info' not in enhanced_question_dict:
                enhanced_question_dict['debug_info'] = []
            
            if not any(info.startswith("[GPT Rephrased") for info in enhanced_question_dict.get('debug_info', [])):
                 enhanced_question_dict['debug_info'].insert(0, gpt_info)

        return enhanced_question_dict

    def _process_file_path(self, file_path):
        if file_path:
            try:
                basename = os.path.basename(file_path)
                return os.path.splitext(basename)[0]
            except Exception as e:
                tqdm_safe_print(f"File path processing error: {e}")
                return "unknown"
        return "unknown"
    
    def _handle_error(self, error_type: ErrorType, message: str, 
                     context: Optional[Dict[str, Any]] = None,
                     exception: Optional[Exception] = None):
        """
        Unified error handling wrapper method
        
        Args:
            error_type: Error type enum
            message: Error message
            context: Additional context information
            exception: Exception object if available
            
        Returns:
            tuple: (None, None, None) - Standard return value for errors
        """
        import inspect
        # Get calling method name automatically
        frame = inspect.currentframe().f_back
        method_name = frame.f_code.co_name if frame else "unknown"
        
        if context is None:
            context = {}
        context["method"] = method_name
        
        return self.error_handler.handle_error(error_type, message, context, exception)

    def _get_relative_bbox(self, y_start, x_start, height, width):
        """Convert pixel coordinates to relative coordinates (0-100)"""
        h, w = self.svf_map.shape
        xmin = round((x_start / w) * 100)
        ymin = round((y_start / h) * 100)
        xmax = round(((x_start + width) / w) * 100)
        ymax = round(((y_start + height) / h) * 100)
        return [xmin, ymin, xmax, ymax]
        # return [ymin, xmin, ymax, xmax]

    def _get_relative_point(self, y, x):
        """Convert pixel coordinates to relative coordinates (0-100) with decimal precision"""
        h, w = self.svf_map.shape
        rel_x = round((x / w) * 100, 1)  # 小数点第一位まで
        rel_y = round((y / h) * 100, 1)  # 小数点第一位まで
        return [rel_x, rel_y]
    
    def _check_region_distance(self, center1, center2, min_distance=50.0):
        """Check if two region centers are far enough apart"""
        y1, x1 = center1
        y2, x2 = center2
        return ((y1 - y2) ** 2 + (x1 - x2) ** 2) ** 0.5 >= min_distance
        # return [rel_y, rel_x]

    # ========== REGION-BASED COMMON FUNCTIONS ==========
    
    def _create_region_based_choices(self, coords, scores, region_size, correct_idx, target_region=None):
        """
        Region-based質問の選択肢作成共通ロジック
        
        Args:
            coords: 地域座標のリスト
            scores: 各地域のスコア
            region_size: 地域のサイズ
            correct_idx: 正解のインデックス
            target_region: 正解地域（高度な用途）
            
        Returns:
            tuple: (regions, correct_answer)
                - regions: ラベル付きregionオブジェクトのリスト
                - correct_answer: 正解の文字列（例: "Region A"）
        """
        # Region objectsを作成
        regions = []
        for i, (coord, score) in enumerate(zip(coords, scores)):
            rel_bbox = self._get_relative_bbox(coord[0], coord[1], region_size, region_size)
            region = {
                'bbox': rel_bbox,
                'score': score,
                'is_correct': (i == correct_idx)
            }
            regions.append(region)
            
            # 正解地域を特定
            if region['is_correct']:
                correct_region = region
        
        regions = assign_balanced_labels(regions, correct_region)
        regions = sorted(regions, key=lambda x: x['label'])  # A, B, C, D順で表示
        
        # 正解の最終ラベルを特定
        correct_answer = f"Region {correct_region['label']}"
        
        return regions, correct_answer
    
    def _generate_grid_statistics(self, analysis_type='basic'):
        """
        3×3グリッド分析の共通ロジック
        
        Args:
            analysis_type: 分析タイプ ('basic', 'detailed', 'urban_density', 'sky_visibility')
            
        Returns:
            list: グリッドセル統計のリスト
        """
        h, w = self.svf_map.shape
        grid_h, grid_w = h // 3, w // 3
        
        # 各グリッドセルのSVF統計を計算
        grid_stats = []
        for i in range(3):
            for j in range(3):
                y_start = i * grid_h
                x_start = j * grid_w
                region = self.svf_map[y_start:y_start+grid_h, x_start:x_start+grid_w]
                valid_mask = ~np.isnan(region) & (region > 0)
                
                if np.sum(valid_mask) > 0:
                    mean_svf = np.mean(region[valid_mask])
                    std_svf = np.std(region[valid_mask]) if analysis_type in ['detailed'] else 0.0
                    
                    cell_stats = {
                        'position': (i, j),
                        'cell_name': self._get_grid_cell_name(i, j),
                        'mean_svf': mean_svf,
                        'bbox': self._get_relative_bbox(y_start, x_start, grid_h, grid_w)
                    }
                    
                    # 詳細分析の場合は標準偏差も追加
                    if analysis_type == 'detailed':
                        cell_stats['std_svf'] = std_svf
                    
                    # 特殊分析用の追加統計
                    if analysis_type in ['urban_density', 'sky_visibility']:
                        cell_stats['pixel_count'] = np.sum(valid_mask)
                        cell_stats['coverage_ratio'] = np.sum(valid_mask) / (grid_h * grid_w)
                    
                    grid_stats.append(cell_stats)
        
        return grid_stats
    
    def _get_grid_cell_name(self, row, col):
        """Convert grid position to cell name (e.g., 'top left', 'middle')"""
        positions = ['top', 'middle', 'bottom']
        positions2 = ['left', 'middle', 'right']
        return f"{positions[row]} {positions2[col]}"
    
    def _format_region_question(self, base_question, regions, include_coordinates=True):
        """
        Region-based質問の標準フォーマット
        
        Args:
            base_question: 基本質問文
            regions: regionオブジェクトのリスト
            include_coordinates: 座標説明を含むかどうか
            
        Returns:
            str: フォーマットされた質問文
        """
        question_text = base_question + "\n\n"
        
        # 各地域の情報を表示
        for region in regions:
            bbox = region['bbox']
            question_text += f"Region {region['label']}: [xmin={bbox[0]}%, ymin={bbox[1]}%, "
            question_text += f"xmax={bbox[2]}%, ymax={bbox[3]}%]\n"
        
        question_text += "\nPlease choose from:\n"
        for region in regions:
            question_text += f"Region {region['label']}\n"
        
        # 座標説明を追加
        if include_coordinates:
            question_text = self._add_coordinate_explanations(question_text, 'region')
        
        return question_text
    
    def _add_coordinate_explanations(self, question_text, format_type):
        """
        座標システム説明の標準化
        
        Args:
            question_text: 既存の質問文
            format_type: 'region', 'grid', 'point' のいずれか
            
        Returns:
            str: 説明が追加された質問文
        """
        if format_type == 'region':
            question_text += "Coordinate Guide: Each region shows [left%, top%, right%, bottom%] as percentage of image size.\n"
            question_text += "Think of the image like a map: [4%, 58%, 20%, 76%] means:\n"
            question_text += "• Start 4% from left edge, 58% down from top\n"
            question_text += "• End 20% from left edge, 76% down from top\n"
            question_text += "This creates a rectangular region in that area of the image."
            
        elif format_type == 'grid':
            question_text += "\nGRID EXPLANATION: The image is divided into a 3×3 grid like a tic-tac-toe board.\n"
            question_text += "- Top row: 'top left', 'top middle', 'top right'\n"
            question_text += "- Middle row: 'middle left', 'middle middle', 'middle right'\n"
            question_text += "- Bottom row: 'bottom left', 'bottom middle', 'bottom right'\n"
            question_text += "Each cell covers exactly 1/9 of the image area."
            
        elif format_type == 'point':
            question_text += "\nPoint coordinates show [x%, y%] as percentage of image dimensions.\n"
            question_text += "For example, Point (25.0%, 75.0%) is 25% from left edge and 75% down from top."
        
        return question_text
    
    def _generate_debug_info(self, choices, scores, detailed_scores=None, evaluation_coords=None):
        """
        Generate standardized debug information.
        
        Args:
            choices: List of choices
            scores: List of scores
            detailed_scores: Detailed scores (optional)
            evaluation_coords: Evaluation coordinates (optional)
            
        Returns:
            list: Debug information list
        """
        debug_info = []
        
        for i, (choice, score) in enumerate(zip(choices, scores)):
            base_info = f"Choice {i+1}: {choice} = Score {score:.3f}"
            
            if detailed_scores and len(detailed_scores) > i:
                base_info += f" | Details: {detailed_scores[i]}"
            
            debug_info.append(base_info)
        
        return debug_info
    
    def _create_question_info_structure(self, question_text, choices_data, debug_info, scores, enhance_category=None):
        """
        Create standardized question information structure.
        
        Args:
            question_text: Question text
            choices_data: Choice data (bboxes, coords, etc.)
            debug_info: Debug information
            scores: Score information
            enhance_category: GPT rephrasing category (optional)  
            
        Returns:
            dict: Standardized question information structure
        """
        question_info = {
            "question": question_text,
            "debug_info": debug_info,
            "scores": {i: float(score) for i, score in enumerate(scores)}
        }
        
        # Add appropriate keys from choices_data
        if isinstance(choices_data, dict):
            for key, value in choices_data.items():
                if key in ['choices_bboxes', 'choices_coords', 'choices']:
                    question_info[key] = value
        elif isinstance(choices_data, list):
            question_info['choices_bboxes'] = choices_data
        
        # Apply GPT rephrasing if enabled
        if enhance_category and hasattr(self, '_enhance_question_diversity'):
            question_info = self._enhance_question_diversity(question_info, enhance_category)
        
        return question_info
    
    # Placeholder for additional question generation methods
    
    

    def _apply_adaptive_weights(self):
        """
        Apply adaptive category weights based on image suitability.
        Problematic categories (building_density and best_landcover_balance) 
        get their weights adjusted based on image characteristics.
        """
        if self._debug:
            tqdm_safe_print("[Adaptive Weights] Checking image suitability for problematic categories...")
        
        # Check building_density suitability
        if 'building_density' in self.QUESTION_TYPES:
            is_suitable, reason = self._check_urban_density_suitability()
            if not is_suitable:
                # Set weight to 0 to skip this category entirely
                self.CATEGORY_WEIGHTS['building_density'] = 0.0
                # Also update the QUESTION_TYPES weight
                self.QUESTION_TYPES['building_density'][0] = 0.0
                if self._debug:
                    tqdm_safe_print(f"[Adaptive Weights] building_density: DISABLED - {reason}")
            else:
                # Increase weight for suitable images (was 3.0, now 4.0)
                self.CATEGORY_WEIGHTS['building_density'] = 4.0
                # Also update the QUESTION_TYPES weight
                self.QUESTION_TYPES['building_density'][0] = 4.0
                if self._debug:
                    tqdm_safe_print(f"[Adaptive Weights] building_density: ENHANCED (4.0x) - {reason}")
        
        # Check best_landcover_balance suitability
        if 'best_landcover_balance' in self.QUESTION_TYPES:
            is_suitable, reason = self._check_natural_artificial_suitability()
            if not is_suitable:
                # Set weight to 0 to skip this category entirely
                self.CATEGORY_WEIGHTS['best_landcover_balance'] = 0.0
                # Also update the QUESTION_TYPES weight
                self.QUESTION_TYPES['best_landcover_balance'][0] = 0.0
                if self._debug:
                    tqdm_safe_print(f"[Adaptive Weights] best_landcover_balance: DISABLED - {reason}")
            else:
                # Increase weight for suitable images (was 5.0, now 6.0)
                self.CATEGORY_WEIGHTS['best_landcover_balance'] = 6.0
                # Also update the QUESTION_TYPES weight
                self.QUESTION_TYPES['best_landcover_balance'][0] = 6.0
                if self._debug:
                    tqdm_safe_print(f"[Adaptive Weights] best_landcover_balance: ENHANCED (6.0x) - {reason}")

    def _check_urban_density_suitability(self):
        """Check if the image is suitable for urban density questions"""
        if self.height_map is None or self.segmentation_map is None:
            return False, "Missing required data (height_map or segmentation_map)"
        
        BUILDING_CLASS_ID = 10
        # GeoNRW landcover classes (see landcover_names):
        # 0: others, 1: forest, 2: water, 3: agricultural, 4: residential,
        # 5: grassland, 6: railways, 7: roads, 8: commercial,
        # 9: bare_soil, 10: buildings
        URBAN_AREA_CLASSES = [4, 6, 7, 8]  # residential, railways, roads, commercial
        
        # Natural vs. artificial masks aligned with landcover_names
        natural_classes = [1, 2, 3, 5, 9]  # forest, water, agricultural, grassland, bare_soil
        artificial_classes = [4, 6, 7, 8, 10]   # residential, railways, roads, commercial, buildings
        
        # Check overall building/urban coverage
        building_pixels = np.sum(self.segmentation_map == BUILDING_CLASS_ID)
        urban_pixels = np.sum(np.isin(self.segmentation_map, URBAN_AREA_CLASSES))
        natural_pixels = np.sum(np.isin(self.segmentation_map, natural_classes))
        artificial_pixels = np.sum(np.isin(self.segmentation_map, artificial_classes))
        total_pixels = self.segmentation_map.size
        
        building_ratio = building_pixels / total_pixels
        urban_ratio = urban_pixels / total_pixels
        natural_ratio = natural_pixels / total_pixels
        artificial_ratio = artificial_pixels / total_pixels
        total_urban_ratio = (building_pixels + urban_pixels) / total_pixels
        
        if total_urban_ratio < 0.15:
            return False, f"Insufficient urban content: {total_urban_ratio:.1%} (need >15%)"
        
        if natural_ratio > 0.9 and artificial_ratio < 0.15:
            return False, f"Predominantly natural area: natural {natural_ratio:.1%}, artificial {artificial_ratio:.1%}"
        
        return True, f"Urban content: {total_urban_ratio:.1%} (building: {building_ratio:.1%}, urban: {urban_ratio:.1%}, natural: {natural_ratio:.1%})"
    
    def urbanDensity(self):
        """都市の密集度を評価する質問
        カテゴリ定義と重み:
        都市密度 | 建物被覆率 (BCR) 0.4 | 床面積比 (FAR) 0.3 | SVF 0.15 | エッジ密度 0.1 | 平均輝度 0.05
        answer: 都市密度の高い領域の中心座標とBBox (y_start, x_start, height, width)
        前提: self.height_map と self.segmentation_map が必要
        """
        # Pre-check suitability
        is_suitable, reason = self._check_urban_density_suitability()
        if not is_suitable:
            if self._debug: tqdm_safe_print(f"[urbanDensity] Image not suitable: {reason}")
            return None, None, None

        if self.height_map is None or self.segmentation_map is None:
            if self._debug: tqdm_safe_print("[urbanDensity] Height map and Segmentation map are required.")
            return None, None, None

        h, w = self.svf_map.shape
        # Consider larger regions for density assessment, e.g., 1/5 to 1/3 of map dimension
        s_dim = min(h,w)
        min_region_size = max(20, s_dim // 5) # ensure min_region_size is at least 20
        max_region_size = max(min_region_size + 10, s_dim // 3) # ensure max_region_size is larger
        if min_region_size >= max_region_size : max_region_size = min_region_size +1
        
        region_size = random.randint(min_region_size, max_region_size)

        if h < region_size or w < region_size:
            if self._debug: tqdm_safe_print(f"[urbanDensity] Map size ({h}x{w}) too small for region size {region_size}.")
            return None, None, None

        valid_mask_svf = ~np.isnan(self.svf_map) & (self.svf_map > 0)
        
        # Efficiently find candidate regions using an integral image for valid SVF pixels
        integral_svf_mask = np.zeros((h + 1, w + 1), dtype=np.int32)
        integral_svf_mask[1:, 1:] = np.cumsum(np.cumsum(valid_mask_svf.astype(np.int32), axis=0), axis=1)

        candidate_regions_starts = []
        MIN_VALID_SVF_PIXEL_RATIO_IN_REGION = 0.6 # At least 60% of pixels in region must have valid SVF
        MIN_BUILDING_URBAN_RATIO_IN_REGION = 0.1 # At least 10% of region should be building/urban from seg map
        BUILDING_CLASS_ID = 10 # GeoNRW building class
        URBAN_AREA_CLASSES = [4, 6, 7, 8] # residential, railways, roads, commercial

        # Grid sampling for candidate region starting points
        step = max(1, region_size // 4)
        for r_y in range(0, h - region_size + 1, step):
            for r_x in range(0, w - region_size + 1, step):
                # SVF valid pixel check
                svf_valid_count = integral_svf_mask[r_y + region_size, r_x + region_size] - \
                                  integral_svf_mask[r_y + region_size, r_x] - \
                                  integral_svf_mask[r_y, r_x + region_size] + \
                                  integral_svf_mask[r_y, r_x]
                if svf_valid_count < (region_size * region_size * MIN_VALID_SVF_PIXEL_RATIO_IN_REGION):
                    continue

                # Segmentation map check for building/urban presence AND exclude pure natural areas
                region_seg_map = self.segmentation_map[r_y:r_y+region_size, r_x:r_x+region_size]
                building_pixels = np.sum(region_seg_map == BUILDING_CLASS_ID)
                urban_pixels = np.sum(np.isin(region_seg_map, URBAN_AREA_CLASSES))
                
                # Natural vs. artificial masks aligned with landcover_names
                natural_classes = [1, 2, 3, 5, 9]  # forest, water, agricultural, grassland, bare_soil
                artificial_classes = [4, 6, 7, 8, 10]   # residential, railways, roads, commercial, buildings
                
                natural_pixels = np.sum(np.isin(region_seg_map, natural_classes))
                artificial_pixels = np.sum(np.isin(region_seg_map, artificial_classes))
                total_pixels = region_size * region_size
                
                natural_ratio = natural_pixels / total_pixels
                artificial_ratio = artificial_pixels / total_pixels
                
                # Exclude pure natural areas (>90% natural, <15% artificial)
                if natural_ratio > 0.9 and artificial_ratio < 0.15:
                    continue
                
                # Require minimum building/urban ratio (15%)
                if (building_pixels + urban_pixels) < (region_size * region_size * 0.15):
                    continue
                
                candidate_regions_starts.append((r_y, r_x))

        if len(candidate_regions_starts) < 4:
            if self._debug: tqdm_safe_print(f"[urbanDensity] Not enough candidate regions found ({len(candidate_regions_starts)}). Needed 4.")
            return None, None, None

        # Select 4 regions
        num_to_select = 4
        selected_indices = np.random.choice(len(candidate_regions_starts), size=min(num_to_select, len(candidate_regions_starts)), replace=False)
        chosen_region_coords = [candidate_regions_starts[i] for i in selected_indices]

        urban_density_scores = []
        detailed_scores_for_debug = []

        for y_start, x_start in chosen_region_coords:
            # Extract regions from all maps
            region_svf = self.svf_map[y_start:y_start+region_size, x_start:x_start+region_size]
            region_height = self.height_map[y_start:y_start+region_size, x_start:x_start+region_size]
            region_seg = self.segmentation_map[y_start:y_start+region_size, x_start:x_start+region_size]
            region_rgb = self.rgb_image[y_start:y_start+region_size, x_start:x_start+region_size] if self.rgb_image is not None else None

            # --- Component Scores --- 
            # Natural vs. artificial masks aligned with landcover_names
            natural_classes = [1, 2, 3, 5, 9]  # forest, water, agricultural, grassland, bare_soil
            artificial_classes = [4, 6, 7, 8, 10]   # residential, railways, roads, commercial, buildings
            
            natural_pixels = np.sum(np.isin(region_seg, natural_classes))
            artificial_pixels = np.sum(np.isin(region_seg, artificial_classes))
            total_pixels = region_size * region_size
            natural_ratio = natural_pixels / total_pixels
            artificial_ratio = artificial_pixels / total_pixels
            is_natural_dominated = natural_ratio > 0.7
            
            # 1. SVF (lower is denser) - Weight 0.15
            valid_region_svf = region_svf[~np.isnan(region_svf) & (region_svf > 0)]
            mean_svf = np.mean(valid_region_svf) if len(valid_region_svf) > 0 else 0.5
            # Don't penalize SVF reduction in natural areas (forests reduce SVF but aren't urban density)
            if is_natural_dominated:
                svf_comp_score = 0.0
            else:
                svf_comp_score = (1.0 - mean_svf) * 0.15

            # 2. Building Coverage Ratio (BCR) - Weight 0.5
            building_mask = (region_seg == BUILDING_CLASS_ID)
            bcr = np.sum(building_mask) / (region_size * region_size)
            bcr_comp_score = bcr * 0.5

            # 3. Floor Area Ratio (FAR) - Weight 0.25
            # Simplified FAR: BCR * average building height in terms of floors
            # Assuming average floor height of 3-4 meters. Height map values need interpretation.
            # This is a very rough estimation. True FAR needs GIS data.
            # Height map values are in meters. Assuming 1 floor = 3.5m.
            FLOOR_HEIGHT_METERS = 3.5 
            avg_building_height_pixels = np.mean(region_height[building_mask]) if np.sum(building_mask) > 0 else 0
            avg_num_floors = avg_building_height_pixels / FLOOR_HEIGHT_METERS if FLOOR_HEIGHT_METERS > 0 else 0
            # Cap average number of floors to a reasonable maximum, e.g., 20, to prevent extreme FAR values from dominating
            avg_num_floors = min(avg_num_floors, 20) 
            far_estimated = bcr * avg_num_floors
            # Normalize FAR score (e.g., cap at FAR of 5.0 for max score component)
            far_comp_score = min(far_estimated / 5.0, 1.0) * 0.25 

            # 4. Edge Density (from RGB if available) - Weight 0.05
            edge_density_comp_score = 0
            edge_density_val = 0
            if region_rgb is not None and region_rgb.ndim == 3 and region_rgb.shape[2] >=1 :
                gray_region_rgb = np.mean(region_rgb, axis=2).astype(np.uint8) if region_rgb.shape[2] >=3 else region_rgb[:,:,0].astype(np.uint8)
                # Simple edge detection (Sobel or just diff)
                edges_x = np.abs(np.diff(gray_region_rgb.astype(float), axis=1))
                edges_y = np.abs(np.diff(gray_region_rgb.astype(float), axis=0))
                # Normalize edge magnitude, e.g., by dividing by max possible diff (255)
                # Then, consider pixels with significant edges.
                edge_threshold = 30 # Out of 255
                edge_density_val = (np.sum(edges_x > edge_threshold) + np.sum(edges_y > edge_threshold)) / (2 * region_size * (region_size-1) + 1e-6) # Normalize by number of possible edges
                edge_density_comp_score = min(edge_density_val / 0.5, 1.0) * 0.05 # Assuming 0.5 edge density is high

            # 5. Brightness (lower is denser/more shadowed) - Weight 0.05
            brightness_comp_score = 0
            brightness_val = 0.5 # Neutral if no RGB
            if region_rgb is not None and region_rgb.ndim == 3:
                brightness_val = np.mean(region_rgb) / 255.0 # Normalize to 0-1
                brightness_comp_score = (1.0 - brightness_val) * 0.05 
            
            total_density_score = svf_comp_score + bcr_comp_score + far_comp_score + edge_density_comp_score + brightness_comp_score
            urban_density_scores.append(total_density_score)
            detailed_scores_for_debug.append({
                "total": total_density_score, "svf_s": svf_comp_score, "bcr_s": bcr_comp_score, "far_s": far_comp_score, 
                "edge_s": edge_density_comp_score, "bright_s": brightness_comp_score,
                "svf_raw": mean_svf, "bcr_raw": bcr, "far_est_raw": far_estimated, "edge_raw": edge_density_val, "bright_raw": brightness_val
            })

        if not urban_density_scores:
             if self._debug: tqdm_safe_print(f"[urbanDensity] No scores were calculated.")
             return None, None, None

        # Select 4 regions based on scores for the question - we want one correct (highest density) and 3 distractors
        # For this question, we are asking "Which area has the highest urban density"
        # So, we need to identify the region with the max score as the correct answer.
        # The choices presented should be these 4 selected regions.

        # Combine region coords with their scores
        region_data_for_selection = list(zip(chosen_region_coords, urban_density_scores))
        
        # Custom bbox distance filtering for urban_density to prevent overlapping regions
        def filter_bbox_distance(candidates_scores, min_distance=50.0):
            """Filter candidates to maintain minimum bbox distance for urban_density"""
            if len(candidates_scores) <= 4:
                return candidates_scores
            
            # Sort by score to prioritize higher density regions
            sorted_candidates = sorted(candidates_scores, key=lambda x: x[1], reverse=True)
            
            selected = []
            for coord, score in sorted_candidates:
                # Check distance from already selected regions
                is_far_enough = True
                for sel_coord, sel_score in selected:
                    dist = ((coord[0] - sel_coord[0]) ** 2 + (coord[1] - sel_coord[1]) ** 2) ** 0.5
                    if dist < min_distance:
                        is_far_enough = False
                        break
                
                if is_far_enough:
                    selected.append((coord, score))
                    if len(selected) >= 8:  # Get more than needed for diversity selection
                        break
            
            # If we don't have enough after strict filtering, fall back to original
            if len(selected) < 4:
                return candidates_scores
            
            return selected
        
        # Apply bbox distance filtering first
        filtered_region_data = filter_bbox_distance(region_data_for_selection, min_distance=50.0)
        
        # Use select_choices_with_diversity to get 4 diverse regions, ensuring the max density is one of them.
        selected_regions_coords, selected_region_scores = select_choices_with_diversity(
            [item[0] for item in filtered_region_data],
            [item[1] for item in filtered_region_data],
            target_count=4,
            ensure_max=self._get_dynamic_ensure_max('urban_density'), # We want the highest density region
            min_score_gap=0.05 # Try for some difference in scores
        )

        if len(selected_regions_coords) < 4:
            if self._debug: tqdm_safe_print(f"[urbanDensity] Could not select 4 diverse regions for choices (got {len(selected_regions_coords)}).")
            return None, None, None

        # Create region objects with bbox information
        candidate_regions = []
        correct_region = None
        pre_shuffle_correct_idx = np.argmax(selected_region_scores)
        
        for i, (coord, score) in enumerate(zip(selected_regions_coords, selected_region_scores)):
            rel_bbox = self._get_relative_bbox(coord[0], coord[1], region_size, region_size)
            region_obj = {
                'bbox': rel_bbox,
                'score': score,
                'is_correct': (i == pre_shuffle_correct_idx)
            }
            candidate_regions.append(region_obj)
            if region_obj['is_correct']:
                correct_region = region_obj
        
        # Use assign_balanced_labels to assign A, B, C, D labels
        candidate_regions = assign_balanced_labels(candidate_regions, correct_region)
        candidate_regions = sorted(candidate_regions, key=lambda x: x['label'])  # A, B, C, D order
        
        # Find correct answer with new label
        correct_answer = f"Region {correct_region['label']}"
        
        choices_str_list = []
        choices_bboxes = []
        for region in candidate_regions:
            region_choice = f"Region {region['label']}"
            choices_str_list.append(region_choice)
            choices_bboxes.append(region['bbox'])
        
        base_question = self._get_question_template('building_density')
        if base_question is None:
            base_question = "Which region shows the most dense urban development?"
        
        question_text = f"{base_question}\nHint: Urban density is characterized by building coverage, building height, and less visible sky.\n"
        question_text += "Scoring method: Locations are scored solely based on Building Coverage Ratio (40%), Floor Area Ratio (30%), Sky View Factor penalty (15%), Edge Density (10%), and Brightness analysis (5%). Higher building coverage and lower sky visibility indicate higher urban density.\n"
        question_text += "Coordinate system: Each region is specified by (xmin, ymin, xmax, ymax) coordinates as percentages of the image dimensions, where (0, 0) is the top-left corner. This defines a rectangle from xmin to xmax horizontally and ymin to ymax vertically.\n\n"
        
        # Show region details with coordinates
        for region in candidate_regions:
            bbox = region['bbox']
            question_text += f"{region['label']}: [xmin={bbox[0]}%, ymin={bbox[1]}%, xmax={bbox[2]}%, ymax={bbox[3]}%]\n"
        
        question_text += "\nPlease choose from:\n"
        for region in candidate_regions:
            region_choice = f"Region {region['label']}"
            question_text += f"{region_choice}\n"
        
        # Find correct answer with new label
        answer_str = correct_answer
        
        # Debug info for each choice 
        debug_info_choices = []
        for i_choice, region in enumerate(candidate_regions):
            # Find the original detailed score for this region to show breakdown
            original_detailed_score = {}
            for i_orig, orig_coord in enumerate(chosen_region_coords): 
                # Convert region bbox back to coord for matching
                region_coord = (orig_coord[0], orig_coord[1])  # y_start, x_start
                rel_bbox = self._get_relative_bbox(region_coord[0], region_coord[1], region_size, region_size)
                if rel_bbox == region['bbox']: # Match found
                    original_detailed_score = detailed_scores_for_debug[i_orig]
                    break
            debug_info_choices.append(f"Choice {i_choice+1} {choices_str_list[i_choice]}: Score={region['score']:.3f} | Details: {original_detailed_score}")

        choices_info_q = {
            "question": question_text,
            "debug_info": debug_info_choices,
            "scores": {i: float(region['score']) for i, region in enumerate(candidate_regions)},
            "choices_bboxes": choices_bboxes
        }
        
        canonical_q_payload = ['urban_density']
        for region in candidate_regions:
            # Convert bbox back to pixel coordinates for canonical payload
            bbox = region['bbox']
            # Approximate pixel coordinates from relative bbox
            h, w = self.svf_map.shape
            y_start = int((bbox[1] / 100.0) * h)
            x_start = int((bbox[0] / 100.0) * w)
            canonical_q_payload.extend([y_start, x_start, region_size, region_size, float(region['score'])])

        if self._debug:
            # Find correct region for debug output
            correct_region_debug = next(r for r in candidate_regions if r['is_correct'])
            tqdm_safe_print(f"[urbanDensity] Correct answer: {answer_str} (Score: {correct_region_debug['score']:.3f})")
            for dbg_item in debug_info_choices:
                tqdm_safe_print(f"  {dbg_item}")
        
        choices_info_q = self._enhance_question_diversity(choices_info_q, canonical_q_payload)
        
        return choices_info_q, answer_str, canonical_q_payload

    def skyVisibility(self):
        """
         完全改修：空の見えやすさを評価する質問 (SVFと建物遮蔽ペナルティ)
        微小差座標除去・統一シャッフル・画像相性判別を実装
        """
        #  確率的難易度調整: 25%の質問のみ難しい条件を適用
        use_hard_conditions = random.random() < 0.25
        
        if use_hard_conditions:
            if self._debug: tqdm_safe_print("[skyVisibility] Using HARD conditions (25% chance)")
        else:
            if self._debug: tqdm_safe_print("[skyVisibility] Using STANDARD conditions (75% chance)")
        
        #  画像相性チェック
        if not self._is_category_suitable('sky_visibility'):
            return self._handle_error(
                error_type=ErrorType.VALIDATION_ERROR,
                message="Category not suitable for this image",
                context={"category": "sky_visibility"}
            )

        if self.segmentation_map is None:
            return self._handle_error(
                error_type=ErrorType.MISSING_REQUIRED_DATA,
                message="Segmentation map is required but not provided"
            )

        y_indices, x_indices = self._get_cached_candidate_points('all')
        if len(y_indices) < 20:
            return self._handle_error(
                error_type=ErrorType.INSUFFICIENT_DATA,
                message=f"Not enough valid SVF points ({len(y_indices)})",
                context={"required": 20, "actual": len(y_indices)}
            )
        
        # Sample a subset of points if too many, for efficiency
        image_area = self.svf_map.shape[0] * self.svf_map.shape[1]
        MAX_CANDIDATES_TO_EVALUATE = min(2000, max(800, image_area // 200))
        if len(y_indices) > MAX_CANDIDATES_TO_EVALUATE:
            # Use stratified sampling to ensure area diversity
            sample_indices = self._stratified_sampling(y_indices, x_indices, MAX_CANDIDATES_TO_EVALUATE)
            y_indices = y_indices[sample_indices]
            x_indices = x_indices[sample_indices]

        candidate_points = []
        skyview_scores = []
        raw_svf_values = [] # For debug
        building_penalties = [] # For debug

        # Building class ID : 10
        BUILDING_CLASS_ID = 10
        # Window size to check for nearby buildings
        WINDOW_SIZE = 5 # 5×5 window around the point
        HALF_WINDOW = WINDOW_SIZE // 2

        svf_values_batch = self.svf_map[y_indices, x_indices]
        svf_score_components_batch = svf_values_batch * 0.7

        for i, (y, x) in enumerate(zip(y_indices, x_indices)):
            svf_value = svf_values_batch[i]
            svf_score_component = svf_score_components_batch[i]

            # Calculate building penalty
            penalty_component = 0
            if self.segmentation_map is not None:
                y_min = max(0, y - HALF_WINDOW)
                y_max = min(self.segmentation_map.shape[0], y + HALF_WINDOW + 1)
                x_min = max(0, x - HALF_WINDOW)
                x_max = min(self.segmentation_map.shape[1], x + HALF_WINDOW + 1)
                
                local_seg_region = self.segmentation_map[y_min:y_max, x_min:x_max]
                
                # Normalize penalty for boundary effects (smaller window near edges)
                expected_size = WINDOW_SIZE * WINDOW_SIZE
                actual_size = local_seg_region.size
                
                if actual_size > 0:
                    building_pixels = np.sum(local_seg_region == BUILDING_CLASS_ID)
                    building_ratio_in_window = building_pixels / actual_size
                    
                    # Amplify penalty near image boundaries to match center evaluation
                    normalization_factor = expected_size / actual_size if actual_size > 0 else 1.0
                    
                    penalty_component = (building_ratio_in_window * 0.3) * normalization_factor
            
            # Adjust edge penalty based on difficulty
            if use_hard_conditions:
                edge_penalty_weight = 0.025
            else:
                edge_penalty_weight = 0.05
                
            # Bias correction: add penalty for points near image edges
            edge_penalty = self._calculate_edge_penalty(y, x, weight=edge_penalty_weight)
            
            # Total score: SVF score minus building penalty and edge penalty
            total_sky_visibility_score = svf_score_component - penalty_component - edge_penalty
            
            candidate_points.append((y, x))
            skyview_scores.append(total_sky_visibility_score)
            raw_svf_values.append(svf_value)
            building_penalties.append(penalty_component)

        if not candidate_points or len(candidate_points) < 4:
            return self._handle_error(
                error_type=ErrorType.INSUFFICIENT_DATA,
                message=f"Not enough candidates after evaluation ({len(candidate_points)})",
                context={"required": 4, "actual": len(candidate_points)}
            )

        # Remove coordinates that are too close together
        filtered_candidates = self._filter_minimum_distance_coordinates(candidate_points)
        if len(filtered_candidates) < 4:
            if self._debug: tqdm_safe_print(f"[skyVisibility] Not enough candidates after distance filtering ({len(filtered_candidates)}).")
            filtered_candidates = candidate_points

        coord_to_index = {coord: i for i, coord in enumerate(candidate_points)}
        
        filtered_scores = []
        filtered_raw_svf = []
        filtered_penalties = []
        for filtered_coord in filtered_candidates:
            idx = coord_to_index.get(filtered_coord)
            if idx is not None:
                filtered_scores.append(skyview_scores[idx])
                filtered_raw_svf.append(raw_svf_values[idx])
                filtered_penalties.append(building_penalties[idx])
            else:
                filtered_scores.append(0.0)
                filtered_raw_svf.append(0.0)
                filtered_penalties.append(0.0)

        # Generate choices with adaptive score gap
        choices_coords, choices_scores = select_choices_with_diversity(
            filtered_candidates,
            filtered_scores,
            target_count=4,
            ensure_max=self._get_dynamic_ensure_max('sky_visibility'), 
            min_score_gap=self._get_adaptive_score_gap('sky_visibility')
        )

        # Fallback: retry with correct-gap-prioritized selection
        if len(choices_coords) < 4:
            if self._debug: tqdm_safe_print(f"[skyVisibility] Fallback: trying correct-gap-prioritized selection (got {len(choices_coords)} initially).")
            choices_coords, choices_scores = select_choices_prioritizing_correct_gap(
                filtered_candidates,
                filtered_scores,
                target_count=4,
                ensure_max=True,
                min_correct_gap=0.03
            )
            
        if len(choices_coords) < 4:
            return self._handle_error(
                error_type=ErrorType.INSUFFICIENT_DATA,
                message=f"Could not select 4 diverse choices even with fallback (got {len(choices_coords)})",
                context={"required": 4, "actual": len(choices_coords)}
            )

        pre_shuffle_correct_idx = np.argmax(choices_scores)
        
        # Unified high-quality shuffle with correct answer tracking
        combined_for_shuffle = []
        for i, coord_choice in enumerate(choices_coords):
            try:
                original_idx = filtered_candidates.index(coord_choice)
                is_correct = (i == pre_shuffle_correct_idx)
                combined_for_shuffle.append((
                    coord_choice, 
                    choices_scores[i], 
                    filtered_raw_svf[original_idx], 
                    filtered_penalties[original_idx],
                    is_correct
                ))
            except ValueError:
                is_correct = (i == pre_shuffle_correct_idx)
                combined_for_shuffle.append((coord_choice, choices_scores[i], 0.0, 0.0, is_correct))
       
        combined_for_shuffle = bias_free_shuffle(combined_for_shuffle)
        
        shuffled_choices_coords = [item[0] for item in combined_for_shuffle]
        shuffled_scores = [item[1] for item in combined_for_shuffle]
        shuffled_svfs_for_choices = [item[2] for item in combined_for_shuffle]
        shuffled_penalties_for_choices = [item[3] for item in combined_for_shuffle]
        is_correct_flags = [item[4] for item in combined_for_shuffle]

        # Convert pixel coordinates to percentage coordinates
        choices_str = []
        for y, x in shuffled_choices_coords:
            rel_coords = self._get_relative_point(y, x)
            choices_str.append(f"({rel_coords[0]:.1f}%, {rel_coords[1]:.1f}%)")  # Display as (x.x%, y.y%)
        
        base_question = self._get_question_template('sky_visibility')
        if base_question is None:
            base_question = "Which location has the highest sky visibility? (Where can you see the most sky?)"
        
        question_text = base_question
        question_text += "\nHint: Areas with fewer surrounding obstacles allow more sky to be visible."
        question_text += "\nScoring method: Locations are scored solely based on Sky View Factor (70%) with a penalty for nearby buildings (30%). Higher scores indicate better sky visibility with fewer obstructions."
        question_text += self._get_standard_coordinate_explanation()
        question_text += "\nPlease choose from:"
        for coord_str_choice in choices_str:
            question_text += f"\n{coord_str_choice}"

        correct_idx = is_correct_flags.index(True)
        answer_str = choices_str[correct_idx]

        debug_info = []
        for i in range(len(shuffled_choices_coords)):
            debug_info.append(f"Point {i+1} {choices_str[i]}: TotalScore={shuffled_scores[i]:.3f} (SVF_raw={shuffled_svfs_for_choices[i]:.3f}*0.7 - Penalty={shuffled_penalties_for_choices[i]:.3f})")

        choices_info_question = {
            "question": question_text,
            "debug_info": debug_info,
            "scores": {i: float(s) for i, s in enumerate(shuffled_scores)}
        }
        
        canonical_question = ['sky_visibility'] + [float(s) for s in shuffled_scores]
        
        if self._debug:
            tqdm_safe_print(f"[skyVisibility] Correct answer: {answer_str} (Score: {shuffled_scores[correct_idx]:.3f})")
            for item in debug_info:
                 tqdm_safe_print(f"  Choice: {item}")

        choices_coords_percent = [self._get_relative_point(y, x) for y, x in shuffled_choices_coords]
        choices_info_question['choices_coords'] = choices_coords_percent
        
        choices_info_question = self._enhance_question_diversity(choices_info_question, canonical_question)

        return choices_info_question, answer_str, canonical_question

    def opennessAssessment(self):
        """空間の開放感を評価する質問
        カテゴリ定義と重み:
        開放感 | 正の Openness 指数 0.5 | 平均 SVF 0.25 | 建物密度(低) 0.15 | 地形平坦度 0.05 | 視覚的雑度(低) 0.05
        answer: 最も開放感のある領域の中心座標 (y,x)
        前提: self.height_map が必須。self.segmentation_map, self.rgb_image があればより良い。
        """
        # np.random.seed(self.seeds['openness_assessment'])
        # random.seed(self.seeds['openness_assessment'])

        if self.height_map is None: # Height map is crucial for Openness Index and terrain flatness
            if self._debug: tqdm_safe_print("[opennessAssessment] Height map is required.")
            return None, None, None

        h, w = self.svf_map.shape
        s_dim = min(h,w)
        min_region_size = max(20, s_dim // 6) 
        max_region_size = max(min_region_size + 10, s_dim // 4) 
        if min_region_size >= max_region_size : max_region_size = min_region_size + 1
        region_size = random.randint(min_region_size, max_region_size)

        if h < region_size or w < region_size:
            if self._debug: tqdm_safe_print(f"[opennessAssessment] Map size ({h}x{w}) too small for region size {region_size}.")
            return None, None, None

        valid_mask_svf = ~np.isnan(self.svf_map) & (self.svf_map > 0)
        valid_mask_height = ~np.isnan(self.height_map)
        
        candidate_region_starts = []
        MAX_CANDIDATE_REGIONS = 50 # Try to find up to 50 candidate regions
        MIN_VALID_PIXEL_RATIO = 0.7 # At least 70% valid pixels in a region

        # Grid sampling for candidate region starting points
        step = max(1, region_size // 3)
        for r_y in range(0, h - region_size + 1, step):
            if len(candidate_region_starts) >= MAX_CANDIDATE_REGIONS * 2: # Generate a bit more to sample from
                break
            for r_x in range(0, w - region_size + 1, step):
                if len(candidate_region_starts) >= MAX_CANDIDATE_REGIONS * 2:
                    break
                # Check valid SVF and Height pixels in the region
                region_svf_mask = valid_mask_svf[r_y:r_y+region_size, r_x:r_x+region_size]
                region_height_mask = valid_mask_height[r_y:r_y+region_size, r_x:r_x+region_size]
                
                if (np.sum(region_svf_mask) / region_svf_mask.size) >= MIN_VALID_PIXEL_RATIO and \
                   (np.sum(region_height_mask) / region_height_mask.size) >= MIN_VALID_PIXEL_RATIO:
                    candidate_region_starts.append((r_y, r_x))
        
        if len(candidate_region_starts) < 4:
            if self._debug: tqdm_safe_print(f"[opennessAssessment] Not enough candidate regions with valid data ({len(candidate_region_starts)}).")
            return None, None, None

        # Randomly select a subset of candidate regions to evaluate fully (e.g. 10-15 regions)
        num_regions_to_evaluate = min(len(candidate_region_starts), max(4, MAX_CANDIDATE_REGIONS // 2))
        selected_eval_indices = np.random.choice(len(candidate_region_starts), num_regions_to_evaluate, replace=False)
        regions_to_evaluate_coords = [candidate_region_starts[i] for i in selected_eval_indices]

        openness_scores = []
        detailed_scores_for_debug = []
        BUILDING_CLASS_ID = 10 
        for y_start, x_start in regions_to_evaluate_coords:
            region_svf_map = self.svf_map[y_start:y_start+region_size, x_start:x_start+region_size]
            region_hgt_map = self.height_map[y_start:y_start+region_size, x_start:x_start+region_size]
            
            valid_svf_in_region = region_svf_map[~np.isnan(region_svf_map) & (region_svf_map > 0)]
            valid_hgt_in_region = region_hgt_map[~np.isnan(region_hgt_map)]

            if len(valid_svf_in_region) == 0 or len(valid_hgt_in_region) == 0:
                continue # Should not happen due to earlier checks, but as safeguard

            # 1. Mean SVF (Weight 0.25)
            mean_svf = np.mean(valid_svf_in_region)
            svf_score = mean_svf * 0.25

            # 2. Openness Index (Weight 0.5)
            # Simplified: Positive Openness = svf_mean + 0.5 * svf_std (from notes for openness_assessment)
            # Another interpretation of Openness (Yokoyama) is related to DEM angles, which is complex.
            # Using the provided hint: svf_mean + 0.5 * svf_std
            svf_std = np.std(valid_svf_in_region)
            # Normalize openness_index to be roughly 0-1. Max possible svf_mean is 1, max svf_std is around 0.5.
            # So raw index can be up to 1 + 0.5*0.5 = 1.25. Divide by this to normalize. 
            openness_index_raw = mean_svf + 0.5 * svf_std
            openness_index_score = (openness_index_raw / 1.25) * 0.5 
            
            # 3. Building Density (Low is good) (Weight 0.15)
            building_density_score = 0.15 # Max score if no segmentation map
            building_ratio_raw = 0.0
            if self.segmentation_map is not None:
                region_seg_map = self.segmentation_map[y_start:y_start+region_size, x_start:x_start+region_size]
                building_pixels = np.sum(region_seg_map == BUILDING_CLASS_ID)
                building_ratio_raw = building_pixels / region_seg_map.size
                building_density_score = (1.0 - building_ratio_raw) * 0.15
            
            # 4. Terrain Flatness (Weight 0.05)
            # Low std dev of height map = flat
            height_std = np.std(valid_hgt_in_region)
            # Normalize: e.g. std_dev < 1m is very flat (score 1), std_dev 10m is not flat (score 0)
            # Using exp decay: exp(-height_std / reference_std_for_flatness)
            terrain_flatness_raw = np.exp(-height_std / 5.0) # 5m std dev as a reference for significant non-flatness
            terrain_flatness_score = terrain_flatness_raw * 0.05

            # 5. Visual Simplicity (Low visual clutter is good) (Weight 0.05)
            visual_simplicity_score = 0.05 # Max score if no RGB image
            visual_complexity_raw = 1.0 # Max complexity if no RGB
            if self.rgb_image is not None:
                region_rgb_img = self.rgb_image[y_start:y_start+region_size, x_start:x_start+region_size]
                if region_rgb_img.ndim == 3 and region_rgb_img.shape[2] >=1:
                    gray_rgb = np.mean(region_rgb_img, axis=2).astype(np.uint8) if region_rgb_img.shape[2] >=3 else region_rgb_img[:,:,0].astype(np.uint8)
                    # Edge variance as a measure of visual clutter (higher variance = more clutter)
                    edges_x = np.abs(np.diff(gray_rgb.astype(float), axis=1))
                    edges_y = np.abs(np.diff(gray_rgb.astype(float), axis=0))
                    edge_variance = (np.var(edges_x) + np.var(edges_y)) / 2 
                    # Normalize: e.g. low variance (<100) is simple, high variance (>1000) is cluttered
                    # Simplicity = 1 / (1 + normalized_variance)
                    visual_complexity_raw = edge_variance
                    # map variance from say 0-2000 to 1-0 for simplicity score
                    # A higher edge variance might mean more clutter, so we want lower variance for simplicity.
                    # assume low variance (e.g. < 100) is very simple (score 1), high (e.g. > 1000) is complex (score 0)
                    # scaled_variance = min(edge_variance / 1000.0, 1.0) # Cap at 1000 for normalization
                    # visual_simplicity_score = (1.0 - scaled_variance) * 0.05
                    # Using exp decay for simplicity: e.g. exp(-variance / ref_variance)
                    visual_simplicity_raw = np.exp(-edge_variance / 500.0) # 500 as ref variance for clutter
                    visual_simplicity_score = visual_simplicity_raw * 0.05

            total_openness_score = svf_score + openness_index_score + building_density_score + terrain_flatness_score + visual_simplicity_score
            openness_scores.append(total_openness_score)
            detailed_scores_for_debug.append({
                "total": total_openness_score, "svf_s": svf_score, "open_idx_s": openness_index_score,
                "bldg_dens_s": building_density_score, "terrain_flat_s": terrain_flatness_score, "vis_simp_s": visual_simplicity_score,
                "svf_m_raw": mean_svf, "svf_std_raw": svf_std, "bldg_ratio_raw": building_ratio_raw, 
                "hgt_std_raw": height_std, "vis_ phức tạp_raw": visual_complexity_raw
            })

        if len(openness_scores) < 4:
            if self._debug: tqdm_safe_print(f"[opennessAssessment] Not enough regions evaluated to form 4 choices ({len(openness_scores)}).")
            return None, None, None

        # Select 4 diverse choices from evaluated regions, ensuring max score is included
        choices_region_starts, choices_total_scores = select_choices_with_diversity(
            regions_to_evaluate_coords,
            openness_scores,
            target_count=4,
            ensure_max=self._get_dynamic_ensure_max('openness_assessment'),
            min_score_gap=0.03 # Smaller gap as scores are combined from many factors
        )

        # フォールバック：4つ未満の場合は正解重視モードで再試行
        if len(choices_region_starts) < 4:
            if self._debug: tqdm_safe_print(f"[opennessAssessment] Fallback: trying correct-gap-prioritized selection (got {len(choices_region_starts)} initially).")
            choices_region_starts, choices_total_scores = select_choices_prioritizing_correct_gap(
                regions_to_evaluate_coords,
                openness_scores,
                target_count=4,
                ensure_max=True,
                min_correct_gap=0.02
            )
            
        if len(choices_region_starts) < 4:
            if self._debug: tqdm_safe_print(f"[opennessAssessment] Could not select 4 diverse regions for choices even with fallback (got {len(choices_region_starts)}).")
            return None, None, None

        # Shuffle for presentation
        final_choices_combined = list(zip(choices_region_starts, choices_total_scores))
        final_choices_combined = bias_free_shuffle(final_choices_combined)
        shuffled_final_coords, shuffled_final_scores = zip(*final_choices_combined)

        highest_idx = np.argmax(shuffled_final_scores)
        correct_idx = highest_idx
        highest_region = {'coord': shuffled_final_coords[highest_idx], 'score': shuffled_final_scores[highest_idx]}
        
        # Create region data structures
        regions = []
        for i, (coord, score) in enumerate(zip(shuffled_final_coords, shuffled_final_scores)):
            rel_bbox = self._get_relative_bbox(coord[0], coord[1], region_size, region_size)
            region = {
                'coord': coord,
                'score': score,
                'bbox': rel_bbox,
                'is_correct': (i == highest_idx)
            }
            regions.append(region)
        
        # Identify correct region
        correct_region = next(r for r in regions if r['is_correct'])
        
        regions = assign_balanced_labels(regions, correct_region)
        regions = sorted(regions, key=lambda x: x['label'])
        
        choices_str_list = []
        choices_bboxes = []
        for region in regions:
            rel_bbox = region['bbox']
            choices_bboxes.append(rel_bbox)
            choices_str_list.append(f"Region {region['label']}: [{rel_bbox[0]}%, {rel_bbox[1]}%, {rel_bbox[2]}%, {rel_bbox[3]}%]")
            
        base_question = self._get_question_template('spatial_openness')
        if base_question is None:
            base_question = "Which location has the most open space? (Where is the most spacious area?)"
        
        question_text = base_question
        question_text += "\nHint: Open areas usually have good sky visibility and fewer surrounding obstacles."
        question_text += "\nScoring method: Locations are scored solely based on spatial openness index (50%), mean Sky View Factor (25%), low building density (15%), terrain flatness (5%), and visual simplicity (5%). Higher scores indicate more spacious and open areas."
        question_text += "\nCoordinate system: Each region is specified by (xmin, ymin, xmax, ymax) coordinates as percentages of the image dimensions, where (0, 0) is the top-left corner. This defines a rectangle from xmin to xmax horizontally and ymin to ymax vertically."
        question_text += "\nPlease choose from:"
        for choice_s in choices_str_list:
            question_text += f"\n{choice_s}"
        
        # Identify correct answer with new label
        correct_region_after_labeling = next(r for r in regions if r['is_correct'])
        answer_str = f"Region {correct_region_after_labeling['label']}"

        debug_info_choices = []
        for i_choice, r_coord_choice in enumerate(shuffled_final_coords):
            original_detailed_score = {}
            for i_orig, orig_coord_eval in enumerate(regions_to_evaluate_coords):
                if orig_coord_eval == r_coord_choice:
                    original_detailed_score = detailed_scores_for_debug[i_orig]
                    break
            debug_info_choices.append(f"Choice {i_choice+1} {choices_str_list[i_choice]}: Score={shuffled_final_scores[i_choice]:.3f} | Details: {original_detailed_score}")

        choices_info_q = {
            "question": question_text,
            "debug_info": debug_info_choices,
            "scores": {i: float(s) for i, s in enumerate(shuffled_final_scores)},
            "choices_bboxes": choices_bboxes
        }
        
        canonical_q_payload = ['openness_assessment']
        for r_coord_c, r_score_c in zip(shuffled_final_coords, shuffled_final_scores):
            # Store region start and size, and score
            canonical_q_payload.extend([int(r_coord_c[0]), int(r_coord_c[1]), region_size, region_size, float(r_score_c)])

        if self._debug:
            correct_label_idx = ord(correct_region_after_labeling['label']) - ord('A')
            correct_score = shuffled_final_scores[correct_label_idx] if correct_label_idx < len(shuffled_final_scores) else 0.0
            tqdm_safe_print(f"[opennessAssessment] Correct answer: {answer_str} (Score: {correct_score:.3f})")
            for dbg_item in debug_info_choices:
                tqdm_safe_print(f"  {dbg_item}")
        
        choices_info_q = self._enhance_question_diversity(choices_info_q, canonical_q_payload)
        
        return choices_info_q, answer_str, canonical_q_payload

    def visibilityRange(self):
        """見通しの良さを評価する質問
        カテゴリ定義と重み:
        見通しの良さ | ビューシェッド最大距離(近似) 0.6 | 平均 SVF 0.25 | 地形起伏(粗度) 0.15
        answer: 見通しが最も良い地点の座標 (y,x)
        前提: self.height_map が必須。
        """
        # np.random.seed(self.seeds['visibility_range'])
        # random.seed(self.seeds['visibility_range'])

        if self.height_map is None:
            if self._debug: tqdm_safe_print("[visibilityRange] Height map is required.")
            return None, None, None

        h, w = self.svf_map.shape
        # Evaluate points, not regions, but consider a window for SVF and roughness around the point.
        # Window size for local context (e.g., for SVF mean and local terrain roughness)
        context_window_size = random.randint(15, 25) # Odd number is good for centering
        if context_window_size % 2 == 0: context_window_size +=1
        half_ctx_win = context_window_size // 2

        # Max distance for viewshed calculation (optimized: 1/8 instead of 1/4)
        max_view_dist_pixels = min(h, w) // 8  # Reduced from //4 to //8 
        if max_view_dist_pixels < 10: # ensure it's a reasonable distance
            if self._debug: tqdm_safe_print(f"[visibilityRange] Map too small for meaningful viewshed (max_view_dist_pixels={max_view_dist_pixels})")
            return None, None, None

        valid_mask_svf = ~np.isnan(self.svf_map) & (self.svf_map > 0)
        valid_mask_height = ~np.isnan(self.height_map)
        # Points for evaluation must have valid height and be far enough from borders for context window and viewshed
        # For simplicity, sample from points that allow full context window and viewshed checks up to a certain distance
        
        candidate_points_coords = []
        y_indices, x_indices = np.where(valid_mask_height & valid_mask_svf)

        # Filter points to ensure they are not too close to border for viewshed/context
        # Simplification: ensure center point itself is valid, viewshed will handle boundaries
        # Context window will also handle boundaries.
        
        if len(y_indices) < 20: # Need enough points to sample from
            if self._debug: tqdm_safe_print(f"[visibilityRange] Not enough valid points on map ({len(y_indices)}).")
            return None, None, None

        # Sample a subset of points for evaluation to save computation
        num_points_to_evaluate = min(len(y_indices), 200) # Evaluate up to 200 points (optimized)
        if len(y_indices) > num_points_to_evaluate:
            # Use stratified sampling to ensure area diversity
            sample_indices = self._stratified_sampling(y_indices, x_indices, num_points_to_evaluate)
        else:
            sample_indices = np.arange(len(y_indices))
        
        points_for_evaluation = []
        for idx in sample_indices:
            py, px = y_indices[idx], x_indices[idx]
            # Ensure point allows context window extraction without going too much out of bounds
            if py >= half_ctx_win and py < h - half_ctx_win and px >= half_ctx_win and px < w - half_ctx_win:
                 points_for_evaluation.append((py,px))
        
        if len(points_for_evaluation) < 4:
            if self._debug: tqdm_safe_print(f"[visibilityRange] Not enough suitable points for evaluation after border filtering ({len(points_for_evaluation)}).")
            return None, None, None

        visibility_scores = []
        detailed_scores_for_debug = []

        for p_y, p_x in points_for_evaluation:
            center_height = self.height_map[p_y, p_x]

            # 1. Viewshed Max Distance (Simplified) - Weight 0.6
            # Simplified: Average max visible distance in N directions. Max distance capped by max_view_dist_pixels.
            num_directions = 8
            total_visible_distance_sum = 0
            raw_distances_in_dirs = []
            
            directions = []
            for i_dir in range(num_directions):
                angle_rad = (i_dir / num_directions) * 2 * math.pi
                directions.append((math.cos(angle_rad), math.sin(angle_rad)))

            for i_dir in range(num_directions):
                cos_angle, sin_angle = directions[i_dir]
                max_dist_in_dir = 0
                # prev_pixel_height_on_los = center_height # LOS starts from viewer height

                step_size = max(1, max_view_dist_pixels // 50)
                
                for dist in range(step_size, max_view_dist_pixels + 1, step_size):
                    curr_target_x = int(round(p_x + dist * cos_angle))
                    curr_target_y = int(round(p_y + dist * sin_angle))

                    if not (0 <= curr_target_x < w and 0 <= curr_target_y < h):
                        break
                    if not valid_mask_height[curr_target_y, curr_target_x]:
                        break

                    target_ground_height = self.height_map[curr_target_y, curr_target_x]
                    visible = True
                    
                    # Check intermediate points along line of sight for early termination
                    for k_step in range(step_size, dist, step_size):
                        inter_x = int(round(p_x + k_step * cos_angle))
                        inter_y = int(round(p_y + k_step * sin_angle))
                        
                        if not (0 <= inter_x < w and 0 <= inter_y < h):
                            visible = False
                            break
                        
                        if not valid_mask_height[inter_y, inter_x]:
                            visible = False
                            break
                        
                        inter_ground_height = self.height_map[inter_y, inter_x]
                        los_height_at_k = center_height + (target_ground_height - center_height) * (k_step / dist)
                        if inter_ground_height > los_height_at_k + 1.0:
                            visible = False
                            break
                    
                    if visible:
                        max_dist_in_dir = dist
                    else:
                        break
                
                total_visible_distance_sum += max_dist_in_dir
                raw_distances_in_dirs.append(max_dist_in_dir)
            
            avg_max_view_dist = total_visible_distance_sum / num_directions
            # Normalize by the theoretical max possible (max_view_dist_pixels)
            viewshed_score_component = (avg_max_view_dist / max_view_dist_pixels) * 0.6

            # 2. Mean SVF in context window - Weight 0.25
            ctx_svf_region = self.svf_map[p_y-half_ctx_win : p_y+half_ctx_win+1, p_x-half_ctx_win : p_x+half_ctx_win+1]
            valid_ctx_svf = ctx_svf_region[~np.isnan(ctx_svf_region) & (ctx_svf_region > 0)]
            mean_svf_in_ctx = np.mean(valid_ctx_svf) if len(valid_ctx_svf) > 0 else 0
            svf_score_component = mean_svf_in_ctx * 0.25

            # 3. Terrain Roughness/Variation in context window - Weight 0.15
            # (Moderate roughness can be good for interesting views, too flat or too chaotic might be bad)
            ctx_hgt_region = self.height_map[p_y-half_ctx_win : p_y+half_ctx_win+1, p_x-half_ctx_win : p_x+half_ctx_win+1]
            valid_ctx_hgt = ctx_hgt_region[~np.isnan(ctx_hgt_region)]
            terrain_std_dev = np.std(valid_ctx_hgt) if len(valid_ctx_hgt) > 1 else 0
            # Score higher for moderate std dev (e.g., 5-15m), lower for very flat or very high std dev.
            # Using a Gaussian-like curve peaking at, say, 10m std dev.
            # exp(- (x - mu)^2 / (2 * sigma^2) )
            optimal_roughness_std = 10.0 # meters
            roughness_sensitivity = 10.0 # controls width of the peak
            terrain_roughness_factor = np.exp(-((terrain_std_dev - optimal_roughness_std)**2) / (2 * roughness_sensitivity**2))
            terrain_score_component = terrain_roughness_factor * 0.15

            total_visibility_score = viewshed_score_component + svf_score_component + terrain_score_component
            visibility_scores.append(total_visibility_score)
            detailed_scores_for_debug.append({
                "total_vis_score": total_visibility_score, "viewshed_s": viewshed_score_component, "svf_s": svf_score_component, "terrain_s": terrain_score_component,
                "avg_view_dist_raw": avg_max_view_dist, "mean_svf_ctx_raw": mean_svf_in_ctx, "terrain_std_raw": terrain_std_dev, "raw_view_dists": raw_distances_in_dirs
            })
            
        if len(visibility_scores) < 4:
            if self._debug: tqdm_safe_print(f"[visibilityRange] Not enough points evaluated to form 4 choices ({len(visibility_scores)}).")
            return None, None, None

        choices_coords, choices_scores = select_choices_with_diversity(
            points_for_evaluation,
            visibility_scores,
            target_count=4,
            ensure_max=self._get_dynamic_ensure_max('visibility_range'),
            min_score_gap=0.03 # Scores are combined, so gap might be smaller
        )

        # フォールバック：4つ未満の場合は正解重視モードで再試行
        if len(choices_coords) < 4:
            if self._debug: tqdm_safe_print(f"[visibilityRange] Fallback: trying correct-gap-prioritized selection (got {len(choices_coords)} initially).")
            choices_coords, choices_scores = select_choices_prioritizing_correct_gap(
                points_for_evaluation,
                visibility_scores,
                target_count=4,
                ensure_max=True,
                min_correct_gap=0.02
            )
            
        if len(choices_coords) < 4:
            if self._debug: tqdm_safe_print(f"[visibilityRange] Could not select 4 diverse points for choices even with fallback (got {len(choices_coords)}).")
            return None, None, None

        final_combined = list(zip(choices_coords, choices_scores))
        final_combined = bias_free_shuffle(final_combined)
        shuffled_final_coords, shuffled_final_scores = zip(*final_combined)

        # Convert pixel coordinates to percentage coordinates
        choices_str_list = []
        for c in shuffled_final_coords:
            rel_coords = self._get_relative_point(c[0], c[1])
            choices_str_list.append(f"Point ({rel_coords[0]:.1f}%, {rel_coords[1]:.1f}%)")  # Display as (x.x%, y.y%)

        base_question = self._get_question_template('visibility_range')
        if base_question is None:
            base_question = "Which location offers the highest visibility range? (Where can you see the furthest?)"
        
        question_text = base_question
        question_text += "\nHint: Areas with good visibility typically have high sky view factor and fewer obstacles in the line of sight."
        question_text += "\nScoring method: Locations are scored solely based on viewshed distance analysis (60%), Sky View Factor (25%), and terrain roughness variation (15%). Higher scores indicate better visibility range with longer line of sight distances."
        question_text += "\nCoordinate system: Each point is specified by (x, y) coordinates as percentages of the image dimensions, where (0, 0) is the top-left corner. 'x' represents the horizontal position (from left to right), and 'y' represents the vertical position (from top to bottom)."
        question_text += "\nPlease choose from:"
        for choice_s in choices_str_list:
            question_text += f"\n{choice_s}"

        correct_idx = np.argmax(shuffled_final_scores)
        answer_str = choices_str_list[correct_idx]

        debug_info_choices = []
        for i_choice, coord_choice in enumerate(shuffled_final_coords):
            original_detailed_score = {}
            for i_orig, orig_eval_coord in enumerate(points_for_evaluation):
                if orig_eval_coord == coord_choice:
                    original_detailed_score = detailed_scores_for_debug[i_orig]
                    break
            debug_info_choices.append(f"Choice {i_choice+1} {choices_str_list[i_choice]}: Score={shuffled_final_scores[i_choice]:.3f} | Details: {original_detailed_score}")

        choices_info_q = {
            "question": question_text,
            "debug_info": debug_info_choices,
            "scores": {i: float(s) for i, s in enumerate(shuffled_final_scores)}
        }
        
        canonical_q_payload = ['visibility_range'] 
        for coord_c, score_c in zip(shuffled_final_coords, shuffled_final_scores):
            canonical_q_payload.extend([int(coord_c[0]), int(coord_c[1]), float(score_c)])

        if self._debug:
            tqdm_safe_print(f"[visibilityRange] Correct answer: {answer_str} (Score: {shuffled_final_scores[correct_idx]:.3f})")
            for dbg_item in debug_info_choices:
                tqdm_safe_print(f"  {dbg_item}")

        choices_coords_percent = [self._get_relative_point(y, x) for y, x in shuffled_final_coords]
        choices_info_q['choices_coords'] = choices_coords_percent
        
        choices_info_q = self._enhance_question_diversity(choices_info_q, canonical_q_payload)

        return choices_info_q, answer_str, canonical_q_payload
    
    def chooseQuestionsToAsk(self, number_question=50):
        """Chooses questions to ask based on available data and weights."""
        available_q_types = list(self.QUESTION_TYPES.keys())
        if not available_q_types:
            tqdm_safe_print("No question types available based on the provided maps.")
            return []

        # balanced_categoriesモードの場合
        if self.balanced_categories:
            return self._chooseQuestionsBalanced(number_question, available_q_types)

        # 通常の質問とハード質問を分離
        normal_q_types = [q for q in available_q_types if not q.startswith('hard_')]
        hard_q_types = [q for q in available_q_types if q.startswith('hard_')]
        
        # ハード質問の割合を決定（全体の20-30%程度）
        hard_ratio = self.hard_ratio
        num_hard_questions = max(1, int(number_question * hard_ratio)) if hard_q_types else 0
        if hard_ratio == 0.0:
            num_hard_questions = 0
        elif hard_ratio == 1.0:
            num_hard_questions = number_question
        num_normal_questions = number_question - num_hard_questions
        
        if self._debug:
            tqdm_safe_print(f"Generating {num_normal_questions} normal questions and {num_hard_questions} hard questions")

        # Generate normal questions
        normal_chosen_types = self._choose_question_types(normal_q_types, num_normal_questions)
        
        # Generate hard questions (after normal questions)
        hard_chosen_types = self._choose_question_types(hard_q_types, num_hard_questions) if hard_q_types else []
        
        # Combine normal and hard questions preserving order
        chosen_types_with_methods = normal_chosen_types + hard_chosen_types

        self.questions = []
        self.answers = []
        self.canonical_questions = []
        
        self._generate_questions(chosen_types_with_methods, number_question)
        
        if self._debug:
            normal_count = sum(1 for q in self.canonical_questions if not (q and len(q) > 0 and str(q[0]).startswith('hard_')))
            hard_count = sum(1 for q in self.canonical_questions if q and len(q) > 0 and str(q[0]).startswith('hard_'))
            tqdm_safe_print(f"Final distribution: {normal_count} normal questions, {hard_count} hard questions")
        
        return self.questions, self.answers, self.canonical_questions

    def _chooseQuestionsBalanced(self, number_question, available_q_types):
        """Select questions in balanced distribution mode."""
        if self._debug:
            tqdm_safe_print(f" Balanced distribution mode: Distributing {number_question} questions across {len(available_q_types)} categories")
        
        # Separate normal and hard questions
        normal_q_types = [q for q in available_q_types if not q.startswith('hard_')]
        hard_q_types = [q for q in available_q_types if q.startswith('hard_')]
        
        # Determine number of hard questions
        hard_ratio = self.hard_ratio
        num_hard_questions = max(1, int(number_question * hard_ratio)) if hard_q_types else 0
        if hard_ratio == 0.0:
            num_hard_questions = 0
        elif hard_ratio == 1.0:
            num_hard_questions = number_question
        num_normal_questions = number_question - num_hard_questions
        
        chosen_types_with_methods = []
        
        # Balanced distribution for normal questions
        if num_normal_questions > 0 and normal_q_types:
            base_questions_per_category = max(1, num_normal_questions // len(normal_q_types))
            remaining_questions = num_normal_questions - (base_questions_per_category * len(normal_q_types))
            
            for q_type in normal_q_types:
                for _ in range(base_questions_per_category):
                    chosen_types_with_methods.append((q_type, self.QUESTION_TYPES[q_type][1], self.QUESTION_TYPES[q_type][2]))
            
            # Distribute remaining questions to high-priority categories
            if remaining_questions > 0:
                weighted_types = [(q_type, self.QUESTION_TYPES[q_type][0]) for q_type in normal_q_types]
                weighted_types.sort(key=lambda x: x[1], reverse=True)
                
                for i in range(remaining_questions):
                    q_type = weighted_types[i % len(weighted_types)][0]
                    chosen_types_with_methods.append((q_type, self.QUESTION_TYPES[q_type][1], self.QUESTION_TYPES[q_type][2]))
        
        # Balanced distribution for hard questions
        if num_hard_questions > 0 and hard_q_types:
            base_hard_per_category = max(1, num_hard_questions // len(hard_q_types))
            remaining_hard = num_hard_questions - (base_hard_per_category * len(hard_q_types))
            
            for q_type in hard_q_types:
                for _ in range(base_hard_per_category):
                    chosen_types_with_methods.append((q_type, self.QUESTION_TYPES[q_type][1], self.QUESTION_TYPES[q_type][2]))
            
            if remaining_hard > 0:
                weighted_hard_types = [(q_type, self.QUESTION_TYPES[q_type][0]) for q_type in hard_q_types]
                weighted_hard_types.sort(key=lambda x: x[1], reverse=True)
                
                for i in range(remaining_hard):
                    q_type = weighted_hard_types[i % len(weighted_hard_types)][0]
                    chosen_types_with_methods.append((q_type, self.QUESTION_TYPES[q_type][1], self.QUESTION_TYPES[q_type][2]))
        
        # Shuffle to remove bias
        chosen_types_with_methods = bias_free_shuffle(chosen_types_with_methods)
        
        if self._debug:
            from collections import Counter
            category_counts = Counter([item[0] for item in chosen_types_with_methods])
            tqdm_safe_print(f" Category distribution result: {dict(category_counts)}")
        
        self._generate_questions(chosen_types_with_methods, number_question)
        
        return self.questions, self.answers, self.canonical_questions

    def _choose_question_types(self, q_types, target_count):
        """Select target number of questions from specified question types."""
        if not q_types or target_count <= 0:
            return []
            
        # Weights for weighted random sampling
        weights = [self.QUESTION_TYPES[q][0] for q in q_types]
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights] if total_weight > 0 else [1.0/len(weights)] * len(weights)
        
        chosen_types_with_methods = []
        if len(q_types) <= target_count:
            # If fewer types available than requested, use all of them first
            chosen_types_with_methods = [(q_type, self.QUESTION_TYPES[q_type][1], self.QUESTION_TYPES[q_type][2]) for q_type in q_types]
            
            # Then add duplicates with different seed variations to reach the requested number
            remaining_questions = target_count - len(q_types)
            seed_variation = 1
            while remaining_questions > 0:
                for q_type in q_types:
                    if remaining_questions <= 0:
                        break
                    chosen_types_with_methods.append((q_type, self.QUESTION_TYPES[q_type][1], self.QUESTION_TYPES[q_type][2], seed_variation))
                    remaining_questions -= 1
                seed_variation += 1
            
            # Shuffle to ensure variety
            chosen_types_with_methods = bias_free_shuffle(chosen_types_with_methods)
        else:
            # Sample question types based on probabilities, ensuring no duplicates
            chosen_q_indices = []
            remaining_indices = list(range(len(q_types)))
            remaining_probs = probabilities.copy()
            
            remaining_indices = bias_free_shuffle(remaining_indices)
            
            while len(chosen_q_indices) < target_count and remaining_indices:
                # Normalize remaining probabilities
                total_remaining = sum(remaining_probs)
                if total_remaining > 0:
                    remaining_probs = [p / total_remaining for p in remaining_probs]
                else:
                    remaining_probs = [1.0/len(remaining_indices)] * len(remaining_indices)
                
                # Choose next question type
                next_idx = np.random.choice(len(remaining_indices), p=remaining_probs)
                chosen_q_indices.append(remaining_indices[next_idx])
                
                # Remove chosen index and its probability
                remaining_indices.pop(next_idx)
                remaining_probs.pop(next_idx)
            
            # Create 3-element tuples for the sampled question types
            chosen_types_with_methods = [(q_types[i], self.QUESTION_TYPES[q_types[i]][1], self.QUESTION_TYPES[q_types[i]][2]) for i in chosen_q_indices]
            
            # Add duplicates with seed variations if we need more questions
            if len(chosen_types_with_methods) < target_count:
                remaining_questions = target_count - len(chosen_types_with_methods)
                seed_variation = 1
                original_methods = chosen_types_with_methods.copy()
                
                while remaining_questions > 0:
                    for q_type, method_func, is_region_based in original_methods:
                        if remaining_questions <= 0:
                            break
                        chosen_types_with_methods.append((q_type, method_func, is_region_based, seed_variation))
                        remaining_questions -= 1
                    seed_variation += 1
                
                # Shuffle to ensure variety
                chosen_types_with_methods = bias_free_shuffle(chosen_types_with_methods)
        
        return chosen_types_with_methods

    def _generate_questions(self, chosen_types_with_methods, number_question):
        """選択された質問タイプから質問を生成"""
        # Track question type counts to ensure balanced distribution
        available_q_types = list(self.QUESTION_TYPES.keys())
        question_type_counts = {q_type: 0 for q_type in available_q_types}
        max_per_type = max(1, len(chosen_types_with_methods) // max(1, len(available_q_types)) + 2)  # Allow some flexibility

        for i, item in enumerate(chosen_types_with_methods):
            if len(item) == 4:  # Has seed variation
                q_type, method_func, is_region_based, seed_variation = item
            else:
                q_type, method_func, is_region_based = item
            
            # Skip if this question type already has too many questions
            if question_type_counts[q_type] >= max_per_type:
                if self._debug:
                    tqdm_safe_print(f"Skipping {q_type} due to max count reached ({question_type_counts[q_type]}/{max_per_type})")
                continue
            
            if self._debug:
                tqdm_safe_print(f"\nGenerating question type: {q_type}")

            # Check for map dependencies again before calling method, just in case
            maps_ok = True
            if q_type in ['building_density', 'spatial_openness', 'visibility_range'] and self.height_map is None:
                maps_ok = False
            if q_type in ['building_density', 'sky_visibility', 'best_landcover_balance'] and self.segmentation_map is None:
                # best_landcover_balance can fallback to RGB, so special check here
                if q_type == 'best_landcover_balance' and self.rgb_image is not None:
                    pass # it's okay
                else:
                    maps_ok = False
            if q_type == 'scenic_quality' and (self.segmentation_map is None or self.height_map is None or self.rgb_image is None):
                maps_ok = False
            
            # Check dependencies for hard questions
            if q_type.startswith('hard_'):
                if q_type in ['hard_urban_analysis', 'hard_scenic_analysis', 'hard_openness_analysis']:
                    if self.height_map is None or self.segmentation_map is None:
                        maps_ok = False
                elif q_type == 'hard_pixel_from_re':
                    pass
            
            if not maps_ok:
                if self._debug:
                    tqdm_safe_print(f"Skipping {q_type} due to missing maps.")
                continue

            # Try multiple attempts for categories with high weights
            category_weight = self.QUESTION_TYPES[q_type][0]
            max_attempts = 3 if category_weight >= 3.0 else 1  # More attempts for important categories
            
            success = False
            for attempt in range(max_attempts):
                try:
                    if self._debug:
                        import time
                        category_start_time = time.time()
                    
                    result = method_func()
                    
                    if self._debug:
                        category_end_time = time.time()
                        category_elapsed_time = category_end_time - category_start_time
                        
                        if q_type not in self.category_timing:
                            self.category_timing[q_type] = []
                        self.category_timing[q_type].append(category_elapsed_time)
                        
                        if category_elapsed_time > 3.0:
                            tqdm_safe_print(f"⏱  Slow category detected: {q_type} ({category_elapsed_time:.2f}s)")
                    if result and len(result) == 3:
                        question, answer, canonical_question = result
                        if question is not None and answer is not None:
                            # Add coordinate system explanation if it's a choice-based question with coordinates
                            # or if it's a region-based question
                            needs_coord_explanation = False
                            if isinstance(question, dict) and 'choices' in question: # Multiple choice questions
                                for choice_text in question['choices']:
                                    if isinstance(choice_text, str) and ('(' in choice_text and ')' in choice_text and ',' in choice_text) or "Region" in choice_text:
                                        needs_coord_explanation = True
                                        break
                            elif isinstance(question, str): # Direct string questions
                                if ('(' in question and ')' in question and ',' in question) or "Region" in question or "point" in question.lower():
                                    needs_coord_explanation = True
                            
                            # if needs_coord_explanation or is_region_based:
                            # if isinstance(question, dict):
                            #     question["question"] = add_short_instruction(question["question"])
                            # else:
                            #     question = add_short_instruction(question)

                            self.questions.append(question)
                            self.answers.append(answer)
                            self.canonical_questions.append(canonical_question)
                            question_type_counts[q_type] += 1  # Increment count for this question type
                            success = True
                            
                            if self._debug:
                                category_type = "HARD" if q_type.startswith('hard_') else "NORMAL"
                                attempt_info = f" (attempt {attempt+1})" if attempt > 0 else ""
                                tqdm_safe_print(f"Generated {category_type} question for {q_type}{attempt_info}: {question}")
                                tqdm_safe_print(f"Answer: {answer}")
                            break
                        elif self._debug:
                            tqdm_safe_print(f"Method for {q_type} returned None or incomplete result (attempt {attempt+1}).")
                except Exception as e:
                    if attempt == max_attempts - 1:  # Last attempt
                        tqdm_safe_print(f"Error generating question for {q_type} after {max_attempts} attempts: {e}")
                        if self._debug:
                            import traceback
                            tqdm_safe_print(traceback.format_exc())
                    elif self._debug:
                        tqdm_safe_print(f"Attempt {attempt+1} failed for {q_type}: {e}, retrying...")
            
            if not success and self._debug:
                tqdm_safe_print(f"Failed to generate any question for {q_type} after {max_attempts} attempts")
        
        # Post-generation balancing for high-priority categories
        self._balance_high_priority_categories(chosen_types_with_methods, question_type_counts, target_total=number_question)
        
        if self._debug:
            tqdm_safe_print(f"Final question type distribution: {question_type_counts}")
            
            # Display error summary if errors occurred
            if self.error_handler.has_errors():
                error_summary = self.error_handler.get_error_summary()
                tqdm_safe_print("\n=== Error Summary ===")
                tqdm_safe_print(f"Total errors: {error_summary['total_errors']}")
                for error_type, count in error_summary['error_counts'].items():
                    tqdm_safe_print(f"  {error_type}: {count}")
        
        return self.questions, self.answers, self.canonical_questions
    
    def _balance_high_priority_categories(self, chosen_types_with_methods, question_type_counts, target_total):
        """Ensure high-priority categories get adequate representation"""
        # Categories with weight >= 3.0 should have at least 1 question each if they were selected
        high_priority_categories = {q_type: weight for q_type, (weight, _, _) in self.QUESTION_TYPES.items() if weight >= 3.0}
        
        # Find selected high-priority categories that have 0 questions
        selected_categories = [item[0] for item in chosen_types_with_methods]
        missing_high_priority = [cat for cat in high_priority_categories if cat in selected_categories and question_type_counts[cat] == 0]
        
        if missing_high_priority and len(self.questions) < target_total:
            if self._debug:
                tqdm_safe_print(f"Attempting to generate missing high-priority categories: {missing_high_priority}")
            
            # Try to generate questions for missing high-priority categories
            additional_slots = target_total - len(self.questions)
            for cat in missing_high_priority[:additional_slots]:
                method_func = self.QUESTION_TYPES[cat][1]
                is_region_based = self.QUESTION_TYPES[cat][2]
                
                # More aggressive retry for missing high-priority categories
                for attempt in range(5):  # Up to 5 attempts
                    try:
                        result = method_func()
                        if result and len(result) == 3:
                            question, answer, canonical_question = result
                            if question is not None and answer is not None:
                                self.questions.append(question)
                                self.answers.append(answer)
                                self.canonical_questions.append(canonical_question)
                                question_type_counts[cat] += 1
                                
                                if self._debug:
                                    tqdm_safe_print(f"Successfully generated missing high-priority question for {cat} (attempt {attempt+1})")
                                break
                    except Exception as e:
                        if self._debug and attempt == 4:  # Last attempt
                            tqdm_safe_print(f"Final attempt failed for high-priority category {cat}: {e}")

    def _calculate_natural_coverage(self, region_seg):
        """Calculate natural element coverage (adapted for GeoNRW)."""
        if region_seg is None or region_seg.size == 0:
            return 0.0
        
        # Natural classes aligned with landcover_names (GeoNRW):
        # 1: forest, 2: water, 3: agricultural, 5: grassland, 9: bare_soil
        NATURAL_CLASSES = [1, 2, 3, 5, 9]
        
        total_pixels = region_seg.size
        background_pixels = np.sum(region_seg == 0)
        valid_pixels = total_pixels - background_pixels
        
        if valid_pixels <= 0:
            return 0.0
        
        natural_pixels = np.sum(np.isin(region_seg, NATURAL_CLASSES))
        return natural_pixels / valid_pixels

    

    def _calculate_balance_penalty(self, natural_ratio):
        """Apply penalty for extreme ratios."""
        # High score in 0.2-0.8 range, penalty as ratio approaches 0 or 1
        if natural_ratio < 0.2 or natural_ratio > 0.8:
            return 0.5
        return 1.0

    def debug_coordinate_info(self, coord_type, pixel_coords, percent_coords=None):
        """Output coordinate information for debugging."""
        if self._debug:
            tqdm_safe_print(f"[{coord_type}] Pixel coordinates: {pixel_coords}")
            if percent_coords:
                debug_msg += f" -> Percent coordinates: (x={percent_coords[0]}%, y={percent_coords[1]}%)"
            # tqdm_safe_print(debug_msg)
        elif coord_type == "region":
            debug_msg = f"[{coord_type}] Pixel coordinates: (y={pixel_coords[0]}, x={pixel_coords[1]}, h={pixel_coords[2]}, w={pixel_coords[3]})"
            if percent_coords:
                debug_msg += f" -> Percent coordinates: (x_min={percent_coords[0]}%, y_min={percent_coords[1]}%, x_max={percent_coords[2]}%, y_max={percent_coords[3]}%)"
            tqdm_safe_print(debug_msg)
        else:
            tqdm_safe_print(f"[{coord_type}] Unknown coordinate type")

    def landcoverType(self):
        """Landcover type analysis - identify land use types in the image."""
        if self.segmentation_map is None:
            if self._debug: tqdm_safe_print("[landcoverType] Segmentation map not available.")
            return None, None, None
            
        # Get unique landcover classes
        unique_classes = np.unique(self.segmentation_map)
        landcover_names = {
            0: 'others', 1: 'forest', 2: 'water', 3: 'agricultural', 
            4: 'residential', 5: 'grassland', 6: 'railways', 
            7: 'roads', 8: 'commercial', 9: 'bare_soil', 10: 'buildings'
        }
        
        present_types = [landcover_names.get(cls, f'class_{cls}') for cls in unique_classes if cls in landcover_names]
        
        if len(present_types) < 2:
            if self._debug: tqdm_safe_print("[landcoverType] Not enough landcover types found.")
            return None, None, None
            
        base_question = self._get_question_template('landcover_type')
        if base_question is None:
            base_question = "Which are land-use types are there in this image?"
        base_question += "Please choose all the land-use types from the following options: " + ", ".join(landcover_names.values())
        answer = ", ".join(present_types)
        
        question_info = {
            "question": base_question,
            "landcover_types": present_types
        }
        
        return question_info, answer, ['landcover_type', present_types]

    def landUse(self):
        """Land use analysis - identify top 3 land use patterns."""
        if self.segmentation_map is None:
            if self._debug: tqdm_safe_print("[landUse] Segmentation map not available.")
            return None, None, None
            
        # Calculate area percentages for each class
        unique_classes, counts = np.unique(self.segmentation_map, return_counts=True)
        total_pixels = self.segmentation_map.size
        
        landcover_names = {
            0: 'others', 1: 'forest', 2: 'water', 3: 'agricultural', 
            4: 'residential', 5: 'grassland', 6: 'railways', 
            7: 'roads', 8: 'commercial', 9: 'bare_soil', 10: 'buildings'
        }
        
        # Sort by counts to get top 3 land uses
        sorted_indices = np.argsort(counts)[::-1]  # Sort in descending order
        top_3_indices = sorted_indices[:3]  # Get top 3
        
        top_3_landuses = []
        top_3_percentages = []
        
        for idx in top_3_indices:
            class_id = unique_classes[idx]
            percentage = (counts[idx] / total_pixels) * 100
            landuse_name = landcover_names.get(class_id, f'class_{class_id}')
            top_3_landuses.append(landuse_name)
            top_3_percentages.append(percentage)
        
        base_question = self._get_question_template('land_use')
        if base_question is None:
            base_question = "What are the top 3 land uses in this area?"
        
        # Ranking format commented out - use simple top 3 selection instead
        # base_question += "Please choose top 3 types from the following options: " + ", ".join(landcover_names.values()) + ", and rank them by their total coverage area. Please answer in the format of '1st: type, 2nd: type, 3rd: type'"
        # answer_parts = []
        # for i, landuse in enumerate(top_3_landuses):
        #     rank = ["1st", "2nd", "3rd"][i]
        #     answer_parts.append(f"{rank}: {landuse}")
        # answer = ", ".join(answer_parts)
        
        base_question += "Please choose top 3 types from the following options: " + ", ".join(landcover_names.values())
        base_question += "\nAnswer format: type1, type2, type3"
        answer = ", ".join(top_3_landuses)
        
        question_info = {
            "question": base_question,
            "top_3_landuses": top_3_landuses,
            "top_3_percentages": top_3_percentages
        }
        
        return question_info, answer, ['land_use', top_3_landuses, top_3_percentages]

    def heightAverage(self):
        """Calculate average height value for a specific region - similar to hardPixel."""
        if self.height_map is None:
            if self._debug: tqdm_safe_print("[heightAverage] Height map is required.")
            return None, None, None
            
        h, w = self.height_map.shape
        
        # Region size calculation (similar to hardPixel)
        min_region_size = max(10, min(h, w) // 20)
        max_region_size = max(min_region_size + 5, min(h, w) // 7)
        region_size = random.randint(min_region_size, max_region_size)
        
        if h < region_size or w < region_size:
            if self._debug: tqdm_safe_print(f"[heightAverage] Map size ({h}x{w}) too small for region size {region_size}.")
            return None, None, None

        # Random position selection
        y_start = random.randint(0, h - region_size)
        x_start = random.randint(0, w - region_size)
        
        # Extract region from height map
        region_height = self.height_map[y_start:y_start+region_size, x_start:x_start+region_size]
        
        # Validity check - 70% of pixels must be valid
        valid_mask = ~np.isnan(region_height) & (region_height >= 0)
        if np.sum(valid_mask) < region_size * region_size * 0.7:
            if self._debug: tqdm_safe_print("[heightAverage] Not enough valid pixels in selected region.")
            return None, None, None
        
        # Calculate average height value
        avg_height_value = np.mean(region_height[valid_mask])
        
        # Round to nearest 10m for consistency
        rounded_height = round(avg_height_value / 10) * 10
        
        # Convert to relative coordinates
        rel_bbox = self._get_relative_bbox(y_start, x_start, region_size, region_size)
        region_str = f"[{rel_bbox[0]}%, {rel_bbox[1]}%, {rel_bbox[2]}%, {rel_bbox[3]}%]"
        
        # Get question template
        base_question = self._get_question_template('height_average')
        if base_question is None:
            base_question = "What is the average height value for the region {region}?"
        
        # Format question with region coordinates
        question_text = base_question.format(region=region_str)
        
        question_text += f"\nNote: The coordinates are given as percentages of the image dimensions in [xmin%, ymin%, xmax%, ymax%] format."
        question_text += f"\nRegion size: {region_size}×{region_size} pixels"
        question_text += f"\n\nPlease answer in 10-meter increments. Answer format: X m"

        answer_str = f"{rounded_height:.0f} m"

        # Debug information
        debug_info = [
            f"Region: ({y_start}, {x_start}) -> {region_str}",
            f"Region size: {region_size}×{region_size} pixels",
            f"Valid pixels: {np.sum(valid_mask)}/{region_size*region_size} ({np.sum(valid_mask)/(region_size*region_size)*100:.1f}%)",
            f"Average height value: {avg_height_value:.6f}",
            f"Rounded height: {rounded_height:.0f}m",
            f"Answer: {answer_str}"
        ]
        
        question_info = {
            "question": question_text,
            "debug_info": debug_info,
            "avg_height_value": float(avg_height_value),
            "rounded_height": float(rounded_height),
            "region_bbox": rel_bbox,
            "region_size": region_size,
            "valid_pixel_count": int(np.sum(valid_mask))
        }
        
        canonical_question = ['height_average', rel_bbox, float(rounded_height), region_size]
        
        if self._debug:
            tqdm_safe_print(f"[heightAverage] Correct answer: {answer_str} for region {region_str}")
            tqdm_safe_print(f"[heightAverage] Raw average: {avg_height_value:.3f}m, Rounded: {rounded_height:.0f}m")
            tqdm_safe_print(f"[heightAverage] Region size: {region_size}×{region_size}, Valid pixels: {np.sum(valid_mask)}")

        return question_info, answer_str, canonical_question


    def highestRegion(self):
        """Highest region analysis - identify region with highest average elevation."""
        if self.height_map is None:
            if self._debug: tqdm_safe_print("[highestRegion] Height map not available.")
            return None, None, None
            
        valid_mask = ~np.isnan(self.height_map) & (self.height_map > 0)
        if np.sum(valid_mask) == 0:
            if self._debug: tqdm_safe_print("[highestRegion] No valid height data.")
            return None, None, None
            
        h, w = self.height_map.shape
        region_size = min(h, w) // 6  # 少し大きめの地域サイズ
        
        # まず画像全体の統計を計算
        global_max_height = np.max(self.height_map[valid_mask])
        global_mean_height = np.mean(self.height_map[valid_mask])
        
        # 複数の地域候補を作成
        candidate_regions = []
        region_labels = ['Region A', 'Region B', 'Region C', 'Region D']
        
        candidate_regions = []
        attempts = 0
        max_attempts = 100
        
        # Generate multiple candidate regions randomly
        while len(candidate_regions) < 4 and attempts < max_attempts:
            attempts += 1
            
            rand_y = random.randint(0, max(0, h - region_size))
            rand_x = random.randint(0, max(0, w - region_size))
            
            # Check for overlap with existing regions
            overlaps = False
            for existing in candidate_regions:
                if (abs(rand_y - existing['y_start']) < region_size//2 and
                    abs(rand_x - existing['x_start']) < region_size//2):
                    overlaps = True
                    break
            if overlaps:
                continue
            
            # Calculate region elevation
            bbox = self._get_relative_bbox(rand_y, rand_x, region_size, region_size)
            region_heights = self.height_map[rand_y:rand_y+region_size, rand_x:rand_x+region_size]
            region_valid_mask = ~np.isnan(region_heights) & (region_heights > 0)
            
            if np.sum(region_valid_mask) > region_size * region_size * 0.3:
                mean_height_in_region = np.mean(region_heights[region_valid_mask])
                max_height_in_region = np.max(region_heights[region_valid_mask])
                rounded_mean_height = round(mean_height_in_region / 5) * 5
                
                candidate_regions.append({
                    'label': None,
                    'mean_height': mean_height_in_region,
                    'max_height': max_height_in_region,
                    'rounded_mean_height': rounded_mean_height,
                    'bbox': bbox,
                    'y_start': rand_y,
                    'x_start': rand_x,
                    'y_end': rand_y + region_size,
                    'x_end': rand_x + region_size,
                    'is_correct': False
                })
        
        if len(candidate_regions) < 4:
            if self._debug: tqdm_safe_print("[highestRegion] Could not generate enough diverse regions.")
            return None, None, None
        
        # Select region with highest average elevation as correct answer
        correct_region = max(candidate_regions, key=lambda x: x['mean_height'])
        correct_region['is_correct'] = True
        
        all_regions = candidate_regions
        correct_region = next(r for r in all_regions if r['is_correct'])
        
        all_regions = assign_balanced_labels(all_regions, correct_region)
        all_regions = sorted(all_regions, key=lambda x: x['label'])
        
        correct_answer = f"Region {correct_region['label']}"
        candidate_regions = all_regions
        
        base_question = self._get_question_template('highest_region')
        if base_question is None:
            base_question = "Which region has the highest average elevation?"
            
        question_text = f"{base_question}\n\n"
        for region in candidate_regions:
            bbox = region['bbox']
            question_text += f"{region['label']}: [xmin={bbox[0]}%, ymin={bbox[1]}%, xmax={bbox[2]}%, ymax={bbox[3]}%]\n"
        
        question_text += f"\nPlease choose from:\n"
        choices_list = []
        for region in candidate_regions:
            region_choice = f"Region {region['label']}"
            question_text += f"{region_choice}\n"
            choices_list.append(region_choice)
        
        question_text += "Coordinate Guide: Each region shows [left%, top%, right%, bottom%] as percentage of image size.\n"
        question_text += "Think of the image like a map: [4%, 58%, 20%, 76%] means:\n"
        question_text += "• Start 4% from left edge, 58% down from top\n"
        question_text += "• End 20% from left edge, 76% down from top\n"
        question_text += "This creates a rectangular region in that area of the image."
        
        pre_shuffle_correct_idx = next(i for i, r in enumerate(candidate_regions) if r['is_correct'])
        
        region_heights = [r['rounded_mean_height'] for r in candidate_regions]
        
        debug_info_choices = []
        for i_choice, region in enumerate(candidate_regions):
            status = "CORRECT" if region['is_correct'] else "wrong"
            debug_info_choices.append(f"Choice {i_choice+1} {region['label']}: AvgHeight={region['rounded_mean_height']}m (bbox: {region['bbox']}) {status}")

        question_info = {
            "question": question_text,
            "avg_height": correct_region['rounded_mean_height'],
            "bbox": correct_region['bbox'],
            "region_choices": choices_list,
            "choices_bboxes": [r['bbox'] for r in candidate_regions],
            "region_details": {region['label']: region['rounded_mean_height'] for region in candidate_regions},
            "choices": choices_list,
            "scores": {i: float(r['rounded_mean_height']) for i, r in enumerate(candidate_regions)},
            "debug_info": debug_info_choices
        }
        
        if self._debug:
            tqdm_safe_print(f"[highestRegion] Global max height: {global_max_height:.1f}m, Global mean: {global_mean_height:.1f}m")
            tqdm_safe_print(f"[highestRegion] Candidate regions:")
            for dbg_item in debug_info_choices:
                tqdm_safe_print(f"  {dbg_item}")
            correct_idx = pre_shuffle_correct_idx
            correct_height = region_heights[correct_idx]
            tqdm_safe_print(f"  Correct answer: {correct_answer} (AvgHeight: {correct_height}m)")
        
        return question_info, correct_answer, ['highest_region', correct_region['bbox'], correct_region['rounded_mean_height']]

    def sunExposure(self):
        """
        Region-based sun exposure analysis - identify region with highest average SVF
        """
        h, w = self.svf_map.shape
        
        # 4つの地域を動的に生成（バイアスフリー）
        candidate_regions = []
        region_labels = ['Region A', 'Region B', 'Region C', 'Region D']
        
        # **完全ランダム地域配置（重複防止付き）**
        for i, label in enumerate(region_labels):
            max_attempts = 50  # 最大試行回数
            
            for attempt in range(max_attempts):
                # ランダムな位置とサイズ
                region_size = random.randint(h//6, h//4)  # 可変サイズ
                center_y = random.randint(region_size//2, h - region_size//2)
                center_x = random.randint(region_size//2, w - region_size//2)
                
                # 既存地域との距離チェック
                is_valid_position = True
                for existing_region in candidate_regions:
                    existing_center = existing_region['position']
                    if not self._check_region_distance((center_y, center_x), existing_center):
                        is_valid_position = False
                        break
                
                if is_valid_position:
                    y_start = max(0, center_y - region_size//2)
                    x_start = max(0, center_x - region_size//2)
                    y_end = min(h, y_start + region_size)
                    x_end = min(w, x_start + region_size)
                    
                    # 地域のSVF統計を計算
                    region_svf = self.svf_map[y_start:y_end, x_start:x_end]
                    valid_mask = ~np.isnan(region_svf) & (region_svf > 0)
                    
                    if np.sum(valid_mask) > 0:
                        mean_svf = np.mean(region_svf[valid_mask])
                        bbox = self._get_relative_bbox(y_start, x_start, y_end-y_start, x_end-x_start)
                        
                        candidate_regions.append({
                            'label': label,
                            'mean_svf': mean_svf,
                            'bbox': bbox,
                            'position': (center_y, center_x)  # デバッグ用
                        })
                        break
            else:
                # 最大試行回数に達した場合のフォールバック
                region_size = random.randint(h//6, h//4)
                center_y = random.randint(region_size//2, h - region_size//2)
                center_x = random.randint(region_size//2, w - region_size//2)
                
                y_start = max(0, center_y - region_size//2)
                x_start = max(0, center_x - region_size//2)
                y_end = min(h, y_start + region_size)
                x_end = min(w, x_start + region_size)
                
                region_svf = self.svf_map[y_start:y_end, x_start:x_end]
                valid_mask = ~np.isnan(region_svf) & (region_svf > 0)
                
                if np.sum(valid_mask) > 0:
                    mean_svf = np.mean(region_svf[valid_mask])
                    bbox = self._get_relative_bbox(y_start, x_start, y_end-y_start, x_end-x_start)
                    
                    candidate_regions.append({
                        'label': label,
                        'mean_svf': mean_svf,
                        'bbox': bbox,
                        'position': (center_y, center_x)
                    })
        
        highest_region = max(candidate_regions, key=lambda x: x['mean_svf'])
        
        pre_shuffle_correct_idx = next(i for i, r in enumerate(candidate_regions) if r == highest_region)
        
        candidate_regions = assign_balanced_labels(candidate_regions, highest_region)
        candidate_regions = sorted(candidate_regions, key=lambda x: x['label'])
        
        if self.coordinate_answers:
            correct_answer = _format_coordinate_answer(highest_region['label'], highest_region['bbox'])
        else:
            correct_answer = f"Region {highest_region['label']}"
        
        region_svf_scores = [r['mean_svf'] for r in candidate_regions]
        
        base_question = self._get_question_template('sun_exposure')
        if base_question is None:
            base_question = "Which region has the highest average sun exposure?"
            
        question_text = f"{base_question}\n\n"
        for region in candidate_regions:
            bbox = region['bbox']
            question_text += f"{region['label']}: [xmin={bbox[0]}%, ymin={bbox[1]}%, xmax={bbox[2]}%, ymax={bbox[3]}%]\n"
        
        question_text += f"\nPlease choose from:\n"
        choices_list = []
        for region in candidate_regions:
            region_choice = f"Region {region['label']}"
            question_text += f"{region_choice}\n"
            choices_list.append(region_choice)
        
        question_text += "Coordinate Guide: Each region shows [left%, top%, right%, bottom%] as percentage of image size.\n"
        question_text += "Think of the image like a map: [4%, 58%, 20%, 76%] means:\n"
        question_text += "• Start 4% from left edge, 58% down from top\n"
        question_text += "• End 20% from left edge, 76% down from top\n"
        question_text += "This creates a rectangular region in that area of the image.\n"
        question_text += "Note: The answer should be determined based on the average SVF (Sky View Factor) score of each region."
        
        debug_info_choices = []
        for i_choice, region in enumerate(candidate_regions):
            status = "CORRECT" if region['label'] == highest_region['label'] else "wrong"
            debug_info_choices.append(f"Choice {i_choice+1} {region['label']}: SVF={region['mean_svf']:.3f} (bbox: {region['bbox']}) {status}")

        question_info = {
            "question": question_text,
            "region_choices": choices_list,
            "choices_bboxes": [r['bbox'] for r in candidate_regions],
            "region_details": {r['label']: r['mean_svf'] for r in candidate_regions},
            "choices": choices_list,
            "scores": {i: float(r['mean_svf']) for i, r in enumerate(candidate_regions)},
            "debug_info": debug_info_choices
        }
        
        if self._debug:
            tqdm_safe_print(f"[sunExposure] Candidate regions:")
            for dbg_item in debug_info_choices:
                tqdm_safe_print(f"  {dbg_item}")
            correct_idx = next(i for i, r in enumerate(candidate_regions) if r['label'] == highest_region['label'])
            correct_svf = region_svf_scores[correct_idx]
            tqdm_safe_print(f"  Correct answer: {correct_answer} (SVF: {correct_svf:.3f})")
        
        return question_info, correct_answer, ['sun_exposure', highest_region['bbox'], highest_region['mean_svf']]

    
# Example usage (for testing purposes, typically called from another script)
if __name__ == '__main__':
    # Create dummy maps for testing
    dummy_svf = np.random.rand(100, 100)
    dummy_height = np.random.rand(100, 100) * 50 # Heights up to 50m
    dummy_seg = np.random.randint(0, 11, size=(100, 100)) # 11 classes like ADE20K
    dummy_rgb = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    print("=== Completely random question generation (no seed set) ===")
    constructor_random = ConstructSVFQuestionRGB(
        estimated_svf_map=dummy_svf, 
        estimated_height_map=dummy_height, 
        estimated_segmentation_map=dummy_seg,
        rgb_image=dummy_rgb,
        file_path="dummy_map_test.json",
        cnt=0,
        debug=True
    )
    questions_random, answers_random, canonical_random = constructor_random.chooseQuestionsToAsk(number_question=7)
    tqdm_safe_print(f"\n--- Generated {len(questions_random)} questions (completely random) ---")
    for i, q in enumerate(questions_random):
        tqdm_safe_print(f"Q{i+1}: {q}")
        tqdm_safe_print(f"A{i+1}: {answers_random[i]}")

    print("\n=== Second run (different results expected) ===")
    constructor_random2 = ConstructSVFQuestionRGB(
        estimated_svf_map=dummy_svf, 
        estimated_height_map=dummy_height, 
        estimated_segmentation_map=dummy_seg,
        rgb_image=dummy_rgb,
        file_path="dummy_map_test.json",
        cnt=0,  # 同じcnt値でも異なる結果
        debug=True
    )
    questions_random2, answers_random2, canonical_random2 = constructor_random2.chooseQuestionsToAsk(number_question=7)
    tqdm_safe_print(f"\n--- Generated {len(questions_random2)} questions (2nd run) ---")
    for i, q in enumerate(questions_random2):
        tqdm_safe_print(f"Q{i+1}: {q}")
        tqdm_safe_print(f"A{i+1}: {answers_random2[i]}")

    # Test with missing segmentation map
    constructor_no_seg = ConstructSVFQuestionRGB(
        estimated_svf_map=dummy_svf, 
        estimated_height_map=dummy_height, 
        # estimated_segmentation_map=None, # Missing segmentation
        rgb_image=dummy_rgb,
        file_path="dummy_map_test_no_seg.json",
        cnt=1,
        debug=True
    )
    questions_no_seg, _, _ = constructor_no_seg.chooseQuestionsToAsk(number_question=5)
    tqdm_safe_print(f"\n--- Generated {len(questions_no_seg)} questions (no segmentation map) ---")

    # Test with missing height map
    constructor_no_height = ConstructSVFQuestionRGB(
        estimated_svf_map=dummy_svf, 
        # estimated_height_map=None, # Missing height
        estimated_segmentation_map=dummy_seg,
        rgb_image=dummy_rgb,
        file_path="dummy_map_test_no_height.json",
        cnt=2,
        debug=True
    )
    questions_no_height, _, _ = constructor_no_height.chooseQuestionsToAsk(number_question=5)
    tqdm_safe_print(f"\n--- Generated {len(questions_no_height)} questions (no height map) ---")

    # Test with only SVF map
    constructor_only_svf = ConstructSVFQuestionRGB(
        estimated_svf_map=dummy_svf, 
        file_path="dummy_map_test_only_svf.json",
        cnt=3,
        debug=True
    )
    questions_only_svf, _, _ = constructor_only_svf.chooseQuestionsToAsk(number_question=5)
    tqdm_safe_print(f"\n--- Generated {len(questions_only_svf)} questions (only SVF map) ---") 