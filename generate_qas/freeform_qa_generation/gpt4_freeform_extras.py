"""
Extras for GPT-4 free-form answer generation.

This module contains helper utilities and optional integration functions that are
not required by the main pipeline in `freeform_main_0717.py`.

Moved from `gpt4_freeform_answer_generator.py` to keep the core module minimal.
"""

from typing import Dict, Tuple, Optional

# Avoid import cycles: import the class lazily inside functions when needed
try:
    from scene_statistics import SceneStatistics  # type: ignore
except Exception:
    SceneStatistics = object  # type: ignore


def grid_position_name(pos: Tuple[int, int]) -> str:
    """Convert grid position to readable name (Top/Middle/Bottom - Left/Center/Right)."""
    row_names = ["top", "middle", "bottom"]
    col_names = ["left", "center", "right"]
    return f"{row_names[pos[0]]}-{col_names[pos[1]]}"


def assess_conservation_value(statistics: "SceneStatistics") -> str:
    """Assess conservation value based on vegetation and diversity."""
    try:
        if getattr(statistics, 'vegetation_ratio', 0) > 0.6 and getattr(statistics, 'spatial_heterogeneity', 0) > 1.0:
            return "high ecological value"
        elif getattr(statistics, 'vegetation_ratio', 0) > 0.4:
            return "moderate conservation value"
        else:
            return "limited conservation priority"
    except Exception:
        return "limited conservation priority"


def prepare_detailed_scene_context(
    statistics: "SceneStatistics",
    scene_id: str,
    helper: Optional[object] = None,
) -> str:
    """Prepare comprehensive scene context for GPT analysis.

    If `helper` is provided and exposes descriptive methods (e.g., from
    GPT4FreeformAnswerGenerator), they will be used; otherwise simple text is used.
    """

    # Fallback descriptors if helper is not provided
    def _desc_scenic(x: float) -> str:
        if x > 0.8:
            return "exceptionally scenic"
        if x > 0.6:
            return "visually appealing"
        if x > 0.4:
            return "moderately scenic"
        if x > 0.2:
            return "simple landscape character"
        return "utilitarian character"

    def _desc_open(x: float) -> str:
        if x > 0.8:
            return "highly open with extensive sky visibility"
        if x > 0.6:
            return "moderately open landscape"
        if x > 0.4:
            return "partially enclosed by vertical elements"
        if x > 0.2:
            return "enclosed with limited sky access"
        return "heavily enclosed environment"

    def _desc_terrain(stdv: float, rng: float) -> str:
        if rng > 30:
            return "highly variable"
        if rng > 15:
            return "moderately variable"
        if rng > 5:
            return "gently undulating"
        return "relatively flat"

    def _desc_complexity(h: float) -> str:
        if h > 1.5:
            return "highly diverse land use patterns"
        if h > 1.0:
            return "moderately varied landscape composition"
        if h > 0.5:
            return "somewhat uniform with distinct zones"
        return "predominantly uniform land cover"

    scenic = getattr(helper, '_describe_scenic_level', _desc_scenic)
    openness = getattr(helper, '_describe_openness_level', _desc_open)
    terrain = getattr(helper, '_describe_terrain_character', _desc_terrain)
    complexity = getattr(helper, '_describe_complexity_level', _desc_complexity)

    context = f"""
SCENE IDENTIFICATION: {scene_id}

LANDSCAPE OVERVIEW:
- Overall landscape classification: {getattr(statistics, 'landscape_type', 'unknown')}
- Spatial diversity index: {getattr(statistics, 'spatial_heterogeneity', 0.0):.3f} (Shannon diversity)
- Scenic quality assessment: {scenic(getattr(statistics, 'scenic_quality', 0.0))} (score: {getattr(statistics, 'scenic_quality', 0.0):.3f})

SKY VISIBILITY ANALYSIS (SVF):
- Mean sky view factor: {getattr(statistics, 'svf_mean', 0.0):.3f}
- SVF variation: {getattr(statistics, 'svf_std', 0.0):.3f}
- Minimum sky access: {getattr(statistics, 'svf_min', 0.0):.3f}
- Maximum sky access: {getattr(statistics, 'svf_max', 0.0):.3f}
- Median sky visibility: {getattr(statistics, 'svf_median', 0.0):.3f}
- Sky openness character: {openness(getattr(statistics, 'svf_mean', 0.0))}

TERRAIN AND ELEVATION CHARACTERISTICS:
- Average elevation: {getattr(statistics, 'height_mean', 0.0):.1f} meters above reference
- Elevation range: {getattr(statistics, 'height_range', 0.0):.1f} meters (from {getattr(statistics, 'height_min', 0.0):.1f}m to {getattr(statistics, 'height_max', 0.0):.1f}m)
- Terrain variation: {getattr(statistics, 'height_std', 0.0):.1f} meters standard deviation
- Topographic character: {terrain(getattr(statistics, 'height_std', 0.0), getattr(statistics, 'height_range', 0.0))}

LAND COVER COMPOSITION (Detailed Breakdown):"""

    land_cover = getattr(statistics, 'land_cover_ratios', {}) or {}
    for land_type, percentage in land_cover.items():
        if percentage > 0.01:
            context += f"\n- {land_type.replace('_', ' ').title()}: {percentage:.1%} of total area"

    context += f"""

ENVIRONMENTAL SUMMARY METRICS:
- Natural vegetation coverage: {getattr(statistics, 'vegetation_ratio', 0.0):.1%}
- Built environment coverage: {getattr(statistics, 'built_ratio', 0.0):.1%}
- Water feature coverage: {getattr(statistics, 'water_ratio', 0.0):.1%}
- Development pressure indicator: {getattr(statistics, 'built_ratio', 0.0) / (getattr(statistics, 'vegetation_ratio', 0.0) + 0.01):.2f}
"""

    grid = getattr(statistics, 'grid_analysis', None)
    if grid:
        try:
            context += f"""

SPATIAL ORGANIZATION ANALYSIS (3x3 Grid):
- Optimal sky visibility zone: {grid_position_name(grid.best_svf_position)} sector
- Highest development potential: {grid_position_name(grid.best_development_position)} sector
- Best solar energy potential: {grid_position_name(grid.best_solar_position)} sector
- Most scenic landscape area: {grid_position_name(grid.best_scenic_position)} sector"""
        except Exception:
            pass

    context += f"""

ANALYTICAL INSIGHTS:
- Landscape complexity: {complexity(getattr(statistics, 'spatial_heterogeneity', 0.0))}
"""

    # Environmental balance / development suitability if available
    try:
        from .gpt4_freeform_answer_generator import GPT4FreeformAnswerGenerator  # lazy import
        helper = helper or GPT4FreeformAnswerGenerator()
        balance = getattr(helper, '_assess_environmental_balance')(statistics)
        suitability = getattr(helper, '_assess_development_suitability')(statistics)
        conservation = assess_conservation_value(statistics)
        context += f"\n- Environmental balance: {balance}\n- Development suitability: {suitability}\n- Conservation value: {conservation}"
    except Exception:
        pass

    return context


def _create_mock_statistics_from_analysis(analysis: Dict) -> "MockSceneStatistics":
    """Create mock statistics object from analysis dictionary for compatibility."""

    class MockSceneStatistics:
        def __init__(self, analysis_data: Dict):
            sky_data = analysis_data.get('sky_visibility', {})
            self.svf_mean = sky_data.get('mean', 0.5)
            self.svf_std = sky_data.get('std', 0.2)
            self.svf_min = 0.1
            self.svf_max = 0.9
            self.svf_median = self.svf_mean

            terrain_data = analysis_data.get('terrain', {})
            self.height_mean = terrain_data.get('mean_height', 50.0)
            self.height_std = terrain_data.get('height_std', 10.0)
            self.height_min = self.height_mean - 20
            self.height_max = self.height_mean + 20
            self.height_range = self.height_max - self.height_min

            # Land cover ratios
            land_use_data = analysis_data.get('land_use', {})
            self.land_cover_ratios = land_use_data if land_use_data else {
                'urban': 0.3, 'vegetation': 0.5, 'water': 0.05, 'other': 0.15
            }

            # Derived metrics
            self.vegetation_ratio = sum(v for k, v in self.land_cover_ratios.items()
                                        if 'vegetation' in k or 'forest' in k or 'agricultural' in k)
            self.built_ratio = sum(v for k, v in self.land_cover_ratios.items()
                                   if 'urban' in k or 'building' in k or 'residential' in k)
            self.water_ratio = sum(v for k, v in self.land_cover_ratios.items()
                                   if 'water' in k)

            self.spatial_heterogeneity = 1.2
            self.scenic_quality = 0.6
            self.landscape_type = "mixed_development"

    return MockSceneStatistics(analysis)


def integrate_gpt4_answer_generation(
    freeform_categories_instance,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> None:
    """Integrate GPT-4 answer generation into FreeformAnalysisCategories instance.

    This monkey-patch function is optional and not used by the main pipeline.
    """

    # Lazy import to avoid circular dependency at module import time
    from .gpt4_freeform_answer_generator import GPT4FreeformAnswerGenerator

    # Create GPT-4 generator
    gpt4_generator = GPT4FreeformAnswerGenerator(api_key=api_key, model=model)

    # Store original methods (not used but kept for reference)
    original_methods = {
        'urban_development': freeform_categories_instance._generate_urban_development_recommendation,
        'renewable_energy': freeform_categories_instance._generate_energy_installation_recommendation,
        'landscape_analysis': freeform_categories_instance._generate_landscape_analysis,
        'water_accumulation': freeform_categories_instance._generate_water_accumulation_analysis,
    }

    # Enhanced wrappers
    def enhanced_urban_development_recommendation(analysis: Dict) -> str:
        question = (
            "Analyze the potential of this area for urban development considering sky visibility, "
            "existing land use patterns, building density, infrastructure accessibility, and environmental sustainability."
        )
        question += (
            " Answer in format: <OBSERVATION>landscape description</OBSERVATION><ANALYSIS>sky visibility, land use, "
            "infrastructure, sustainability with specific SVF values and percentages</ANALYSIS><CONCLUSION>development "
            "potential rating (High/Moderate/Low) with justification</CONCLUSION>. Include specific numerical data (SVF 0.0-1.0, "
            "elevation meters, land cover percentages)."
        )
        mock_stats = _create_mock_statistics_from_analysis(analysis)
        return gpt4_generator.generate_answer(
            question=question,
            statistics=mock_stats,
            scene_id=getattr(freeform_categories_instance, 'file_path', 'unknown'),
            analysis_type='urban_development',
        )

    def enhanced_energy_installation_recommendation(analysis: Dict) -> str:
        question = (
            "Analyze the potential of this area for solar panel and wind power generation installation, considering solar "
            "irradiance potential, wind exposure, land availability, and environmental impact."
        )
        question += (
            " Answer in format: <OBSERVATION>energy-relevant landscape characteristics</OBSERVATION><ANALYSIS>solar potential, "
            "wind exposure, available area, environmental constraints with specific SVF values and percentages</ANALYSIS>"
            "<CONCLUSION>installation feasibility rating (Excellent/Good/Limited) with capacity estimates</CONCLUSION>. "
            "Include specific numerical data (SVF 0.0-1.0, elevation meters, usable land percentages)."
        )
        mock_stats = _create_mock_statistics_from_analysis(analysis)
        return gpt4_generator.generate_answer(
            question=question,
            statistics=mock_stats,
            scene_id=getattr(freeform_categories_instance, 'file_path', 'unknown'),
            analysis_type='renewable_energy',
        )

    def enhanced_landscape_analysis(analysis: Dict) -> str:
        question = (
            "Analyze overall landscape characteristics including natural vs artificial balance, topographical features, "
            "ecological connectivity, visual landscape quality, and biodiversity potential."
        )
        question += (
            " Answer in format: <OBSERVATION>landscape composition and visual characteristics</OBSERVATION><ANALYSIS>natural-"
            "artificial balance, topographical features, ecological connectivity, visual quality with specific percentages and "
            "elevation data</ANALYSIS><CONCLUSION>landscape quality assessment (High/Moderate/Low) with management recommendations"
            "</CONCLUSION>. Include specific numerical data (land cover percentages, elevation statistics, diversity indices, SVF values)."
        )
        mock_stats = _create_mock_statistics_from_analysis(analysis)
        return gpt4_generator.generate_answer(
            question=question,
            statistics=mock_stats,
            scene_id=getattr(freeform_categories_instance, 'file_path', 'unknown'),
            analysis_type='landscape_analysis',
        )

    def enhanced_water_accumulation_analysis(analysis: Dict) -> str:
        question = (
            "Analyze water accumulation potential including topographical flow patterns, low-lying areas, drainage efficiency, "
            "and flood risk evaluation."
        )
        question += (
            " Answer in format: <OBSERVATION>terrain characteristics for water flow</OBSERVATION><ANALYSIS>flow patterns, "
            "low-lying areas, drainage efficiency, flood risk with specific elevation statistics and gradients</ANALYSIS>"
            "<CONCLUSION>accumulation risk assessment (High/Moderate/Low) with vulnerable areas and mitigation recommendations"
            "</CONCLUSION>. Include specific numerical data (elevation min/max/mean/std, gradients, land cover percentages)."
        )
        mock_stats = _create_mock_statistics_from_analysis(analysis)
        return gpt4_generator.generate_answer(
            question=question,
            statistics=mock_stats,
            scene_id=getattr(freeform_categories_instance, 'file_path', 'unknown'),
            analysis_type='water_accumulation',
        )

    # Apply monkey-patch
    freeform_categories_instance._generate_urban_development_recommendation = enhanced_urban_development_recommendation
    freeform_categories_instance._generate_energy_installation_recommendation = enhanced_energy_installation_recommendation
    freeform_categories_instance._generate_landscape_analysis = enhanced_landscape_analysis
    freeform_categories_instance._generate_water_accumulation_analysis = enhanced_water_accumulation_analysis

    print(" GPT-4 answer generation integrated successfully")

