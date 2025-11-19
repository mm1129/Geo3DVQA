"""
GPT-4 Based Free-form Answer Generator for SVF-based VQA System

This module generates high-quality Ground Truth (GT) answers for free-form questions
by integrating statistics, question context, and image analysis using GPT-4o-mini.

Based on the pattern from freeform_caption_generator.py, this system:
- Takes scene statistics, question text, and optional image data
- Generates comprehensive, analytical answers using GPT-4o-mini
- Provides fallback mechanisms for API unavailability
- Includes bias prevention and quality control

Key Features:
- Multi-modal data integration (SVF, DSM, Segmentation)
- GPT-4o-mini powered natural language generation  
- Structured prompt engineering for consistent answers
- Quality validation and length control
- Rate limiting and error handling
"""

import json
import numpy as np
import random
import time
import os
from typing import Dict, List, Optional, Any
from dataclasses import asdict, is_dataclass
from pathlib import Path
import logging
from datetime import datetime
import base64
from io import BytesIO
import yaml
import re

from scene_statistics import SceneStatistics

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("  python-dotenv not installed. Using system environment variables only")

# 画像処理用のインポート
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("  PIL not available. Image processing features will be disabled.")

logger = logging.getLogger(__name__)

class GPT4FreeformAnswerGenerator:
    """
    GPT-4 powered answer generator for free-form VQA questions
    
    Generates detailed, analytical answers that integrate:
    - Comprehensive scene statistics (SVF, DSM, Segmentation)
    - Question context and analysis requirements
    - Optional visual image analysis
    - Domain-specific expertise (urban planning, environmental assessment, etc.)
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini", debug: bool = False, output_dir: str = 'output'):
        """
        Initialize the GPT-4 answer generator
        
        Args:
            api_key: OpenAI API key (optional, can use environment variable)
            model: GPT model to use (default: gpt-4o-mini)
            debug: Enable debug output for API call tracking
        """
        self.debug = debug
        
        # API設定
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.api_enabled = self.api_key is not None
        self.model = model
        
        # レート制限設定
        self.min_interval = 0.8  # 800ms minimum interval for stability
        self.last_api_call = 0
        
        # Answer length policy aligned with question instruction (70-80 words)
        self.min_answer_words = 60
        self.max_answer_words = 100
        self.target_answer_words = 75
        self.output_dir = output_dir
        
        if self.debug:
            print(f" GPT4FreeformAnswerGenerator initialized:")
            print(f"   API enabled: {self.api_enabled}")
            print(f"   Model: {self.model}")
            print(f"   Target answer length: {self.target_answer_words} words")
            print(f"   Debug mode: {self.debug}")
    
    def generate_answer(self, 
                       question: str, 
                       statistics: SceneStatistics, 
                       scene_id: str,
                       analysis_type: str = "comprehensive",
                       image_data: Optional[np.ndarray] = None,
                       question_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate comprehensive answer for free-form question using GPT-4
        
        Args:
            question: The question text to answer
            statistics: Scene statistics computed from multi-modal data
            scene_id: Unique identifier for the scene
            analysis_type: Type of analysis (urban_development, renewable_energy, etc.)
            image_data: Optional RGB image data for visual analysis
            question_context: Additional context about the question
            
        Returns:
            Generated answer text
        """
        try:
            if self.debug:
                print(f" Generating answer for scene {scene_id}, type: {analysis_type}")
            
            # Generate GPT-4 powered answer
            if self.api_enabled:
                answer = self._generate_gpt_answer(
                    question, statistics, scene_id, analysis_type, image_data, question_context
                )
            
                
                if answer and self._validate_answer_quality(answer):
                    if self.debug:
                        print(f" GPT-4 answer generated successfully ({len(answer.split())} words)")
                    return answer
                else:
                    if self.debug:
                        print(f"  GPT-4 answer quality insufficient, using fallback")
            
            # Fallback to statistical answer generation
            fallback_answer = self._generate_fallback_answer(
                question, statistics, analysis_type
            )
            
            if self.debug:
                print(f" Fallback answer generated ({len(fallback_answer.split())} words)")
            
            return fallback_answer
            
        except Exception as e:
            print(f" Answer generation failed for scene {scene_id}: {e}")
            return self._generate_emergency_fallback(question, analysis_type)
    
    def _generate_gpt_answer(self, 
                           question: str, 
                           statistics: SceneStatistics, 
                           scene_id: str,
                           analysis_type: str,
                           image_data: Optional[np.ndarray] = None,
                           question_context: Optional[Dict[str, Any]] = None,
                        ) -> Optional[str]:
        """Generate answer using GPT-4 with comprehensive context"""
        
        try:
            # Prepare comprehensive scene context using all fields from SceneStatistics
            try:
                if is_dataclass(statistics):
                    scene_context_dict = asdict(statistics)
                else:
                    scene_context_dict = dict(getattr(statistics, '__dict__', {}))
            except Exception:
                scene_context_dict = {}

            # Add human-readable grid analysis names if available (Top/Middle/Bottom - Left/Center/Right)
            try:
                if hasattr(statistics, 'grid_analysis') and statistics.grid_analysis is not None:
                    from scene_statistics import serialize_grid_analysis  # lazy import
                    human_grid = serialize_grid_analysis(statistics.grid_analysis)
                    if human_grid:
                        scene_context_dict['svf_grid_summary_3x3'] = human_grid
                        scene_context_dict['grid_analysis_human_readable'] = human_grid
                elif hasattr(statistics, 'svf_grid_summary_3x3') and statistics.svf_grid_summary_3x3 is not None:
                    scene_context_dict['svf_grid_summary_3x3'] = statistics.svf_grid_summary_3x3
                elif hasattr(statistics, 'grid_analysis_human_readable') and statistics.grid_analysis_human_readable is not None:
                    # Backward-compatibility
                    scene_context_dict['svf_grid_summary_3x3'] = statistics.grid_analysis_human_readable
                    scene_context_dict['grid_analysis_human_readable'] = statistics.grid_analysis_human_readable
            except Exception:
                pass

            # Save YAML context to file for traceability
            os.makedirs(self.output_dir, exist_ok=True)  # ディレクトリを作成
            scene_context_file = f'{self.output_dir}/scene_context_{scene_id}_{analysis_type}.yaml'
            print(f" Saving scene context to: {scene_context_file}")
            try:
                with open(scene_context_file, 'w') as f:
                    yaml.safe_dump(scene_context_dict, f, sort_keys=False, allow_unicode=True)
                print(f" Scene context saved successfully: {scene_context_file}")
            except Exception as e:
                print(f" Failed to save scene context: {e}")
            # Create analysis-specific expert prompt
            expert_prompt = self._create_expert_prompt(analysis_type)
            
            # Construct comprehensive prompt
            scene_context_yaml = yaml.safe_dump(scene_context_dict, sort_keys=False, allow_unicode=True)
            # Derive EXPECTED_GRID from statistics when available, guided by analysis_type
            expected_grid = None
            try:
                ga = scene_context_dict.get('grid_analysis') or {}
                hyd = ga.get('hydrology') if isinstance(ga, dict) else None
                optimal = ga.get('optimal_locations') if isinstance(ga, dict) else None
                extreme = ga.get('extreme_locations') if isinstance(ga, dict) else None

                def _safe_name(d, key):
                    if not isinstance(d, dict):
                        return None
                    obj = d.get(key)
                    if isinstance(obj, dict):
                        return obj.get('name') or obj.get('grid_name')
                    return None

                # Map analysis_type → expected key
                if analysis_type == 'water_accumulation':
                    expected_grid = (
                        _safe_name(hyd, 'best_water_accumulation_position')
                        or _safe_name(extreme, 'lowest_height_position')
                    )
                elif analysis_type == 'renewable_energy':
                    expected_grid = (
                        _safe_name(optimal, 'best_solar_position')
                        or _safe_name(optimal, 'best_svf_position')
                    )
                elif analysis_type == 'urban_development':
                    expected_grid = _safe_name(optimal, 'best_development_position')
                elif analysis_type == 'landscape_analysis':
                    expected_grid = _safe_name(optimal, 'best_scenic_position')
                else:
                    # comprehensive or fallback: follow general priority
                    expected_grid = (
                        _safe_name(hyd, 'best_water_accumulation_position')
                        or _safe_name(extreme, 'lowest_height_position')
                        or _safe_name(optimal, 'best_solar_position')
                        or _safe_name(optimal, 'best_svf_position')
                        or _safe_name(optimal, 'best_development_position')
                        or _safe_name(optimal, 'best_scenic_position')
                    )
            except Exception:
                expected_grid = None
            prompt = f"""
{expert_prompt}

QUESTION TO ANSWER:
{question}

STATISTICAL CONTEXT (FOR OBSERVATION ONLY) [YAML]:
{scene_context_yaml}

ANSWER GENERATION REQUIREMENTS:
Follow the exact format specified in the question above. If the question specifies <OBSERVATION> and <CONCLUSION>, use that structure.
The total length of the answer should be 70-80 words.

STRICT SECTION RULES:
- <OBSERVATION>: Use concrete, image-grounded facts and numerical statistics from the YAML (SVF 0-1, elevation in meters, land-cover %). No interpretation.
- <CONCLUSION>: Do NOT introduce any new numbers or facts not present in <OBSERVATION>. Synthesize ONLY from <OBSERVATION> and what is visually consistent across the 2x2 image panel (RGB, SVF, DSM, SEG). Ignore the YAML in this section.
- If any discrepancy arises between numbers and visuals, prioritize visual consistency; still avoid adding new numbers in <CONCLUSION>.
- Avoid statements like "natural vegetation and water" unless clearly supported by the SEG panel.

QUALITATIVE DESCRIPTOR STANDARDIZATION (STRICT):
- Use standardized descriptors tied to numeric ranges for key metrics (e.g., SVF):
  excellent (>0.8), above-average (0.6-0.8), moderate (0.4-0.6), below-average (0.2-0.4), poor (<0.2).
- Avoid vague or lenient terms such as "good", "nice", or "adequate". Match descriptors to the actual values.

IMPORTANT NAMING RULES FOR GRID POSITIONS:
- If provided, prefer svf_grid_summary_3x3.optimal_locations.*.name for location words.
- Otherwise use grid_analysis_human_readable.optimal_locations.*.name.
- The naming format is Top/Middle/Bottom combined with Left/Center/Right (e.g., "Middle-Left").
- Do NOT use compass directions like north/south/east/west in your answer.
GRID CHECK:
- Mention exactly one sector; ensure it matches RGB/SVF/DSM/SEG.

DATA SCOPE (STRICT):
- Top-level stats (svf_mean, built_ratio, vegetation_ratio, height_*, scenic_quality, spatial_heterogeneity) describe the whole scene.
- Grid numbers exist only under svf_grid_summary_3x3 or grid_analysis.* and apply to a single sector.
- Do not attach whole-scene numbers to a sector. Use "overall ..." for global values.
- If grid numbers are absent, use qualitative comparison only (no invented percentages).
- In metrics-only runs, treat all metrics as whole-scene aggregates.

CERTAINTY POLICY FOR GRID MENTIONS (STRICT):
- Mention AT MOST ONE grid sector per each perspective. Do NOT list multiple sectors (e.g., avoid "Top-Left and Bottom-Center").
- Choose the single sector that is most certain from the provided statistics using this mapping:
  * water_accumulation: use best_water_accumulation_position (selected by local low elevation + gentle slope + high water pixel ratio)
  * renewable_energy (solar/wind): use best_solar_position; if unavailable, fall back to best_svf_position
  * urban_development: use best_development_position
  * landscape_analysis: use best_scenic_position
- If the corresponding position is missing/ambiguous (no clear best), OMIT grid location mentions entirely.
- Always verify the chosen sector against the grid matrices (SVF/height/etc.) before stating it.

EXPECTED GRID (if provided; use exactly this single sector for location mention; do not list multiple):
{expected_grid if expected_grid else 'N/A'}

Generate your answer following the format specified in the question.
IMPORTANT: Answer strictly according to the question. Do not add extra information or deviate from the requested format.

"""

            # Add image analysis text if available (kept for backward compatibility)
            if image_data is not None and HAS_PIL:
                image_prompt = self._create_image_analysis_prompt(image_data)
                if image_prompt:
                    prompt += f"\n\nVISUAL IMAGE ANALYSIS:\n{image_prompt}"
            else:
                print("No image data available")
            # Call GPT API with optional image
            response = self._call_gpt_api(prompt, image_data=image_data)
            if response and 'choices' in response:
                answer_text = response['choices'][0]['message']['content'].strip()
                answer_text = self._post_process_answer(answer_text)

                # Auto-correct step using verification feedback (grid consistency)
                try:
                    corrected = self._auto_correct_with_verification(
                        answer_text=answer_text,
                        statistics_context_yaml=scene_context_yaml,
                        image_data=image_data,
                        analysis_type=analysis_type,
                    )
                    if corrected:
                        return corrected
                except Exception as _e:
                    if self.debug:
                        print(f"  Auto-correction skipped: {_e}")
                # Structured two-stage candidate for water_accumulation (3x3 tiles)
                try:
                    if analysis_type == 'water_accumulation':
                        from freeform_qa_generation.structured.two_stage import generate_structured_water_answer
                        structured = generate_structured_water_answer(
                            model=self.model,
                            api_key=self.api_key,
                            stats_obj=scene_context_dict,
                            question=question,
                            n_best=2,
                            max_repairs=2,
                        )
                        if isinstance(structured, str) and structured.strip():
                            try:
                                from freeform_qa_generation.verification.conclusion_rules import canonicalize_answer
                                structured = canonicalize_answer(structured)
                            except Exception:
                                pass
                            # Prefer structured if it passes basic validation length
                            if self._validate_answer_quality(structured):
                                return structured
                except Exception as _e2:
                    if self.debug:
                        print(f"  Structured two-stage generation skipped: {_e2}")
                return answer_text
            
        except Exception as e:
            if self.debug:
                print(f" GPT answer generation failed: {e}")
        
        return None
    
    
    
    def _create_expert_prompt(self, analysis_type: str) -> str:
        """Create analysis-type specific expert system prompt"""
        
        expert_prompts = {
            'urban_development': """
You are an expert urban planner and landscape architect with extensive experience in sustainable development analysis. 
Your expertise includes land use planning, building density assessment, infrastructure evaluation, and environmental impact analysis.
Focus on development potential, planning considerations, infrastructure needs, and sustainable design principles.""",
            
            'renewable_energy': """
You are a renewable energy systems engineer and environmental consultant specializing in solar and wind energy site assessment.
Your expertise includes solar irradiance analysis, wind resource evaluation, site suitability assessment, and environmental impact considerations.
Focus on energy generation potential, installation feasibility, environmental constraints, and grid integration considerations.""",
            
            'landscape_analysis': """
You are a landscape ecologist and environmental scientist with expertise in spatial analysis and ecosystem assessment.
Your expertise includes landscape ecology, habitat connectivity, biodiversity assessment, and environmental quality evaluation.
Focus on ecological characteristics, environmental functions, spatial patterns, and landscape sustainability.""",
            
            'water_accumulation': """
You are a hydrologist and environmental engineer specializing in water management and flood risk assessment.
Your expertise includes drainage analysis, stormwater management, flood modeling, and water resource planning.
Focus on water flow patterns, accumulation zones, drainage efficiency, and flood risk mitigation.""",
            
            'comprehensive': """
You are a multidisciplinary environmental consultant with expertise in landscape analysis, urban planning, and environmental assessment.
Your expertise spans multiple domains including spatial analysis, environmental science, and sustainable development.
Focus on integrating multiple perspectives to provide comprehensive landscape analysis."""
        }
        
        return expert_prompts.get(analysis_type, expert_prompts['comprehensive'])
    
    def _create_image_analysis_prompt(self, image_data: np.ndarray) -> Optional[str]:
        """Create image analysis prompt if PIL is available"""
        if not HAS_PIL or image_data is None:
            return None
        
        try:
            # Convert numpy array to PIL Image
            if image_data.dtype != np.uint8:
                image_data = (image_data * 255).astype(np.uint8)
            
            if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                pil_image = Image.fromarray(image_data, 'RGB')
            else:
                return None
            
            # Create visual analysis context
            h, w = image_data.shape[:2]
            return f"""
VISUAL IMAGE CHARACTERISTICS:
- Image dimensions: {w} x {h} pixels
- Visual analysis available for cross-reference with statistical data
- Use this visual context to validate and enhance statistical interpretations
- Focus on observable landscape patterns, development density, and spatial organization"""
            
        except Exception as e:
            if self.debug:
                print(f"  Image analysis preparation failed: {e}")
            return None
    
    def _encode_image_to_data_url(self, image_data: np.ndarray) -> Optional[str]:
        """Encode numpy image to PNG data URL for multimodal chat input"""
        if not HAS_PIL or image_data is None:
            return None
        try:
            arr = image_data
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255)
                arr = arr.astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            pil_img = Image.fromarray(arr, 'RGB')
            buffer = BytesIO()
            pil_img.save(buffer, format='PNG')
            b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{b64}"
        except Exception as e:
            if self.debug:
                print(f"  Image base64 encoding failed: {e}")
            return None
    
    def _call_gpt_api(self, prompt: str, image_data: Optional[np.ndarray] = None) -> Optional[Dict]:
        """Call GPT API with rate limiting and optional image content"""
        if not self.api_enabled:
            return None

        try:
            # Rate limiting
            current_time = time.time()
            elapsed = current_time - self.last_api_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)

            model = str(self.model)

            def is_responses_pref_model(m: str) -> bool:
                # Use Responses API for gpt-5, o*, gpt-4o* models
                return m.startswith(("gpt-5", "o", "gpt-4o"))

            def supports_temperature_override(m: str) -> bool:
                # Block temperature for models that are known to not support it
                fixed = m.startswith(("gpt-5", "o", "gpt-4o", "gpt-4o-mini-transcribe", "gpt-4o-transcribe"))
                return not fixed

            image_url = self._encode_image_to_data_url(image_data) if image_data is not None else None

            # Responses API path
            if is_responses_pref_model(model):
                if self.debug:
                    print("[gpt4_freeform] using Responses API")

                content_blocks = []
                if image_url:
                    content_blocks = [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": {"url": image_url}},
                    ]
                else:
                    content_blocks = [{"type": "input_text", "text": prompt}]

                try:
                    resp = client.responses.create(
                        model=model,
                        input=[
                            {"role": "system", "content": [{"type": "input_text", "text": "You are an expert landscape analyst providing detailed, evidence-based assessments of environmental and spatial characteristics."}]},
                            {"role": "user",   "content": content_blocks},
                        ],
                        max_output_tokens=600,
                        # temperature not sent for safety
                    )
                    content = getattr(resp, "output_text", "") or ""
                    if not content:
                        try:
                            first_output = resp.output[0]
                            parts = getattr(first_output, "content", [])
                            texts = [p.text for p in parts if getattr(p, "type", "") == "output_text" and hasattr(p, "text")]
                            content = "\n".join(texts).strip()
                        except Exception:
                            content = ""
                    if not content:
                        raise RuntimeError("Empty response content from Responses API")

                    self.last_api_call = time.time()
                    return {"choices": [{"message": {"content": content}}]}

                except Exception as e:
                    if self.debug:
                        print(f" Responses API call failed: {e}")
                    return None

            # Chat Completions path
            if image_url:
                user_content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
                if self.debug:
                    print("[gpt4_freeform] sending multimodal message with image attached")
            else:
                user_content = prompt

            messages = [
                {"role": "system", "content": "You are an expert landscape analyst providing detailed, evidence-based assessments of environmental and spatial characteristics."},
                {"role": "user", "content": user_content},
            ]

            kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": 600,
            }

            # Only send temperature if model supports it and default_temperature is set
            if supports_temperature_override(model) and getattr(self, "default_temperature", None) is not None:
                kwargs["temperature"] = float(self.default_temperature)

            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as e:
                msg = str(e)
                if "unsupported_value" in msg and "'temperature'" in msg:
                    if self.debug:
                        print("↻ Retrying without temperature due to model restriction")
                    kwargs.pop("temperature", None)
                    response = client.chat.completions.create(**kwargs)
                else:
                    if self.debug:
                        print(f" GPT API call failed: {e}")
                    return None

            self.last_api_call = time.time()
            return {"choices": [{"message": {"content": response.choices[0].message.content}}]}

        except Exception as e:
            if self.debug:
                print(f" GPT API call failed: {e}")
            return None
    
    def _post_process_answer(self, answer_text: str) -> str:
        """Post-process GPT answer for quality and length.

        Keep output concise and do not add generic filler sentences. Enforce
        approximate length bounds and canonicalize section tags.
        """
        if not answer_text:
            return ""
        
        # Clean up formatting
        answer_text = answer_text.strip()
        
        # Check word count and adjust if necessary
        words = answer_text.split()
        word_count = len(words)
        
        if word_count > self.max_answer_words:
            # Truncate to max length, trying to end at sentence boundary
            truncated_words = words[:self.max_answer_words]
            truncated_text = ' '.join(truncated_words)
            
            # Find last complete sentence
            last_period = truncated_text.rfind('.')
            if last_period > len(truncated_text) * 0.8:  # Keep if period is in last 20%
                answer_text = truncated_text[:last_period + 1]
            else:
                answer_text = truncated_text + '.'
        
        elif word_count < self.min_answer_words:
            # Too short: return as-is and let verification gate it (no generic filler)
            try:
                from freeform_qa_generation.verification.conclusion_rules import canonicalize_answer
                answer_text = canonicalize_answer(answer_text)
            except Exception:
                pass
            return answer_text
        
        # Canonicalize sections to ensure exactly one OBSERVATION and one CONCLUSION
        try:
            from freeform_qa_generation.verification.conclusion_rules import canonicalize_answer
            answer_text = canonicalize_answer(answer_text)
        except Exception:
            pass
        
        return answer_text
    
    def _validate_answer_quality(self, answer: str) -> bool:
        """Validate answer quality and completeness"""
        if not answer or len(answer.strip()) < 50:
            return False
        
        word_count = len(answer.split())
        if word_count < self.min_answer_words or word_count > self.max_answer_words + 20:
            return False
        
        # Check for some evidence of data integration
        data_indicators = ['svf', 'elevation', 'height', 'coverage', 'percentage', '%', 'meter', 'ratio']
        has_data_reference = any(indicator in answer.lower() for indicator in data_indicators)
        
        return has_data_reference
    
    def _generate_fallback_answer(self, question: str, statistics: SceneStatistics, analysis_type: str) -> str:
        """Generate fallback answer using statistical data when GPT is unavailable"""
        
        # Basic landscape characterization
        if statistics.built_ratio > 0.4:
            landscape_char = f"This urban landscape with {statistics.built_ratio:.1%} built coverage"
        elif statistics.vegetation_ratio > 0.6:
            landscape_char = f"This natural landscape with {statistics.vegetation_ratio:.1%} vegetation coverage"
        else:
            landscape_char = f"This mixed landscape with {statistics.built_ratio:.1%} built and {statistics.vegetation_ratio:.1%} natural coverage"
        
        # Sky visibility analysis
        sky_analysis = f"demonstrates {self._describe_openness_level(statistics.svf_mean)} with average SVF of {statistics.svf_mean:.3f}"
        
        # Terrain characteristics
        terrain_analysis = f"The terrain shows {statistics.height_range:.1f}m elevation variation, indicating {self._describe_terrain_character(statistics.height_std, statistics.height_range)} topographic conditions"
        
        # Analysis-type specific insights
        type_specific = {
            'urban_development': (
                f"Development potential is "
                f"{self._standardized_descriptor(statistics.svf_mean)} "
                f"given the sky accessibility and terrain characteristics. "
                f"Infrastructure planning should consider the {statistics.spatial_heterogeneity:.2f} spatial diversity index."
            ),
            
            'renewable_energy': (
                f"Solar energy potential is {self._standardized_descriptor(statistics.svf_mean)} "
                f"with mean sky visibility of {statistics.svf_mean:.3f}. "
                f"Available installation area comprises approximately {(1-statistics.built_ratio)*100:.0f}% of the landscape."
            ),
            
            'landscape_analysis': (
                f"The landscape exhibits {self._describe_complexity_level(statistics.spatial_heterogeneity)} spatial complexity "
                f"with Shannon diversity of {statistics.spatial_heterogeneity:.3f}. "
                f"Environmental quality reflects {self._describe_scenic_level(statistics.scenic_quality)} landscape character."
            ),
            
            'water_accumulation': (
                f"Water accumulation risk is {'elevated' if statistics.height_std < 5 else 'moderate'} in low-lying areas. "
                f"The {statistics.height_range:.1f}m elevation range creates {'significant' if statistics.height_range > 20 else 'moderate'} drainage gradients."
            )
        }
        
        specific_analysis = type_specific.get(analysis_type, f"The landscape characteristics suggest {self._assess_environmental_balance(statistics)} environmental conditions with {self._assess_development_suitability(statistics)} development suitability.")
        
        # Combine into comprehensive answer
        answer = f"{landscape_char} {sky_analysis}. {terrain_analysis}. {specific_analysis} The integration of these factors creates a distinctive spatial organization with specific implications for land use planning and environmental management."
        
        return self._post_process_answer(answer)
    
    def _generate_emergency_fallback(self, question: str, analysis_type: str) -> str:
        """Generate emergency fallback when all else fails"""
        return f"Based on available landscape data, this area requires comprehensive {analysis_type.replace('_', ' ')} assessment considering multiple environmental and spatial factors. Detailed analysis would integrate sky visibility patterns, terrain characteristics, and land cover distribution to provide specific recommendations for the landscape management and development considerations."
    
    # Helper methods for descriptive analysis (inherited from freeform_caption_generator pattern)
    
    
    
    def _describe_scenic_level(self, score: float) -> str:
        """Convert scenic score to descriptive level"""
        if score > 0.8:
            return "exceptionally scenic"
        elif score > 0.6:
            return "visually appealing"
        elif score > 0.4:
            return "moderately scenic"
        elif score > 0.2:
            return "simple landscape character"
        else:
            return "utilitarian character"
    
    def _describe_openness_level(self, svf: float) -> str:
        """Convert SVF to openness description"""
        if svf > 0.8:
            return "highly open with extensive sky visibility"
        elif svf > 0.6:
            return "moderately open landscape"
        elif svf > 0.4:
            return "partially enclosed by vertical elements"
        elif svf > 0.2:
            return "enclosed with limited sky access"
        else:
            return "heavily enclosed environment"
    
    def _describe_complexity_level(self, heterogeneity: float) -> str:
        """Convert spatial heterogeneity to complexity description"""
        if heterogeneity > 1.5:
            return "highly diverse land use patterns"
        elif heterogeneity > 1.0:
            return "moderately varied landscape composition"
        elif heterogeneity > 0.5:
            return "somewhat uniform with distinct zones"
        else:
            return "predominantly uniform land cover"
    
    def _describe_terrain_character(self, height_std: float, height_range: float) -> str:
        """Describe terrain characteristics"""
        if height_range > 30:
            return "highly variable"
        elif height_range > 15:
            return "moderately variable"
        elif height_range > 5:
            return "gently undulating"
        else:
            return "relatively flat"
    
    def _assess_environmental_balance(self, statistics: SceneStatistics) -> str:
        """Assess environmental balance"""
        nat_ratio = statistics.vegetation_ratio
        built_ratio = statistics.built_ratio
        
        if nat_ratio > 0.7:
            return "natural-dominated"
        elif built_ratio > 0.6:
            return "urban-dominated"
        elif abs(nat_ratio - built_ratio) < 0.2:
            return "balanced natural-built"
        else:
            return "mixed development"
    
    def _assess_development_suitability(self, statistics: SceneStatistics) -> str:
        """Assess development suitability"""
        if statistics.svf_mean > 0.6 and statistics.height_std < 10:
            return "high"
        elif statistics.svf_mean > 0.4 and statistics.height_std < 20:
            return "moderate"
        else:
            return "constrained"

    def _standardized_descriptor(self, score: float) -> str:
        """Map a 0.0-1.0 score to standardized qualitative descriptor."""
        if score > 0.8:
            return "excellent"
        elif score > 0.6:
            return "above-average"
        elif score > 0.4:
            return "moderate"
        elif score > 0.2:
            return "below-average"
        else:
            return "poor"

    def _auto_correct_with_verification(self, answer_text: str, statistics_context_yaml: str, image_data: Optional[np.ndarray], analysis_type: Optional[str] = None) -> Optional[str]:
        """Run verification once and, if statistics-backed grid mismatch/multiple/missing is detected,
        minimally rewrite only the <CONCLUSION> to align with the single expected sector, then re-verify once.
        Returns corrected text if the second verification passes; otherwise returns None.
        """
        try:
            from freeform_qa_generation.verification.pipeline import verify_answer
            from freeform_qa_generation.verification.conclusion_rules import extract_sections, canonicalize_answer
        except Exception:
            return None

        data_url = self._encode_image_to_data_url(image_data) if image_data is not None else None
        verdict = verify_answer(
            answer_text,
            panel_image_data_url=data_url,
            enable_llm=True,
            statistics_context=statistics_context_yaml,
            analysis_type=analysis_type,
        )
        rules = (verdict or {}).get("rules", {})
        grid_info = rules.get("grid_feedback") or ((rules.get("violations") or {}).get("grid_consistency"))
        if not isinstance(grid_info, dict):
            return None

        status = str(grid_info.get("status", "unknown"))
        expected = grid_info.get("expected")
        if not expected or status in ("match", "unknown"):
            return None

        sections = extract_sections(answer_text or "")
        obs = sections.get("OBSERVATION", "")
        con = sections.get("CONCLUSION", "")
        expected_human = "-".join([p.capitalize() for p in str(expected).split("-")])

        # Replace any grid mentions with a single expected sector; remove extras.
        # Accept both orders (row-col and col-row) and British spelling
        # Accept both row-col and col-row; also normalize standalone center/middle as central cell
        pattern = r"\b(?:(?:top|middle|bottom)[ -](?:left|center|centre|right)|(?:left|center|centre|right)[ -](?:top|middle|bottom)|center|centre|middle)\b"
        replaced_once = {"done": False}

        def _repl(m):
            if not replaced_once["done"]:
                replaced_once["done"] = True
                return expected_human
            return ""

        new_con = re.sub(pattern, _repl, con, flags=re.IGNORECASE)

        # Also replace compass direction mentions (e.g., southeast) with the expected grid name
        compass_pattern = r"\b(?:north|south|east|west|northeast|north[- ]east|northwest|north[- ]west|southeast|south[- ]east|southwest|south[- ]west)\b"
        new_con = re.sub(compass_pattern, expected_human, new_con, flags=re.IGNORECASE)

        if (status == "missing") and (expected_human.lower() not in new_con.lower()):
            # Inject a concise location mention early in the sentence without adding numbers
            period = new_con.find(".")
            inject = f" in {expected_human}"
            if period != -1:
                new_con = new_con[:period] + inject + new_con[period:]
            else:
                new_con = new_con.strip()
                if not new_con.endswith("."):
                    new_con += "."
                new_con = new_con + inject

        corrected = canonicalize_answer(
            f"<OBSERVATION>{obs}</OBSERVATION><CONCLUSION>{new_con.strip()}</CONCLUSION>"
        )

        verdict2 = verify_answer(
            corrected,
            panel_image_data_url=data_url,
            enable_llm=True,
            statistics_context=statistics_context_yaml,
            analysis_type=analysis_type,
        )
        if bool((verdict2 or {}).get("passed", False)):
            return corrected
        return None

