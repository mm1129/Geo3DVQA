import os
import random
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
try:
    from utils import tqdm_safe_print, bias_free_shuffle
except ImportError:
    # フォールバック: utilsが見つからない場合
    def tqdm_safe_print(*args, **kwargs):
        print(*args, **kwargs)
    def bias_free_shuffle(choices_list):
        import time
        rng = random.Random(int(time.time() * 1000000) % 2147483647)
        shuffled = choices_list.copy()
        rng.shuffle(shuffled)
        return shuffled
import itertools

# Import pixel-level implementations from svf_questions_re.py
try:
    from svf_questions_re import ConstructSVFQuestion
    PIXEL_LEVEL_AVAILABLE = True
except ImportError:
    PIXEL_LEVEL_AVAILABLE = False
    # pixel level questions will be skipped silently

def assign_balanced_labels(regions, target_region):
    labels = ['A', 'B', 'C', 'D']
    available_labels = labels[:len(regions)]
    
    # 最強改良: 複数ラウンドランダム化で完全均等分布を実現
    # Method 1: Multiple shuffling rounds for stronger randomization
    for _ in range(5):  # 5回シャッフルで完全ランダム化
        random.shuffle(available_labels)
    
    # Method 2: Random choice with dynamic selection probability
    # ランダムファクターを追加してより均等な分布を確保
    random_factor = random.random()
    position_weights = [1.0 + 0.3 * random.random() for _ in available_labels]
    weighted_choice = random.choices(available_labels, weights=position_weights, k=1)[0]
    
    # Method 3: Use the most randomized result
    final_choice = random.choice([available_labels[0], weighted_choice])
    
    target_region['label'] = final_choice
    
    # 残りのラベルをシャッフル
    remaining_labels = [l for l in available_labels if l != final_choice]
    
    # さらに強力なシャッフル
    for _ in range(3):
        random.shuffle(remaining_labels)
    
    # 他の地域に残りのラベルを配布
    other_regions = [r for r in regions if r != target_region]
    for i, region in enumerate(other_regions):
        if i < len(remaining_labels):
            region['label'] = remaining_labels[i]
    
    return regions
class SVFHardQuestionMixin:
    """
    SVF質問生成のハードカテゴリ実装を提供するMixinクラス
    ConstructSVFQuestionRGBクラスに統合される
    """
    
    def _init_hard_templates(self, mode="train"):
        """Initialize hard category templates based on mode"""
        # Hard category question templates
        self.HARD_QUESTION_TEMPLATES = {
        'hard_sun_exposure': [
            "Which location receives the most sunlight? (Which spot looks brightest or most open to the sky?)",
            "Where is solar exposure highest?",
            "Which area gets the maximum sun exposure?",
            "Which spot gets the highest sun exposure?",
            "Where does the sun reach most directly?",
            "Which location has the strongest solar access?",
            "Where is direct sunlight most abundant?"
        ],
        'hard_scenic_quality': [
            "Which region has the highest scenic quality score?",
            "Where is the landscape quality index highest?",
            "Which area offers the highest visual quality rating?",
            "Where can you find the highest scenic value?",
            "Which region provides the highest landscape assessment score?",
            "Where is the environmental quality measurement highest?"
        ],
        'hard_pixel': [
            "What is the average SVF value for the region {region}?",
            "Calculate the mean Sky View Factor within the area {region}.",
            "Determine the average SVF measurement for region {region}.",
            "What is the regional average SVF value at {region}?",
            "Find the mean Sky View Factor value within the specified region {region}."
        ],
        'hard_grid_5×5': [
            "Which cell in the 5×5 grid has the highest average SVF value?",
            "In the 5×5 grid analysis, which cell shows the most sky visibility?",
            "Which 5×5 grid cell exhibits the maximum sky openness?",
            "Among all cells in the 5×5 grid, which has the highest SVF measurement?",
            "Which grid cell in the 5×5 division shows the greatest sky access?"
        ],
        'hard_metric': [
            "What is the {metric} of SVF values in the region from ({x1}%, {y1}%) to ({x2}%, {y2}%)?",
            "Calculate the {metric} for the Sky View Factor in the specified area ({x1}%, {y1}%) to ({x2}%, {y2}%).",
            "Determine the {metric} of sky visibility measurements in region ({x1}%, {y1}%) to ({x2}%, {y2}%).",
            "What is the {metric} value for SVF data in the area bounded by ({x1}%, {y1}%) and ({x2}%, {y2}%)?",
            "Find the {metric} of Sky View Factor values within the region ({x1}%, {y1}%) to ({x2}%, {y2}%)."
        ],
        'hard_ranking': [
            "Rank these three regions by their SVF values from highest to lowest: {region1}, {region2}, {region3}.",
            "Order these areas by sky visibility from most to least open: {region1}, {region2}, {region3}.",
            "Arrange these regions by their openness levels from highest to lowest: {region1}, {region2}, {region3}.",
            "Sort these three areas by SVF values in descending order: {region1}, {region2}, {region3}.",
            "List these regions from most to least sky-visible: {region1}, {region2}, {region3}."
        ],
        # Integrated Hard Categories
        'hard_urban_analysis': [
            "Provide a comprehensive urban density analysis for the region from ({x1}%, {y1}%) to ({x2}%, {y2}%), including building coverage ratio, floor area ratio, SVF impact, edge density, and integrated urban density score.",
            "Analyze the urban characteristics of the area bounded by ({x1}%, {y1}%) and ({x2}%, {y2}%), calculating BCR, FAR, SVF influence, visual complexity, and overall urban density metrics.",
            "Perform detailed urban planning analysis for region ({x1}%, {y1}%) to ({x2}%, {y2}%), evaluating building density, height distribution, sky visibility impact, and integrated development intensity.",
            "Calculate comprehensive urban metrics for the specified area ({x1}%, {y1}%) to ({x2}%, {y2}%), including coverage ratios, floor area calculations, SVF effects, and overall density assessment.",
            "Execute integrated urban analysis for region ({x1}%, {y1}%) to ({x2}%, {y2}%), determining building statistics, height analysis, visibility impacts, and composite urban density score."
        ],
        'hard_scenic_analysis': [
            "Conduct comprehensive scenic quality assessment for the region from ({x1}%, {y1}%) to ({x2}%, {y2}%), analyzing SVF contribution, naturalness, diversity, terrain variation, visual harmony, and integrated scenic score.",
            "Evaluate landscape quality metrics for area ({x1}%, {y1}%) to ({x2}%, {y2}%), including sky openness, natural elements ratio, environmental diversity, topographic interest, and overall scenic value.",
            "Perform detailed scenic analysis for region ({x1}%, {y1}%) to ({x2}%, {y2}%), calculating openness score, nature content, visual diversity, terrain complexity, color harmony, and integrated quality rating.",
            "Analyze environmental aesthetics for the specified area ({x1}%, {y1}%) to ({x2}%, {y2}%), evaluating sky visibility, natural coverage, landscape variety, elevation changes, and composite scenic quality.",
            "Execute comprehensive landscape assessment for region ({x1}%, {y1}%) to ({x2}%, {y2}%), determining openness factors, naturalness metrics, diversity indices, terrain scores, and integrated scenic evaluation."
        ],
        'hard_openness_analysis': [
            "Provide comprehensive openness analysis for the region from ({x1}%, {y1}%) to ({x2}%, {y2}%), calculating SVF score, openness index, building density impact, terrain flatness, visual simplicity, and integrated openness score.",
            "Analyze spatial openness characteristics for area ({x1}%, {y1}%) to ({x2}%, {y2}%), evaluating sky visibility, openness metrics, urban obstruction effects, topographic smoothness, and overall openness rating.",
            "Perform detailed openness assessment for region ({x1}%, {y1}%) to ({x2}%, {y2}%), determining SVF contribution, openness indices, building interference, terrain uniformity, visual clarity, and composite openness value.",
            "Calculate comprehensive openness metrics for the specified area ({x1}%, {y1}%) to ({x2}%, {y2}%), including visibility scores, spatial openness, density penalties, flatness measures, and integrated openness evaluation.",
            "Execute integrated openness analysis for region ({x1}%, {y1}%) to ({x2}%, {y2}%), measuring sky access, openness factors, obstruction impacts, terrain consistency, visual simplicity, and overall openness score."
        ]
        }
        
        # Add test-mode specific question templates
        if mode == "test":
            self.HARD_QUESTION_TEMPLATES['hard_sun_exposure'].append("Which site demonstrates optimal photovoltaic potential through maximum celestial exposure?")
            self.HARD_QUESTION_TEMPLATES['hard_scenic_quality'].append("Which district exhibits superior aesthetic valuation metrics?")
            self.HARD_QUESTION_TEMPLATES['hard_pixel'].append("Compute the regional mean SVF for area {region}.")
            self.HARD_QUESTION_TEMPLATES['hard_grid_5×5'].append("Which cellular subdivision within the 5×5 matrix demonstrates peak atmospheric accessibility?")
            self.HARD_QUESTION_TEMPLATES['hard_metric'].append("Compute the {metric} statistical parameter for SVF distribution within spatial boundaries ({x1}%, {y1}%) to ({x2}%, {y2}%).")
            self.HARD_QUESTION_TEMPLATES['hard_ranking'].append("Establish hierarchical ordering of spatial zones by atmospheric visibility indices: {region1}, {region2}, {region3}.")
            self.HARD_QUESTION_TEMPLATES['hard_urban_analysis'].append("Execute multidimensional urban morphology assessment for territorial extent ({x1}%, {y1}%) to ({x2}%, {y2}%), incorporating structural density metrics, volumetric ratios, atmospheric visibility coefficients, boundary complexity indices, and composite urbanization parameters.")
            self.HARD_QUESTION_TEMPLATES['hard_scenic_analysis'].append("Perform comprehensive environmental quality evaluation for territorial zone ({x1}%, {y1}%) to ({x2}%, {y2}%), analyzing atmospheric accessibility factors, ecological composition ratios, biodiversity indices, topographical complexity metrics, chromatic harmony coefficients, and integrated aesthetic assessment parameters.")
            self.HARD_QUESTION_TEMPLATES['hard_openness_analysis'].append("Conduct systematic spatial permeability analysis for geographic extent ({x1}%, {y1}%) to ({x2}%, {y2}%), quantifying atmospheric accessibility indices, spatial liberation coefficients, structural impedance factors, topographical uniformity metrics, visual complexity parameters, and integrated spatial freedom assessment.")

    def _get_question_template(self, category, **kwargs):
        """Get question template for given category"""
        if hasattr(self, 'HARD_QUESTION_TEMPLATES') and category in self.HARD_QUESTION_TEMPLATES:
            templates = self.HARD_QUESTION_TEMPLATES[category]
            template = random.choice(templates)
            try:
                return template.format(**kwargs)
            except KeyError:
                return template
        return None
    def hardPixel(self):
        """領域平均SVF値を問う（Hard）
        変更: ピクセル単位ではなく、小さな領域の平均SVF値を求める質問に変更
        """
        h, w = self.svf_map.shape
        
        # 小さな領域サイズを設定（画像サイズの5-15%程度）
        min_region_size = max(10, min(h, w) // 20)  # 最小10ピクセル
        max_region_size = max(min_region_size + 5, min(h, w) // 7)  # 最大で画像の1/7
        region_size = random.randint(min_region_size, max_region_size)
        
        if h < region_size or w < region_size:
            if self._debug: tqdm_safe_print(f"[hardPixel] Map size ({h}x{w}) too small for region size {region_size}.")
            return None, None, None

        # ランダムな位置を選択
        y_start = random.randint(0, h - region_size)
        x_start = random.randint(0, w - region_size)
        
        # 領域の切り出し
        region_svf = self.svf_map[y_start:y_start+region_size, x_start:x_start+region_size]
        
        # 有効性チェック
        valid_mask = ~np.isnan(region_svf) & (region_svf > 0)
        if np.sum(valid_mask) < region_size * region_size * 0.7:  # 70%以上が有効である必要
            if self._debug: tqdm_safe_print("[hardPixel] Not enough valid pixels in selected region.")
            return None, None, None
        
        # 領域の平均SVF値を計算
        avg_svf_value = np.mean(region_svf[valid_mask])
        
        # 相対座標に変換
        rel_bbox = self._get_relative_bbox(y_start, x_start, region_size, region_size)
        region_str = f"[{rel_bbox[0]}%, {rel_bbox[1]}%, {rel_bbox[2]}%, {rel_bbox[3]}%]"
        
        # 質問テンプレートを更新（領域平均用）
        region_templates = [
            "What is the average SVF value for the region {region}?",
            "Calculate the mean Sky View Factor within the area {region}.",
            "Determine the average SVF measurement for region {region}.",
            "What is the regional average SVF value at {region}?",
            "Find the mean Sky View Factor value within the specified region {region}."
        ]
        
        template = random.choice(region_templates)
        question_text = template.format(region=region_str)
        
        question_text += f"\nNote: The coordinates are given as percentages of the image dimensions in [xmin%, ymin%, xmax%, ymax%] format."
        question_text += f"\nRegion size: {region_size}×{region_size} pixels"
        question_text += f"\n\nIMPORTANT: Calculate the average SVF value for all valid pixels within the specified region. Provide the exact result rounded to 1 decimal place. SVF value is between 0.0 and 1.0. Answer format: X.X"

        answer_str = f"{avg_svf_value:.1f}"

        debug_info = [
            f"Region: ({y_start}, {x_start}) -> {region_str}",
            f"Region size: {region_size}×{region_size} pixels",
            f"Valid pixels: {np.sum(valid_mask)}/{region_size*region_size} ({np.sum(valid_mask)/(region_size*region_size)*100:.1f}%)",
            f"Average SVF value: {avg_svf_value:.6f}",
            f"Answer: {answer_str}"
        ]
        
        question_info = {
            "question": question_text,
            "debug_info": debug_info,
            "avg_svf_value": float(avg_svf_value),
            "region_bbox": rel_bbox,
            "region_size": region_size,
            "valid_pixel_count": int(np.sum(valid_mask))
        }
        
        canonical_question = ['hard_pixel', rel_bbox, float(avg_svf_value), region_size]
        
        if self._debug:
            tqdm_safe_print(f"[hardPixel] Correct answer: {answer_str} for region {region_str}")
            tqdm_safe_print(f"[hardPixel] Region size: {region_size}×{region_size}, Valid pixels: {np.sum(valid_mask)}")

        question_info = self._enhance_question_diversity(question_info, canonical_question)
        
        return question_info, answer_str, canonical_question
    def _find_diverse_regions_for_ranking(self, region_size_percent_w=15, region_size_percent_h=15, min_svf_diff=0.15):
        """
        Find diverse regions for ranking questions
        """
        h, w = self.svf_map.shape
        region_w = int(w * region_size_percent_w / 100)
        region_h = int(h * region_size_percent_h / 100)
        
        regions = []
        max_attempts = 100
        attempts = 0
        
        while len(regions) < 3 and attempts < max_attempts:
            x = random.randint(0, max(1, w - region_w))
            y = random.randint(0, max(1, h - region_h))
            
            region_svf = self.svf_map[y:y+region_h, x:x+region_w]
            valid_mask = ~np.isnan(region_svf) & (region_svf > 0)
            if region_svf.size > 0 and np.sum(valid_mask) > 0:
                avg_svf = np.mean(region_svf[valid_mask])
                
                region_data = {
                    'label': f"Region {chr(65 + len(regions))}",
                    'bbox_percent': [
                        int(x * 100 / w), int(y * 100 / h),
                        int((x + region_w) * 100 / w), int((y + region_h) * 100 / h)
                    ],
                    'avg_svf': avg_svf
                }
                
                # Check for sufficient difference with existing regions
                is_diverse = True
                for existing in regions:
                    if abs(avg_svf - existing['avg_svf']) < min_svf_diff:
                        is_diverse = False
                        break
                
                if is_diverse:
                    regions.append(region_data)
            
            attempts += 1
        
        if len(regions) >= 3:
            max_diff = max(r['avg_svf'] for r in regions) - min(r['avg_svf'] for r in regions)
            return regions, max_diff
        else:
            return None, 0.0

    def _generate_dummy_choices(self, correct_value, num_choices=4):
        """
        Generate dummy choices for pixel value questions
        """
        choices = [correct_value]
        
        # Generate plausible alternative values
        for _ in range(num_choices - 1):
            # Add some noise to the correct value
            noise = random.uniform(-0.3, 0.3)
            dummy_value = max(0.0, min(1.0, correct_value + noise))
            # Round to 4 decimal places
            dummy_value = round(dummy_value, 4)
            if dummy_value not in choices:
                choices.append(dummy_value)
        
        # Fill up to num_choices if needed
        while len(choices) < num_choices:
            dummy_value = round(random.uniform(0.0, 1.0), 4)
            if dummy_value not in choices:
                choices.append(dummy_value)
        
        return choices

    def _get_relative_point(self, y, x):
        """Convert pixel coordinates to relative percentage coordinates with decimal precision"""
        h, w = self.svf_map.shape
        rel_x = round((x / w) * 100, 1)  # 小数点第一位まで
        rel_y = round((y / h) * 100, 1)  # 小数点第一位まで
        return [rel_x, rel_y]

    def _get_question_template(self, category, **kwargs):
        """Get question template for given category"""
        if hasattr(self, 'HARD_QUESTION_TEMPLATES') and category in self.HARD_QUESTION_TEMPLATES:
            templates = self.HARD_QUESTION_TEMPLATES[category]
            template = random.choice(templates)
            try:
                return template.format(**kwargs)
            except KeyError:
                return template
        return None

    def hardRanking(self):
        #  確率的難易度調整: 20-30%の質問のみ難しい条件を適用
        use_hard_conditions = random.random() < 0.25  # 25%の確率で難しい条件
        
        overall_attempts = 0
        MAX_OVERALL_ATTEMPTS = 50 if use_hard_conditions else 30  # 難しい場合は試行回数増加
        best_regions = None
        max_svf_diff = 0.0
        efficiency_attempts = 0

        # 難易度に応じた閾値設定
        if use_hard_conditions:
            base_min_diff = 0.08  # 難しい条件: 高い差分要求
            early_exit_threshold = 0.06  # 厳格な早期終了
            if self._debug:
                tqdm_safe_print(f"[hardRanking] Using HARD conditions (25% chance)")
        else:
            base_min_diff = 0.15  # 元の条件: より高い初期要求
            early_exit_threshold = 0.06  # 元の条件
            if self._debug:
                tqdm_safe_print(f"[hardRanking] Using STANDARD conditions (75% chance)")

        while overall_attempts < MAX_OVERALL_ATTEMPTS:
            efficiency_attempts = overall_attempts + 1
            region_size_w = random.uniform(10, 25)
            region_size_h = random.uniform(10, 25)
            
            # 条件に応じた最小差分の設定
            if use_hard_conditions:
                min_diff = max(0.03, base_min_diff - (0.01 * overall_attempts))  # 段階的緩和
            else:
                min_diff = max(0.05, base_min_diff - (0.02 * overall_attempts))  # 元の条件: 0.15から0.05まで段階的緩和
            
            current_best_regions, current_max_svf_diff = self._find_diverse_regions_for_ranking(
                region_size_percent_w=region_size_w,
                region_size_percent_h=region_size_h,
                min_svf_diff=min_diff
            )
            
            if current_best_regions and current_max_svf_diff > max_svf_diff:
                max_svf_diff = current_max_svf_diff
                best_regions = current_best_regions
                if self._debug:
                    tqdm_safe_print(f"  [hardRanking] New best regions found with SVF diff: {max_svf_diff:.3f} (Attempt {overall_attempts+1})")

            overall_attempts += 1
            # 早期終了条件
            if max_svf_diff > early_exit_threshold: 
                break

        if not best_regions:
            if self._debug:
                tqdm_safe_print("[hardRanking] Could not find three regions with sufficient SVF difference.")
            #  フォールバック: 最低限の差でも受け入れる
            try:
                fallback_regions, fallback_diff = self._find_diverse_regions_for_ranking(
                    region_size_percent_w=20,
                    region_size_percent_h=20,
                    min_svf_diff=0.01  # 極めて小さい差でも受け入れ
                )
                if fallback_regions and len(fallback_regions) >= 3:
                    best_regions = fallback_regions
                    if self._debug:
                        tqdm_safe_print(f"[hardRanking] Using fallback regions with diff: {fallback_diff:.3f}")
            except:
                pass
                
            if not best_regions:
                return None, None, None

        # バイアス防止: 正解地域を特定してから統計的ラベル配布
        # 最高SVF地域を特定
        highest_region = max(best_regions, key=lambda item: item['avg_svf'])
        
        # 地域データ構造を調整（Region A/B/C/D対応）
        for region in best_regions:
            region['is_correct'] = (region == highest_region)
        
        # assign_balanced_labels適用して統計的バイアス防止
        best_regions = assign_balanced_labels(best_regions, highest_region)
        
        # SVFでソートして正解順序を決定（新しいラベルで）
        best_regions_sorted = sorted(best_regions, key=lambda item: item['avg_svf'], reverse=True)
        correct_ranking_labels = [item['label'] for item in best_regions_sorted]
        answer_str = ", ".join([f"Region {label}" for label in correct_ranking_labels])

        # 質問表示: 常にA, B, C順で統一（可読性向上）
        question_regions_sorted = sorted(best_regions, key=lambda x: x['label'])
        
        # Phase 1&2: 質問テンプレートの多様化とGPTリフレーズング
        base_question = self._get_question_template('hard_ranking')
        if base_question is None:
            base_question = "Rank these three regions by their SVF values from highest to lowest: Region A, Region B, Region C."
        
        # 質問文: 常にA, B, C順で表示（統一性確保）
        question_text = base_question
        
        question_text += "\n\nPlease consider the following regions for comparison:\n"
        for region_data in question_regions_sorted:
            bbox = region_data['bbox_percent']
            question_text += f"- {region_data['label']}: [xmin={bbox[0]}%, ymin={bbox[1]}%, xmax={bbox[2]}%, ymax={bbox[3]}%]\n"
        
        question_text += f"\nBased on the SVF analysis, rank these regions from highest to lowest SVF value.\n"
        question_text += f"Format your answer as: Region A, Region B, Region C"
        
        debug_info = [f"Correct Ranking: {answer_str}"]
        # SVF順でデバッグ情報を表示
        for r_data in best_regions_sorted:
            debug_info.append(f"  {r_data['label']}: avg_svf={r_data['avg_svf']:.3f}, BBox={r_data['bbox_percent']}")
        debug_info.append(f"Efficiency: Found suitable regions after {efficiency_attempts} overall attempts.")
        
        # Region A, Region B, Region C形式の選択肢を生成（全可能な順列）
        region_labels = ['Region A', 'Region B', 'Region C']
        all_permutations = list(itertools.permutations(region_labels))
        choices_str = [', '.join(perm) for perm in all_permutations]
        
        question_info = {
            "question": question_text,
            "choices": choices_str,
            "debug_info": debug_info,
            "efficiency_attempts": efficiency_attempts
        }
        
        canonical_question = ['hard_ranking', {r['label']: r['bbox_percent'] for r in best_regions}, correct_ranking_labels]
        question_info = self._enhance_question_diversity(question_info, canonical_question)
        return question_info, answer_str, canonical_question
    def hardPixelFromRE(self):
        """RE(Relative Elevation)データからピクセル単位の詳細分析"""
        if not PIXEL_LEVEL_AVAILABLE:
            if self._debug: tqdm_safe_print("[hardPixelFromRE] Pixel-level implementation not available.")
            return None, None, None
        
        # ConstructSVFQuestionから基本的なピクセル分析を呼び出し
        pixel_impl = ConstructSVFQuestion(
            self.svf_map, 
            self.height_map, 
            self.segmentation_map,
            debug=self._debug
        )
        
        # 基本的なピクセル分析を実行
        pixel_result = pixel_impl.pixelSVF()
        if pixel_result is None or pixel_result[0] is None:
            if self._debug: tqdm_safe_print("[hardPixelFromRE] Base pixel analysis failed.")
            return None, None, None
        
        base_question_info, base_answer, base_canonical = pixel_result
        
        # ハードカテゴリ向けの質問に変換
        if isinstance(base_question_info, dict) and 'question' in base_question_info:
            enhanced_question = base_question_info['question']
            enhanced_question += f"\n\nADVANCED ANALYSIS: This question requires precise pixel-level SVF calculation."
            enhanced_question += f"\nConsider local topographic variations and micro-scale environmental factors."
            enhanced_question += f"\nProvide the result with high precision (1 decimal place)."
            
            # Debug情報を拡張
            debug_info = base_question_info.get('debug_info', [])
            debug_info.extend([
                "[HARD] Enhanced with topographic analysis",
                "[HARD] Micro-scale environmental factors considered",
                "[HARD] High-precision calculation required"
            ])
            
            enhanced_question_info = {
                "question": enhanced_question,
                "debug_info": debug_info
            }
            
            # 元の情報を保持
            for key, value in base_question_info.items():
                if key not in enhanced_question_info:
                    enhanced_question_info[key] = value
            
            enhanced_canonical = ['hard_pixel_from_re'] + (base_canonical if isinstance(base_canonical, list) else [base_canonical])
            
            if self._debug:
                tqdm_safe_print(f"[hardPixelFromRE] Enhanced pixel analysis completed")
                for item in debug_info:
                    if item.startswith("[HARD]"):
                        tqdm_safe_print(f"  {item}")
            
            return enhanced_question_info, base_answer, enhanced_canonical
        
        # フォールバック: 基本的なハードピクセル質問
        return self.hardPixel()