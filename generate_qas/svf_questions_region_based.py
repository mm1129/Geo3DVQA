import numpy as np
import random
import os
import math
from svf_questions_rgb_estimated import ConstructSVFQuestionRGB, tqdm_safe_print, write_json_line, select_choices_with_diversity
from utils import bias_free_shuffle
import argparse
import json
from PIL import Image
from tqdm import tqdm
import time
from multiprocessing import Pool
import sys
from datetime import datetime
from utils import add_short_instruction
import itertools

# Import pixel-level implementations from svf_questions_re.py
try:
    from svf_questions_re import ConstructSVFQuestion
    PIXEL_LEVEL_AVAILABLE = False
    tqdm_safe_print("Warning: svf_questions_re.py not available. Pixel-level questions will be skipped.")
except ImportError:
    PIXEL_LEVEL_AVAILABLE = False

def assign_balanced_labels(regions, target_region):
    """
    正解位置を統計的に均等分布させるためのラベル配布関数
    
    Args:
        regions: 地域リスト
        target_region: 正解地域
        
    Returns:
        ラベルが再配布された地域リスト
    """
    labels = ['A', 'B', 'C', 'D']
    available_labels = labels[:len(regions)]
    
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

class ConstructSVFQuestionRegionBased(ConstructSVFQuestionRGB):
    def __init__(self, estimated_svf_map, estimated_height_map=None, estimated_segmentation_map=None, rgb_image=None, file_path=None, cnt=0, debug=False, use_gpt_rephrase=False, openai_api_key=None, hard_ratio=0.0, mode="train", coordinate_answers=False, balanced_categories=False):
        # 親クラスの初期化メソッドを呼び出す
        super().__init__(
            estimated_svf_map,
            estimated_height_map,
            estimated_segmentation_map,
            rgb_image,
            file_path,
            cnt,
            debug,
            use_gpt_rephrase,
            openai_api_key,
            hard_ratio,
            mode,
            coordinate_answers,
            balanced_categories
        )
        
        REGION_CATEGORY_WEIGHTS = {
            
            # 極めて悪い成績 (20%以下) - 5倍重み
            'grid_5×5': 5.0,                        # 11.40% (図表: "Grid 5×5")
            'best_landcover_balance': 5.0,          # 23.31% (図表: "Best landcover balance")
            
            # 悪い成績 (20-40%) - 3倍重み  
            'visibility_range': 3.0,                # 31.23% (図表: "Visibility Range")
            'building_density': 3.0,                # 36.24% (図表: "Building Density")
            'precise_svf': 3.0,                     # 40.75% (図表: "Precise SVF")
            'spatial_openness': 3.0,                # 40.93% (図表: "Spatial Openness")
            
            # 普通 (40-60%) - 2倍重み
            'sky_visibility': 2.0,                  # 46.18% (図表: "Sky Visibility")
            'regional_svf_variability': 2.0,        # 51.64% (図表: "Regional SVF variability")
            
            # 良好 (60%以上) - 標準重み
            'grid_extremes': 1.0,                   # 63.52% (図表: "Grid Extremes")
            'region_ranking': 1.0,                  # 93.51% (図表: "Region Ranking")
            'sun_exposure': 1.0,                    # (図表: "Sun Exposure")
            
            # Legacy names for backward compatibility
            'hard_grid_5×5': 5.0,
            'natural_artificial_ratio': 5.0,
            'land_composition': 5.0,
            'urban_density': 3.0,
            'hard_pixel': 3.0,
            'openness_assessment': 3.0,
            'svf_extreme_grid': 1.0,
            'hard_ranking': 1.0,
        }

        self.REGION_QUESTION_TYPES = {
            'regional_svf_variability': [REGION_CATEGORY_WEIGHTS.get('svf_region_analysis', 1.0), self.svfRegionAnalysis, 1],
        }
        
        # Update QUESTION_TYPES with region-based types
        self.QUESTION_TYPES.update(self.REGION_QUESTION_TYPES)

    def _get_dynamic_ensure_max(self, category_name=None):
        """Always include maximum value in choices"""
        return True

    def _get_relative_bbox(self, y_start, x_start, height, width):
        """Convert pixel coordinates to relative coordinates (0-100)"""
        h, w = self.svf_map.shape
        xmin = round((x_start / w) * 100)
        ymin = round((y_start / h) * 100)
        xmax = round(((x_start + width) / w) * 100)
        ymax = round(((y_start + height) / h) * 100)
        return [xmin, ymin, xmax, ymax]
    def _get_grid_cell_name(self, row, col):
        """Convert grid position to cell name (e.g., 'top left', 'middle')"""
        positions = ['top', 'middle', 'bottom']
        positions2 = ['left', 'middle', 'right']
        return f"{positions[row]} {positions2[col]}"

    def _get_relative_point(self, y, x):
        """Convert pixel coordinates to relative coordinates (0-100) with decimal precision"""
        h, w = self.svf_map.shape
        rel_x = round((x / w) * 100, 1)
        rel_y = round((y / h) * 100, 1)
        return [rel_x, rel_y]

    def svfRegionAnalysis(self):
        """
        相対座標（%）ベースのSVF変動性分析質問
        
        SVF variability/consistency analysis - statistical variance evaluation
        """
        if self._debug:
            tqdm_safe_print(f"DEBUG: svfRegionAnalysis method called")
        
        h, w = self.svf_map.shape
        
        # 3つの領域を生成してSVF変動性を比較
        min_size = min(h, w) // 10
        max_size = min(h, w) // 4
        
        # bbox大きさ統一: 固定サイズを使用
        fixed_region_size = (min_size + max_size) // 2  # 平均サイズを採用
        
        regions = []
        max_attempts = 100
        
        # 3つの有効な領域を見つける
        for attempt in range(max_attempts):
            temp_regions = []  # 一時的な領域リスト
            for i in range(3):
                # bbox大きさ統一: 固定サイズを使用
                region_h = fixed_region_size
                region_w = fixed_region_size
                y_start = random.randint(0, h - region_h)
                x_start = random.randint(0, w - region_w)
                
                region = self.svf_map[y_start:y_start+region_h, x_start:x_start+region_w]
                valid_mask = ~np.isnan(region) & (region > 0)
                
                if np.sum(valid_mask) > (region_h * region_w * 0.5):
                    bbox = self._get_relative_bbox(y_start, x_start, region_h, region_w)
                    mean_svf = np.mean(region[valid_mask])
                    std_svf = np.std(region[valid_mask])
                    temp_regions.append({
                        'bbox': bbox,
                        'mean_svf': mean_svf,
                        'std_svf': std_svf
                    })
                    
            if len(temp_regions) >= 3:
                # 標準偏差の範囲をチェック
                std_values = [r['std_svf'] for r in temp_regions]
                if (max(std_values) - min(std_values)) >= 0.05:  # 十分な変動性の差があるかチェック（親クラスのMIN_SCORE_GAP_STANDARDと統一）
                    regions = temp_regions[:3]  # 最初の3つの領域を使用
                    break
        
        if len(regions) < 3:
            if self._debug:
                tqdm_safe_print(f"DEBUG: svfRegionAnalysis - No valid regions found, returning None")
            if self._debug: tqdm_safe_print("[svfRegionAnalysis] Could not find three valid regions with sufficient variation in variability.")
            return None, None, None

        # 領域に適切なラベルを付ける（A, B, C）
        for i, region in enumerate(regions):
            region['label'] = chr(65 + i)  # A, B, C

        # 質問タイプをランダムに選択（最も変動が大きい/小さい）
        question_type = random.choice(['most_variable', 'most_consistent'])
        
        if question_type == 'most_variable':
            target_region = max(regions, key=lambda r: r['std_svf'])
            analysis_type = "highest SVF standard deviation"
        else:
            target_region = min(regions, key=lambda r: r['std_svf'])
            analysis_type = "lowest SVF standard deviation"
        
        base_question = self._get_question_template('svf_region_analysis', analysis_type=analysis_type)
        if base_question is None:
            if question_type == 'most_variable':
                base_question = "Among these regions, which one shows the highest SVF variability (standard deviation)?"
            else:
                base_question = "Among these regions, which one shows the most consistent SVF values (lowest standard deviation)?"
        
        regions = assign_balanced_labels(regions, target_region)
        regions = sorted(regions, key=lambda x: x['label'])  # 常にA,B,C,D順で表示
        
        question_text = base_question + "\n\n"
        for region in regions:
            question_text += f"Region {region['label']}: [xmin={region['bbox'][0]}%, ymin={region['bbox'][1]}%, "
            question_text += f"xmax={region['bbox'][2]}%, ymax={region['bbox'][3]}%]\n"
        
        question_text += f"\nPlease choose from:\n"
        for region in regions:
            question_text += f"Region {region['label']}\n"
        question_text += "Coordinate Guide: Each region shows [left%, top%, right%, bottom%] as percentage of image size.\n"
        question_text += "Think of the image like a map: [4%, 58%, 20%, 76%] means:\n"
        question_text += "• Start 4% from left edge, 58% down from top\n"
        question_text += "• End 20% from left edge, 76% down from top\n"
        question_text += "This creates a rectangular region in that area of the image."

        # 正解を新しいラベルで特定
        correct_region_after_labeling = next(r for r in regions if r == target_region)
        answer = f"Region {correct_region_after_labeling['label']}"

        # Debug情報を追加
        debug_info = [
            f"Analysis type: {analysis_type}",
            f"Target region: {target_region['label']} (mean={target_region['mean_svf']:.6f}, std={target_region['std_svf']:.6f})",
            f"Question type: {question_type}"
        ]
        
        # 全領域の統計をdebug_infoに追加
        for region in regions:
            debug_info.append(f"Region {region['label']}: mean={region['mean_svf']:.6f}, std={region['std_svf']:.6f}, bbox={region['bbox']}")
        
        question_info = {
            "question": question_text,
            "choices_bboxes": [r['bbox'] for r in regions],
            "choices": [f"Region {r['label']}" for r in regions],
            "debug_info": debug_info
        }
        
        if self._debug:
            tqdm_safe_print(f"DEBUG: svfRegionAnalysis - Question generation completed")
            tqdm_safe_print(f"DEBUG: question text length: {len(question_text)} chars")
            tqdm_safe_print(f"DEBUG: first 100 chars: {question_text[:100]}...")
        
        question_info = self._enhance_question_diversity(question_info, 'regional_svf_variability')
        
        if self._debug:
            tqdm_safe_print(f"DEBUG: After GPT rephrasing, question length: {len(question_info.get('question', ''))} chars")
            tqdm_safe_print(f"DEBUG: After GPT rephrasing, first 100 chars: {question_info.get('question', '')[:100]}...")

        if self._debug:
            tqdm_safe_print(f"[svfRegionAnalysis] Question type: {question_type}")
            tqdm_safe_print(f"  Target region: {target_region['label']}")
            tqdm_safe_print(f"  Target SVF std: {target_region['std_svf']:.3f}")
            for region in regions:
                tqdm_safe_print(f"  Region {region['label']}: mean = {region['mean_svf']:.3f}, std = {region['std_svf']:.3f}, bbox = {region['bbox']}")

        return question_info, answer, ['regional_svf_variability', question_type, [r['bbox'] for r in regions]]

    def chooseQuestionsToAsk(self, number_question=50):
        """
        リージョンベース版の質問選択メソッド
        親クラスの実装を継承しつつ、追加された質問タイプも考慮する
        """
        #  タイミング計測: 全体の生成時間を測定
        import time
        total_start_time = time.time()
        
        # 親クラスのchooseQuestionsToAskメソッドを呼び出し
        # これにより、既存の質問タイプと新しい質問タイプの両方が考慮される
        questions, answers, canonical_questions = super().chooseQuestionsToAsk(number_question)
        
        #  タイミング計測: 全体処理完了
        total_end_time = time.time()
        self.total_generation_time = total_end_time - total_start_time
        
        # デバッグ情報の出力
        if self._debug:
            tqdm_safe_print(f"[ConstructSVFQuestionRegionBased] Generated {len(questions)} questions")
            tqdm_safe_print(f"Available question types: {list(self.QUESTION_TYPES.keys())}")
            tqdm_safe_print(f"Region-specific types: {list(self.REGION_QUESTION_TYPES.keys())}")
            
            #  タイミングサマリーの表示
            self.print_timing_summary()
        
        return questions, answers, canonical_questions

    def print_timing_summary(self):
        """
         カテゴリ別タイミング分析とサマリー表示
        
        実行時間の詳細な分析により、パフォーマンスボトルネックを特定
        """
        if not hasattr(self, 'category_timing') or not self.category_timing:
            tqdm_safe_print(" Timing data not available")
            return
        
        tqdm_safe_print("\n" + "="*60)
        tqdm_safe_print(" Category Performance Analysis (Region-Based Questions)")
        tqdm_safe_print("="*60)
        
        # 統計計算用のデータ収集
        all_times = []
        category_stats = {}
        
        for category, times in self.category_timing.items():
            if times:  # 空のリストをスキップ
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                total_time = sum(times)
                
                category_stats[category] = {
                    'avg': avg_time,
                    'min': min_time, 
                    'max': max_time,
                    'total': total_time,
                    'count': len(times),
                    'times': times
                }
                all_times.extend(times)
        
        if not category_stats:
            tqdm_safe_print("No valid timing data available")
            return
        
        # 全体統計
        total_category_time = sum(stats['total'] for stats in category_stats.values())
        total_questions = sum(stats['count'] for stats in category_stats.values())
        overall_avg = sum(all_times) / len(all_times) if all_times else 0
        
        tqdm_safe_print(f"\nSummary:")
        tqdm_safe_print(f"  Total execution time: {self.total_generation_time:.2f}s")
        tqdm_safe_print(f"  Total category processing time: {total_category_time:.2f}s")
        tqdm_safe_print(f"  Total questions: {total_questions}")
        tqdm_safe_print(f"  Average question generation time: {overall_avg:.3f}s/question")
        if self.total_generation_time > 0:
            tqdm_safe_print(f"  Overall throughput: {total_questions/self.total_generation_time:.2f} questions/s")
        
        # カテゴリを平均時間でソート（遅い順）
        sorted_categories = sorted(category_stats.items(), key=lambda x: x[1]['avg'], reverse=True)
        
        tqdm_safe_print(f"\nCategory Details (sorted by average time):")
        tqdm_safe_print(f"{'Category':<25} {'Avg':<8} {'Min':<8} {'Max':<8} {'Total':<8} {'Count':<4}")
        tqdm_safe_print("-" * 70)
        
        for category, stats in sorted_categories:
            # リージョンベース特有のカテゴリにマーカーを追加
            is_region_specific = category in self.REGION_QUESTION_TYPES
            marker = "" if is_region_specific else "  "
            
            category_display = category[:23] if len(category) <= 23 else category[:20] + "..."
            tqdm_safe_print(f"{marker}{category_display:<23} {stats['avg']:.3f}s  {stats['min']:.3f}s  "
                          f"{stats['max']:.3f}s  {stats['total']:.2f}s  {stats['count']:>3}")
        
        # パフォーマンス問題の特定
        tqdm_safe_print(f"\nPerformance Analysis:")
        
        # 遅いカテゴリ (3秒以上)
        slow_categories = [(cat, stats) for cat, stats in category_stats.items() if stats['avg'] > 3.0]
        if slow_categories:
            tqdm_safe_print(f"  Slow categories (≥3s):")
            for cat, stats in sorted(slow_categories, key=lambda x: x[1]['avg'], reverse=True):
                is_region = "" if cat in self.REGION_QUESTION_TYPES else "  "
                tqdm_safe_print(f"    {is_region}{cat}: {stats['avg']:.2f}s (max: {stats['max']:.2f}s)")
        
        # 変動が大きいカテゴリ
        variable_categories = []
        for cat, stats in category_stats.items():
            if stats['count'] > 1:  # 複数回実行されたもののみ
                variance = sum((t - stats['avg'])**2 for t in stats['times']) / len(stats['times'])
                std_dev = variance ** 0.5
                cv = std_dev / stats['avg'] if stats['avg'] > 0 else 0  # 変動係数
                if cv > 0.5:  # 変動係数が50%以上
                    variable_categories.append((cat, cv, std_dev))
        
        if variable_categories:
            tqdm_safe_print(f"  Categories with high time variation:")
            for cat, cv, std_dev in sorted(variable_categories, key=lambda x: x[1], reverse=True):
                is_region = "" if cat in self.REGION_QUESTION_TYPES else "  "
                tqdm_safe_print(f"    {is_region}{cat}: CV={cv:.1%}, σ={std_dev:.3f}s")
        
        # リージョンベース特有のカテゴリの分析
        region_categories = {cat: stats for cat, stats in category_stats.items() 
                           if cat in self.REGION_QUESTION_TYPES}
        if region_categories:
            region_total_time = sum(stats['total'] for stats in region_categories.values())
            region_total_count = sum(stats['count'] for stats in region_categories.values())
            
            tqdm_safe_print(f"\nRegion-Based Category Analysis:")
            tqdm_safe_print(f"  Number of categories: {len(region_categories)}")
            tqdm_safe_print(f"  Total execution time: {region_total_time:.2f}s ({region_total_time/total_category_time:.1%})")
            tqdm_safe_print(f"  Total questions: {region_total_count} ({region_total_count/total_questions:.1%})")
            if region_total_count > 0:
                tqdm_safe_print(f"  Average generation time: {region_total_time/region_total_count:.3f}s/question")
        
        # 推奨事項
        tqdm_safe_print(f"\nOptimization Recommendations:")
        if slow_categories:
            slowest_cat = max(slow_categories, key=lambda x: x[1]['avg'])
            tqdm_safe_print(f"  Priority: Optimize {slowest_cat[0]} ({slowest_cat[1]['avg']:.2f}s)")
        
        if variable_categories:
            most_variable = max(variable_categories, key=lambda x: x[1])
            tqdm_safe_print(f"  Stability: Reduce variation in {most_variable[0]} (CV={most_variable[1]:.1%})")
        
        if total_category_time / self.total_generation_time < 0.8:
            overhead = self.total_generation_time - total_category_time
            tqdm_safe_print(f"  Overhead: Optimize system processing time ({overhead:.2f}s)")
        
        tqdm_safe_print("="*60)

def main():
    """GeoNRWデータセットから質問を生成するメイン関数"""
    # グローバルランダムシードの設定
    GLOBAL_SEED = 42  # 固定シード値
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="SVFデータから質問と回答を生成するスクリプト")
    parser.add_argument("--svf_path", type=str,
                      help="SVFファイルが含まれるルートディレクトリのパス")
    parser.add_argument("--geonrw_path", type=str,
                      help="GeoNRWデータ（高さ、セグメンテーション、RGB）のルートパス")
    parser.add_argument("--mode", type=str, choices=["train", "test", "both"], default="both",
                      help="処理モード: 'train'、'test'、または 'both'（両方）")
    parser.add_argument("--versions", nargs='+', default=["medium", "large"],
                      help="生成するデータセットのバージョン ('medium', 'large')")
    parser.add_argument("--question_types", type=str, default="all",
                      help="生成する質問タイプ（カンマ区切り、または'all'）")
    parser.add_argument("--visual_hints", action="store_true",
                      help="視覚的ヒントを質問に追加する")
    parser.add_argument("--simplify", action="store_true",
                      help="質問と回答を簡略化する")
    parser.add_argument("--output_dir", type=str, default="svf_qa_output_simple",
                      help="出力ディレクトリのパス（デフォルト: 'svf_qa_output')")
    parser.add_argument("--num_processes", type=int, default=0,
                      help="使用するプロセス数（デフォルト: 0=自動）")
    parser.add_argument("--batch_size", type=int, default=8,
                      help="バッチサイズ（デフォルト: 10）")
    parser.add_argument("--max_files", type=int, default=0,
                      help="処理する最大ファイル数（デフォルト: 0=すべて）")
    parser.add_argument("--update_plot_ids", action="store_true",
                      help="生成されたプロットファイルに質問IDを適用する")
    parser.add_argument("--conversation", action="store_true", default=True,
                      help="出力を会話形式のJSONにする")
    parser.add_argument("--skip_plots", action="store_true", default=True,
                      help="プロット生成をスキップする")
    parser.add_argument("--debug", action="store_true", default=False,
                      help="デバッグモードを有効にする")
    # 新しいオプションを追加
    parser.add_argument("--with_hints", action="store_true",
                      help="ヒント付きの出力ファイルを生成する")
    parser.add_argument("--with_svf_array", action="store_true",
                      help="SVF配列データを含む出力ファイルを生成する")
    parser.add_argument("--with_svf_values", action="store_true",
                      help="選択肢のSVF値を含む出力ファイルを生成する")
    parser.add_argument("--seed", type=int, default=42,
                      help="乱数生成用のシード値（デフォルト: 42）")
    parser.add_argument("--reconstruct_svf", action="store_true",
                      help="SVF値を再構築する")
    args = parser.parse_args()
    
    # 1ファイルあたりの質問数を動的に設定するためのプレースホルダー
    args.questions_per_file = 50 # デフォルト値

    # コマンドライン引数からシード値を取得して設定
    GLOBAL_SEED = args.seed
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    
    tqdm_safe_print(f"Random seed: {GLOBAL_SEED}")
    
    start_time = time.time()
    
    # パス設定をコマンドライン引数から取得
    svf_base_path = args.svf_path 
    geonrw_path = args.geonrw_path
    svf_train = "../SynRS3D/GeoNRW_dsm/svf/skyview_umep_train"
    svf_test = "../SynRS3D/GeoNRW_dsm/svf/skyview_umep_test"
    geonrw_path = "../SynRS3D/GeoNRW_dsm"
    
    tqdm_safe_print("Starting processing...")
    
    # 出力ディレクトリの設定
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    tqdm_safe_print(f"Output directory: {output_dir}")

    # バージョンごとの質問数設定
    version_config = {
        "medium": {"train": 12500, "test": 3500},
        "large": {"train": 80000, "test": 20000}
    }
    
    versions_to_process = args.versions
    if "all" in versions_to_process:
        versions_to_process = ["medium", "large"]

    for version in versions_to_process:
        if version not in version_config:
            tqdm_safe_print(f"Warning: Configuration for version '{version}' not found. Skipping.")
            continue
        
        tqdm_safe_print(f"\n===== Starting {version.upper()} version generation =====")
        
        # 処理モードに応じて実行
        modes_to_process = []
        if args.mode == "train":
            modes_to_process.append(("train", svf_train))
        elif args.mode == "test":
            modes_to_process.append(("test", svf_test))
        elif args.mode == "both":
            modes_to_process.append(("train", svf_train))
            modes_to_process.append(("test", svf_test))
        
        total_question_count = 0
        total_files_processed = 0
        total_files_count = 0
        
        # 各モードを処理
        for mode_name, svf_path in modes_to_process:
            tqdm_safe_print(f"\nStarting {mode_name.upper()} data processing...")
            
            # 出力ファイルパスの設定（モード名とバージョン名を含む）
            date = datetime.now().strftime("%m%d")
            file_suffix = f"{mode_name}_{version}_{date}"
            question_file = os.path.join(output_dir, f"svf_questions_{file_suffix}.jsonl")
            answer_file = os.path.join(output_dir, f"svf_answers_{file_suffix}.jsonl")
            detailed_file = os.path.join(output_dir, f"svf_detailed_{file_suffix}.json")
            conversation_file = os.path.join(output_dir, f"svf_conversation_{file_suffix}.json") if args.conversation else None
            
            # 既存のファイルを削除（新規作成のため）
            for file in [question_file, answer_file, detailed_file, conversation_file]:
                if file and os.path.exists(file):
                    os.remove(file)
            
            # 処理対象のファイルを収集
            all_files_for_mode = []
            if not os.path.exists(svf_path):
                tqdm_safe_print(f"Warning: SVF path not found: {svf_path}")
                continue
            for area in os.listdir(svf_path):
                area_path = os.path.join(svf_path, area)
                if not os.path.isdir(area_path):
                    continue
                    
                svf_files = [f for f in os.listdir(area_path) if f.endswith('_svf_umep.tif')]
                all_files_for_mode.extend([(os.path.join(area_path, f), geonrw_path, area, args) for f in svf_files])
            
            num_available_files = len(all_files_for_mode)
            if num_available_files == 0:
                tqdm_safe_print(f"Warning: No files found for processing in {mode_name} mode. Skipping.")
                continue

            target_question_count = version_config[version][mode_name]
            
            # ファイルあたりの質問数を計算（切り上げ）
            questions_per_file = (target_question_count + num_available_files - 1) // num_available_files
            args.questions_per_file = questions_per_file
            
            tqdm_safe_print(f"Target questions: {target_question_count}, Available files: {num_available_files}")
            tqdm_safe_print(f"Questions per file: {questions_per_file}")
            
            # 処理するファイルを決定 (max_filesが指定されていなければすべて)
            if args.max_files > 0:
                files_to_process = all_files_for_mode[:args.max_files]
                tqdm_safe_print(f"Processing {len(files_to_process)}/{num_available_files} files (max_files limit applied)")
            else:
                files_to_process = all_files_for_mode
                tqdm_safe_print(f"Processing all {num_available_files} files")

            total_files_count += len(files_to_process)
            
            # マルチプロセス処理の設定
            num_processes = args.num_processes if args.num_processes > 0 else max(1, os.cpu_count() - 1)
            num_processes = min(num_processes, os.cpu_count())
            tqdm_safe_print(f"Using {num_processes} process(es)")
            
            # バッチ処理の準備
            batch_size = args.batch_size
            file_batches = [files_to_process[i:i + batch_size] for i in range(0, len(files_to_process), batch_size)]
            tqdm_safe_print(f"Batches: {len(file_batches)} (batch size: {batch_size})")
            
            question_id_counter = 0
            files_processed = 0
            from svf_questions_re import process_single_file
            # マルチプロセスで処理実行
            conversation_data = []
            try:
                from functools import partial
                date = datetime.now().strftime("%m%d%H%M")
                svf_plots = f"svf_plots_{date}"
                os.makedirs(svf_plots, exist_ok=True)
                
                # svf_plotsを固定した関数を作成
                process_file_with_plots = partial(process_single_file, svf_plots=svf_plots)
                
                # 修正したプール処理
                with Pool(processes=num_processes) as pool:
                    with tqdm(total=len(files_to_process), desc=f"{mode_name} ({version}) ファイル処理中") as pbar:
                        for result in pool.imap_unordered(process_file_with_plots, files_to_process):
                            if result['success']:
                                # 質問・回答をJSONLファイルに書き込む
                                for q, a, cat in zip(result['questions'], result['answers'], result['canonical_questions']):
                                    question_id_counter += 1
                                    
                                    # 画像パスの設定
                                    image_path = f"{result['area']}/{os.path.basename(result['image_path'])}"
                                    
                                    # 回答用データ
                                    data_ans = { "question_id": question_id_counter, "image": image_path, "answer": a, "text": q["question"] if isinstance(q, dict) else q, "category": cat[0] if cat else "unknown" }
                                    if isinstance(q, dict):
                                        if "choices_bboxes" in q: data_ans["choices_bboxes"] = q["choices_bboxes"]
                                        if "choices_coords" in q: data_ans["choices_coords"] = q["choices_coords"]
                                    
                                    # 質問用データ
                                    data_ques = { "question_id": question_id_counter, "image": image_path, "text": q["question"] if isinstance(q, dict) else q, "category": cat[0] if cat else "unknown" }
                                    if isinstance(q, dict):
                                        if "choices_bboxes" in q: data_ques["choices_bboxes"] = q["choices_bboxes"]
                                        if "choices_coords" in q: data_ques["choices_coords"] = q["choices_coords"]
                                    
                                    # 詳細情報用データ（スコア情報を含む）
                                    data_detailed = { "question_id": question_id_counter, "image": image_path, "answer": a, "text": q["question"] if isinstance(q, dict) else q, "category": cat[0] if cat else "unknown", "debug_info": q.get("debug_info", []) if isinstance(q, dict) else [] }
                                    if isinstance(q, dict):
                                        if "choices_bboxes" in q: data_detailed["choices_bboxes"] = q["choices_bboxes"]
                                        if "choices_coords" in q: data_detailed["choices_coords"] = q["choices_coords"]
                                        if "scores" in q: data_detailed["scores"] = q["scores"]
                                    
                                    write_json_line(answer_file, data_ans)
                                    write_json_line(question_file, data_ques)
                                    write_json_line(detailed_file, data_detailed)
                                    
                                    # 会話形式データを作成
                                    if args.conversation:
                                        conv_data = {
                                            "id": f"{version}-{str(question_id_counter).zfill(12)}",
                                            "image": image_path,
                                            "conversations": [{"from": "human", "value": f"<image>\n{q['question'] if isinstance(q, dict) else q}"}, {"from": "gpt", "value": str(a)}],
                                            "metadata": {"category": cat[0] if cat else "unknown", "mode": mode_name, "version": version}
                                        }
                                        conversation_data.append(conv_data)
                                    
                                files_processed += 1
                            else:
                                tqdm_safe_print(f"\nError occurred: {result['file_path']}, {result['error']}")
                            pbar.update(1)
                
                # 会話形式データを一度にファイルに書き込む
                if args.conversation and conversation_data:
                    with open(conversation_file, 'w', encoding='utf-8') as f:
                        json.dump(conversation_data, f, ensure_ascii=False, indent=2)
                    tqdm_safe_print(f"Conversation data saved: {conversation_file}")
                    
            except KeyboardInterrupt:
                tqdm.write("\nInterrupting processing...")
                sys.exit(1)
            
            total_question_count += question_id_counter
            total_files_processed += files_processed
            
            tqdm_safe_print(f"\n{mode_name.upper()} ({version.upper()}) processing completed")
            tqdm_safe_print(f"  Files processed: {files_processed}/{len(files_to_process)}")
            tqdm_safe_print(f"  Questions generated: {question_id_counter}")
            tqdm_safe_print(f"  Output files:")
            tqdm_safe_print(f"    - Questions: {question_file}")
            tqdm_safe_print(f"    - Answers: {answer_file}")
            tqdm_safe_print(f"    - Details: {detailed_file}")
            if args.conversation and conversation_file:
                tqdm_safe_print(f"    - Conversation: {conversation_file}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    tqdm_safe_print("\n=== All processing completed ===")
    tqdm_safe_print(f"Processing time: {processing_time:.2f}s ({processing_time/60:.2f} minutes)")
    tqdm_safe_print(f"Total files processed: {total_files_processed}/{total_files_count}")
    tqdm_safe_print(f"Total questions generated: {total_question_count}")
    if processing_time > 0:
        tqdm_safe_print(f"Generation rate: {total_question_count/processing_time:.2f} questions/s")
        tqdm_safe_print(f"File processing rate: {total_files_processed/processing_time:.2f} files/s")
    tqdm_safe_print(f"Output files are listed above for each mode")

if __name__ == "__main__":
    main() 