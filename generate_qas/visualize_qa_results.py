import os
import cv2
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import re
from typing import List, Dict, Any, Tuple, Optional
# 追加: rasterioの条件付きインポート（オプション）
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
from PIL import Image
from matplotlib.patches import Rectangle, Circle
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import re

class QAVisualizer:
    def __init__(self, qa_json_path, image_dir, svf_dir=None):
        """
        Initialize the QA visualizer
        
        Args:
            qa_json_path (str): Path to the QA results JSON file
            image_dir (str): Directory containing the RGB images
            svf_dir (str): Directory containing the SVF images
        """
        self.qa_json_path = qa_json_path
        self.image_dir = image_dir
        self.svf_dir = svf_dir
        self.load_qa_results()
        
        # Define category groups
        self.point_categories = ['sun_exposure', 'hard_sun_exposure', 'sky_visibility', 'visibility_range']
        self.region_categories = ['urban_density', 'sun_exposure_percent', 'openness_assessment', 'svf_region_analysis', 'natural_artificial_ratio', 'scenic_quality']
        self.grid_categories = ['svf_grid_analysis', 'svf_extreme_grid']
        self.comparison_categories = ['svf_comparison', 'svf_average_comparison']
        
        # Define colors for visualization
        self.colors = {
            'correct': 'green',
            'incorrect': 'red',
            'choice': 'blue',
            'grid': 'yellow',
            'region': 'cyan',
            'comparison': 'magenta'
        }
        
        # Create SVF colormap
        self.svf_cmap = LinearSegmentedColormap.from_list(
            'svf', ['darkblue', 'blue', 'cyan', 'yellow', 'orange', 'red'], N=256)
        
        # Helper to shorten very long strings for titles
        def _shorten(text: str, max_chars: int = 60) -> str:
            if not isinstance(text, str):
                text = str(text)
            return text if len(text) <= max_chars else text[: max(0, max_chars - 1)] + '…'
        self._shorten = _shorten
        
    def load_qa_results(self):
        """Load QA results from JSON or JSONL file"""
        try:
            with open(self.qa_json_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Try to load as JSON array first
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    self.qa_results = data
                    return
                else:
                    # Single object, convert to list
                    self.qa_results = [data]
                    return
            except json.JSONDecodeError:
                # Failed as JSON, try as JSONL
                pass
            
            # Load as JSONL (line by line)
            self.qa_results = []
            for line in content.split('\n'):
                line = line.strip()
                if line:
                    try:
                        self.qa_results.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Skip invalid lines
                        continue
                        
        except Exception as e:
            print(f"Error loading QA results: {e}")
            self.qa_results = []
            
    def load_image(self, image_path):
        """Load RGB image with error handling"""
        # Extract area name from path (e.g., "aachen" from "aachen/123_456_rgb.jp2")
        full_path = os.path.join(self.image_dir, image_path)
        if not os.path.exists(full_path):
            # Try without directory structure
            image_name = os.path.basename(image_path)
            full_path = os.path.join(self.image_dir, image_name)
            
        # Check if file exists before opening
        if not os.path.exists(full_path):
            # Return placeholder image if file not found
            return np.ones((512, 512, 3), dtype=np.uint8) * 128  # Gray placeholder
            
        # Make sure numpy is imported
        import numpy as np
        try:
            # Ensure JP2 is decoded to 8-bit RGB consistently for visualization
            return np.array(Image.open(full_path).convert('RGB'))
        except Exception as e:
            print(f" Error loading image {full_path}: {e}")
            # Return placeholder image
            return np.ones((512, 512, 3), dtype=np.uint8) * 128
    
    def load_svf_image(self, rgb_image_path):
        """Load corresponding SVF image"""
        if not self.svf_dir:
            return None
            
        # Convert RGB path to SVF path
        # duisburg/335_5696_rgb.jp2 -> 335_5696_dem_svf_umep.tif
        base_name = os.path.basename(rgb_image_path)
        area_name = rgb_image_path.split('/')[0] if '/' in rgb_image_path else None
        print("area_name", area_name)
        if '_rgb.jp2' in base_name:
            svf_name = base_name.replace('_rgb.jp2', '_dem_svf_umep.tif')
        else:
            return None
            
        svf_path = os.path.join(self.svf_dir, area_name, svf_name)
        print("svf_path", svf_path)
        if os.path.exists(svf_path):
            try:
                svf_img = cv2.imread(svf_path, cv2.IMREAD_UNCHANGED)
                if svf_img is not None:
                    # 修正: SVFは通常グレースケールまたは単一チャンネルなのでBGR変換は不要
                    # しかし、3チャンネルの場合はBGR→RGB変換を適用
                    if len(svf_img.shape) == 3 and svf_img.shape[2] == 3:
                        svf_img = cv2.cvtColor(svf_img, cv2.COLOR_BGR2RGB)
                        print(f" Converting SVF image from BGR to RGB: {svf_path}")
                    return svf_img.astype(np.float32)
            except Exception as e:
                print(f"Error loading SVF image {svf_path}: {e}")
        return None
    
    def load_seg_image(self, rgb_image_path):
        """Load corresponding segmentation image"""
        # Convert RGB path to segmentation path
        # duisburg/335_5696_rgb.jp2 -> duisburg/335_5696_seg.tif
        seg_path = rgb_image_path.replace('_rgb.jp2', '_seg.tif')
        full_path = os.path.join(self.image_dir, seg_path)
        
        if not os.path.exists(full_path):
            # Try without directory structure
            seg_name = os.path.basename(seg_path)
            full_path = os.path.join(self.image_dir, seg_name)
            
        if os.path.exists(full_path):
            try:
                # 修正: rasterioを使用してTIFファイルを適切に読み込み
                try:
                    with rasterio.open(full_path) as src:
                        seg_img = src.read(1)  # 最初のバンドを読み込み
                        print(f"Successfully loaded segmentation image: {full_path}")
                        print(f"  Shape: {seg_img.shape}, Data type: {seg_img.dtype}")
                        print(f"  Value range: {seg_img.min()} - {seg_img.max()}")
                        print(f"  Unique values: {np.unique(seg_img)}")
                        return seg_img.astype(np.uint8)
                except ImportError:
                    print("rasterio not available, falling back to cv2")
                    # fallback to cv2
                    seg_img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
                    if seg_img is not None:
                        # 修正: 多チャンネルの場合は最初のチャンネルのみ使用
                        if len(seg_img.shape) == 3:
                            seg_img = seg_img[:, :, 0]
                        print(f"Successfully loaded segmentation image (cv2): {full_path}")
                        print(f"  Shape: {seg_img.shape}, Data type: {seg_img.dtype}")
                        print(f"  Value range: {seg_img.min()} - {seg_img.max()}")
                        print(f"  Unique values: {np.unique(seg_img)}")
                        return seg_img
                    else:
                        print(f"Failed to load segmentation image using cv2: {full_path}")
                        return None
            except Exception as e:
                print(f"Error loading segmentation image {full_path}: {e}")
        else:
            print(f"Segmentation image not found: {full_path}")
        return None
    
    def create_segmentation_colormap(self, seg_img):
        """Create colormap for segmentation image"""
        if seg_img is None:
            return None, None
        
        # 修正: デバッグ情報の追加と値範囲チェック
        unique_vals = np.unique(seg_img)
        print(f"Creating segmentation colormap:")
        print(f"  Unique values: {unique_vals}")
        print(f"  Value range: {seg_img.min()} - {seg_img.max()}")
        
        # 指定された色マップ
        colors = {
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
        
        from matplotlib.colors import ListedColormap
        
        # 修正: 実際に存在する値に基づいてカラーマップを調整
        max_val = int(unique_vals.max())
        if max_val > 10:
            print(f"Unexpected high value detected: {max_val}")
            # 高い値を0-10の範囲にマッピング
            seg_img_normalized = np.clip(seg_img, 0, 10)
            unique_vals = np.unique(seg_img_normalized)
            print(f"  Unique values after normalization: {unique_vals}")
        else:
            seg_img_normalized = seg_img
        
        # 必要な色数を決定（最大値+1、最低11色）
        num_colors = max(11, max_val + 1)
        color_list = []
        for i in range(num_colors):
            if i in colors:
                color_list.append(np.array(colors[i])/255.0)
            else:
                # デフォルト色（白）
                color_list.append(np.array([255, 255, 255])/255.0)
        
        cmap = ListedColormap(color_list)
        
        # 修正: BoundaryNorm用の境界を作成（離散クラスの1:1マッピング）
        boundaries = np.arange(num_colors + 1) - 0.5  # クラス境界: [-0.5, 0.5, 1.5, 2.5, ...]
        norm = BoundaryNorm(boundaries=boundaries, ncolors=num_colors)
        
        # 修正: カラーマップの設定情報を出力
        print(f"Created colormap: {num_colors} colors")
        print(f"  Values used: {unique_vals}")
        print(f"  Boundary settings: {boundaries}")
        
        return cmap, norm, unique_vals
    
    def parse_percentage_coordinates(self, coord_str, img_shape):
        """Parse percentage coordinates to pixel coordinates with decimal precision"""
        h, w = img_shape[:2]
        
        # Handle percentage point format: (37.5%, 68.2%) - 小数点第一位まで対応
        point_match = re.search(r'\((\d+\.?\d*)%,\s*(\d+\.?\d*)%\)', coord_str)
        if point_match:
            x_pct, y_pct = map(float, point_match.groups())
            x = int(x_pct * w / 100)
            y = int(y_pct * h / 100)
            # デバッグ: 座標変換を確認
            print(f"Coordinate conversion: {coord_str} -> ({x_pct:.1f}%, {y_pct:.1f}%) -> pixel({x}, {y}) in {w}x{h} image")
            return x, y
        
        # Handle percentage region format (x1%, y1%, x2%, y2%): Region (0%, 13%, 19%, 32%) - 小数点対応
        region_match = re.search(r'Region\s*\((\d+\.?\d*)%,\s*(\d+\.?\d*)%,\s*(\d+\.?\d*)%,\s*(\d+\.?\d*)%\)', coord_str)
        if region_match:
            x1_pct, y1_pct, x2_pct, y2_pct = map(float, region_match.groups())
            x1 = int(x1_pct * w / 100)
            y1 = int(y1_pct * h / 100)
            x2 = int(x2_pct * w / 100)
            y2 = int(y2_pct * h / 100)
            # Calculate width and height from [x1, y1, x2, y2]
            width = x2 - x1
            height = y2 - y1
            return x1, y1, width, height  # x, y, width, height
            
        # Handle percentage region format: [xmin=54%, ymin=84%, xmax=66%, ymax=95%] - 小数点対応
        region_match = re.findall(r'(\w+)=(\d+\.?\d*)%', coord_str)
        if region_match:
            coords = {}
            for key, value in region_match:
                coords[key] = float(value)
            
            if all(k in coords for k in ['xmin', 'ymin', 'xmax', 'ymax']):
                x1 = int(coords['xmin'] * w / 100)
                y1 = int(coords['ymin'] * h / 100)
                x2 = int(coords['xmax'] * w / 100)
                y2 = int(coords['ymax'] * h / 100)
                # Calculate width and height from [x1, y1, x2, y2]
                width = x2 - x1
                height = y2 - y1
                return x1, y1, width, height  # x, y, width, height
                
        return None
    
    def parse_coordinates(self, coord_str, img_shape=None):
        """Parse coordinate string to get x, y values"""
        if not isinstance(coord_str, str):
            return None
            
        # Handle percentage coordinates
        if '%' in coord_str and img_shape is not None:
            return self.parse_percentage_coordinates(coord_str, img_shape)
            
        # Handle pixel point format: (y, x)
        if coord_str.startswith('(') and ')' in coord_str and 'Region' not in coord_str:
            try:
                coords = coord_str.strip('()').split(',')
                if len(coords) == 2:
                    x, y = map(int, coords)
                    return x, y
            except ValueError:
                return None
                
        # Handle region format: Region at (y, x) with size heightxwidth
        elif coord_str.startswith('Region at'):
            try:
                parts = coord_str.split('(')[1].split(')')[0].split(',')
                x, y = map(int, parts)
                size_part = coord_str.split('size ')[1]
                if 'x' in size_part:
                    height, width = map(int, size_part.split('x'))
                    return x, y, width, height
            except (ValueError, IndexError):
                return None
                
        return None
    
    def extract_region_from_text(self, text, img_shape):
        """Extract region coordinates from question text with decimal precision"""
        regions = {}
        
        # Pattern 1: Region A/B/C/D/E/F: [xmin=54.5%, ymin=84.2%, xmax=66.8%, ymax=95.1%] - 拡張対応
        region_pattern_1 = r'Region ([A-Z]):\s*\[([^\]]+)\]'
        matches_1 = re.findall(region_pattern_1, text)
        
        for region_name, coord_str in matches_1:
            # Parse coordinates from the matched string - 小数点対応
            coord_matches = re.findall(r'(\w+)=(\d+\.?\d*)%', coord_str)
            if coord_matches:
                coords = {}
                for key, value in coord_matches:
                    coords[key] = float(value)
                
                # Convert to pixel coordinates
                if all(k in coords for k in ['xmin', 'ymin', 'xmax', 'ymax']):
                    h, w = img_shape[:2]
                    x1 = int(coords['xmin'] * w / 100)
                    y1 = int(coords['ymin'] * h / 100)
                    x2 = int(coords['xmax'] * w / 100)
                    y2 = int(coords['ymax'] * h / 100)
                    width = x2 - x1
                    height = y2 - y1
                    
                    # 修正: 複数の形式でキーを保存してマッピングを強化
                    region_key_full = f'Region {region_name}'
                    regions[region_key_full] = (x1, y1, width, height)
                    regions[region_name] = (x1, y1, width, height)  # 短縮形も保存
                    print(f"Debug: Extracted region {region_key_full} -> {region_name}: ({x1}, {y1}, {width}, {height})")
        
        # Pattern 2: Region (63.5%, 24.2%, 87.8%, 48.9%) - 小数点対応
        region_pattern_2 = r'Region \((\d+\.?\d*)%,\s*(\d+\.?\d*)%,\s*(\d+\.?\d*)%,\s*(\d+\.?\d*)%\)'
        matches_2 = re.findall(region_pattern_2, text)
        
        for i, (xmin, ymin, xmax, ymax) in enumerate(matches_2):
            h, w = img_shape[:2]
            x1 = int(float(xmin) * w / 100)
            y1 = int(float(ymin) * h / 100)
            x2 = int(float(xmax) * w / 100)
            y2 = int(float(ymax) * h / 100)
            width = x2 - x1
            height = y2 - y1
            regions[f'Option {i+1}'] = (x1, y1, width, height)
        
        # 追加: Pattern 3: より柔軟なRegion定義をサポート  
        # "Region A: (x%, y%, x%, y%)" や "A: [coordinates]" などの形式
        region_pattern_3 = r'(?:Region\s+)?([A-Z]):\s*(?:\[([^\]]+)\]|\(([^)]+)\))'
        matches_3 = re.findall(region_pattern_3, text)
        
        for region_name, bracket_coords, paren_coords in matches_3:
            coord_str = bracket_coords or paren_coords
            if coord_str:
                # 座標パースを試行
                coord_matches = re.findall(r'(\w+)=(\d+\.?\d*)%', coord_str)
                if coord_matches:
                    coords = {}
                    for key, value in coord_matches:
                        coords[key] = float(value)
                    
                    if all(k in coords for k in ['xmin', 'ymin', 'xmax', 'ymax']):
                        h, w = img_shape[:2]
                        x1 = int(coords['xmin'] * w / 100)
                        y1 = int(coords['ymin'] * h / 100)
                        x2 = int(coords['xmax'] * w / 100)
                        y2 = int(coords['ymax'] * h / 100)
                        width = x2 - x1
                        height = y2 - y1
                        
                        # 複数の形式でキーを保存
                        if f'Region {region_name}' not in regions:
                            regions[f'Region {region_name}'] = (x1, y1, width, height)
                            regions[region_name] = (x1, y1, width, height)
                            print(f"Debug: Extracted region (Pattern 3) {region_name}: ({x1}, {y1}, {width}, {height})")
                
        return regions
    
    def extract_points_from_text(self, text, img_shape):
        """Extract point coordinates from question text with decimal precision"""
        points = []
        
        # Pattern for percentage points in text: (37.5%, 68.2%) - 小数点対応
        point_pattern = r'\((\d+\.?\d*)%?,\s*(\d+\.?\d*)%?\)'
        matches = re.findall(point_pattern, text)
        
        for x_pct, y_pct in matches:
            h, w = img_shape[:2]
            x = int(float(x_pct) * w / 100)
            y = int(float(y_pct) * h / 100)
            points.append((x, y))
            
        return points
    
    def extract_choices_from_text(self, text):
        """Return a list of choice strings parsed from the question text.

        It looks for the substring 'Please choose from:' then collects the
        non-empty lines that follow. Numbering such as '1. ' or '2. ' is
        stripped. Parsing stops when we reach an empty line or a line that
        starts with explanatory phrases like 'Note:' or 'For short answer'.
        """
        if not isinstance(text, str):
            return []

        lower_text = text.lower()
        marker = 'please choose from:'
        idx = lower_text.find(marker)
        if idx == -1:
            return []

        choices_block = text[idx + len(marker):].splitlines()
        choices = []
        for line in choices_block:
            stripped = line.strip()
            if not stripped:
                # blank line terminates the choices list
                break
            # stop at explanatory lines
            if stripped.lower().startswith(('note:', 'for short answer', 'hint:', 'grid explanation')):
                break
            # remove leading numbering e.g. "1. " or "2) "
            stripped = re.sub(r'^\s*\d+[\.)]\s*', '', stripped)
            choices.append(stripped)
        return choices
    
    def add_overlays_to_axis(self, ax, qa_result, img_shape, overlay_type='all'):
        """Add overlays (points, regions, etc.) to an axis"""
        category = qa_result['category']
        
        # -------- Ensure choices list is available by parsing text if needed --------
        if ('choices' not in qa_result or not qa_result.get('choices')) and 'text' in qa_result:
            parsed_choices = self.extract_choices_from_text(qa_result['text'])
            if parsed_choices:
                qa_result['choices'] = parsed_choices
        
        if category in self.point_categories:
            # 修正: sun_exposure カテゴリの場合、デバッグ情報からSVF値を抽出
            choice_svf_values = {}
            if 'debug_info' in qa_result:
                for debug_line in qa_result['debug_info']:
                    # "Point 1 (35%, 42%): SVF = 0.850" のような形式をパース
                    import re
                    match = re.search(r'Point (\d+) \(([^)]+)\): SVF = ([\d.]+)', debug_line)
                    if match:
                        point_num = int(match.group(1))
                        coord_str = f"({match.group(2)})"
                        svf_val = float(match.group(3))
                        choice_svf_values[coord_str] = svf_val
            
            # Plot all choices (choicesフィールドが存在する場合のみ)
            if 'choices' in qa_result:
                for i, choice in enumerate(qa_result['choices']):
                    coords = self.parse_coordinates(choice, img_shape)
                    if coords and len(coords) == 2:
                        x, y = coords
                        # 修正: 選択肢のマーカーサイズを調整（中間サイズ）
                        ax.plot(x, y, 'o', color=self.colors['choice'], alpha=0.7, markersize=12)
                        
                        # 追加: SVF値を数値で表示
                        svf_val = choice_svf_values.get(choice, None)
                        if svf_val is not None:
                            ax.text(x + 5, y - 10, f"SVF={svf_val:.3f}", 
                                  color='white', fontsize=8, fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.2', facecolor='blue', alpha=0.7))
                    
            # Plot correct answer
            coords = self.parse_coordinates(qa_result['answer'], img_shape)
            if coords and len(coords) == 2:
                x, y = coords
                # 修正: 正解マーカーサイズを調整（中間サイズ）
                ax.plot(x, y, 'D', color=self.colors['correct'], markersize=10, 
                        markeredgewidth=2, markeredgecolor='white', zorder=10)
                ax.annotate('Correct', (x, y), xytext=(12, -12), textcoords='offset points', 
                            color='white', fontsize=12, fontweight='bold', 
                            bbox=dict(boxstyle='round,pad=0.3', facecolor=self.colors['correct'], alpha=0.8), zorder=11)
                
                # 追加: 正解のSVF値を表示
                correct_svf = choice_svf_values.get(qa_result['answer'], None)
                if correct_svf is not None:
                    ax.text(x + 5, y + 15, f"SVF={correct_svf:.3f}", 
                          color='white', fontsize=10, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.8))
            
            # Also check for points in text (for comparison questions)
            points = self.extract_points_from_text(qa_result['text'], img_shape)
            for i, (x, y) in enumerate(points):
                ax.plot(x, y, 's', color='orange', markersize=6, alpha=0.8)
                ax.annotate(f'P{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', 
                           color='orange', fontsize=8, fontweight='bold')
                           
        elif category in self.region_categories:
            # Extract and plot percentage-based regions from text (質問文内のRegionパターン)
            regions = self.extract_region_from_text(qa_result['text'], img_shape)
            print(f"Debug: Extracted regions from text: {regions}")  # デバッグ情報を追加
            
            # Plot all choices - choices配列からもregion bboxを抽出
            if 'choices' in qa_result:
                print(f"Debug: Processing choices: {qa_result['choices']}")
                correct_answer = qa_result.get('answer', '')
                
                for i, choice in enumerate(qa_result['choices']):
                    # 正解かどうかを判定
                    is_correct = (choice == correct_answer)
                    
                    # 色とスタイルを設定 - 正解と不正解のコントラストを最大化
                    if is_correct:
                        edge_color = '#00FF00'  # 正解は鮮やかな緑
                        linewidth = 6  # より太く
                        linestyle = '-'
                        label = '✓ CORRECT'
                        label_color = 'white'
                        label_bg = '#00AA00'  # 濃い緑
                        alpha = 1.0
                        fill_alpha = 0.2  # 薄い緑の塗りつぶし
                    else:
                        edge_color = '#FF0000'  # 不正解はすべて同じ鮮やかな赤
                        linewidth = 3
                        linestyle = '--'
                        label = f'✗ Wrong'  # シンプルに
                        label_color = 'white'
                        label_bg = '#CC0000'  # 濃い赤
                        alpha = 0.8
                        fill_alpha = 0.1  # 薄い赤の塗りつぶし
                    
                    # 修正: 選択肢が単純なテキスト（例：Region A）の場合、質問文から対応する座標を検索
                    coords = None
                    
                    # まず通常の座標解析を試す
                    coords = self.parse_coordinates(choice, img_shape)
                    
                    # 座標が見つからない場合、質問文から対応する地域を検索
                    if not coords:
                        # 修正: より柔軟なマッピングロジック
                        potential_keys = []
                        
                        # パターン1: "Region A" -> ["Region A", "A"]
                        if "Region" in choice:
                            region_letter = choice.split()[-1]
                            potential_keys.extend([choice, region_letter, f"Region {region_letter}"])
                        # パターン2: "A" -> ["A", "Region A"]  
                        elif len(choice) == 1 and choice.isalpha():
                            potential_keys.extend([choice, f"Region {choice}"])
                        # パターン3: その他
                        else:
                            potential_keys.append(choice)
                        
                        # すべての可能なキーを試す
                        for key in potential_keys:
                            if key in regions:
                                coords = regions[key]
                                print(f"Debug: Found coordinates for {choice} -> {key}: {coords}")
                                break
                        
                        # まだ見つからない場合、大文字小文字を無視して検索
                        if not coords:
                            for region_key, region_coords in regions.items():
                                if any(k.lower() == region_key.lower() for k in potential_keys):
                                    coords = region_coords
                                    print(f"Debug: Found coordinates for {choice} (case-insensitive) -> {region_key}: {coords}")
                                    break
                    
                    if coords and len(coords) == 4:
                        x, y, w, h = coords
                        # 塗りつぶし付きの矩形で視認性を最大化
                        rect = Rectangle((x, y), w, h, 
                                       fill=True,  # 塗りつぶしを有効
                                       facecolor=edge_color,
                                       alpha=fill_alpha,
                                       edgecolor=edge_color,
                                       linewidth=linewidth,
                                       linestyle=linestyle)
                        ax.add_patch(rect)
                        
                        # ラベルのフォントサイズを大きくして目立たせる
                        ax.text(x + w//2, y + h//2, label, 
                               ha='center', va='center', 
                               color='white', 
                               fontsize=14, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.4', facecolor=label_bg, alpha=0.95, edgecolor='white', linewidth=2))
                        print(f"Debug: Drew {'CORRECT' if is_correct else 'WRONG'} region {choice} at ({x}, {y}, {w}, {h}) with {edge_color}")
                    else:
                        # Region (x%, y%, z%, w%)形式の場合の処理
                        region_match = re.match(r'Region \((\d+)%,\s*(\d+)%,\s*(\d+)%,\s*(\d+)%\)', choice)
                        if region_match:
                            xmin, ymin, xmax, ymax = map(int, region_match.groups())
                            h, w = img_shape[:2]
                            x1 = int(xmin * w / 100)
                            y1 = int(ymin * h / 100)
                            x2 = int(xmax * w / 100)
                            y2 = int(ymax * h / 100)
                            width = x2 - x1
                            height = y2 - y1
                            
                            # 塗りつぶし付きの矩形で視認性を最大化
                            rect = Rectangle((x1, y1), width, height, 
                                           fill=True,  # 塗りつぶしを有効
                                           facecolor=edge_color,
                                           alpha=fill_alpha,
                                           edgecolor=edge_color,
                                           linewidth=linewidth,
                                           linestyle=linestyle)
                            ax.add_patch(rect)
                            
                            # ラベルのフォントサイズを大きくして目立たせる
                            ax.text(x1 + width//2, y1 + height//2, label, 
                                   ha='center', va='center', 
                                   color='white', 
                                   fontsize=14, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.4', facecolor=label_bg, alpha=0.95, edgecolor='white', linewidth=2))
                            print(f"Debug: Drew {'CORRECT' if is_correct else 'WRONG'} bbox at ({x1}, {y1}, {width}, {height}) with {edge_color}")
                        else:
                            print(f"Debug: Could not find coordinates for choice: {choice}")
                    
            # 注意: 正解の描画は上のchoices処理で既に実行済み（重複を避けるため、ここでは何もしない）
            # choicesにない場合のみ、別途正解を描画する
            if 'choices' not in qa_result or qa_result['answer'] not in qa_result.get('choices', []):
                coords = self.parse_coordinates(qa_result['answer'], img_shape)
                if coords and len(coords) == 4:
                    x, y, w, h = coords
                    rect = Rectangle((x, y), w, h,
                                    fill=True,
                                    facecolor='#00FF00',
                                    alpha=0.2,
                                    edgecolor='#00FF00',
                                    linewidth=6)
                    ax.add_patch(rect)
                    ax.text(x + w//2, y + h//2, '✓ CORRECT', 
                           ha='center', va='center', 
                           color='white', 
                           fontsize=14, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='#00AA00', alpha=0.95, edgecolor='white', linewidth=2))
                else:
                    # 正解がRegion (x%, y%, z%, w%)形式の場合
                    answer_match = re.match(r'Region \((\d+)%,\s*(\d+)%,\s*(\d+)%,\s*(\d+)%\)', qa_result['answer'])
                    if answer_match:
                        xmin, ymin, xmax, ymax = map(int, answer_match.groups())
                        h, w = img_shape[:2]
                        x1 = int(xmin * w / 100)
                        y1 = int(ymin * h / 100)
                        x2 = int(xmax * w / 100)
                        y2 = int(ymax * h / 100)
                        width = x2 - x1
                        height = y2 - y1
                        
                        rect = Rectangle((x1, y1), width, height,
                                        fill=True,
                                        facecolor='#00FF00',
                                        alpha=0.2,
                                        edgecolor='#00FF00',
                                        linewidth=6)
                        ax.add_patch(rect)
                        ax.text(x1 + width//2, y1 + height//2, '✓ CORRECT', 
                               ha='center', va='center', 
                               color='white', 
                               fontsize=14, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='#00AA00', alpha=0.95, edgecolor='white', linewidth=2))
            
            # 質問文内のRegionパターンも描画（従来通り）
            colors = ['red', 'blue', 'green', 'purple', 'orange']  # 色を追加
            for i, (region_name, coords) in enumerate(regions.items()):
                if len(coords) == 4:
                    x, y, w, h = coords
                    color = colors[i % len(colors)]
                    print(f"Debug: Drawing {region_name} at ({x}, {y}, {w}, {h}) with color {color}")  # デバッグ情報
                    rect = Rectangle((x, y), w, h,
                                    fill=False,
                                    edgecolor=color,
                                    linewidth=3,  # 線を太くして見やすくする
                                    linestyle='--')
                    ax.add_patch(rect)
                    ax.text(x + w//2, y + h//2, region_name, 
                           ha='center', va='center', 
                           color=color, 
                           fontsize=12, fontweight='bold',  # フォントサイズを少し大きくする
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                           
        elif category in self.grid_categories:
            # Draw 3x3 grid
            h, w = img_shape[:2]
            grid_h, grid_w = h // 3, w // 3
            
            for i in range(3):
                for j in range(3):
                    rect = Rectangle((j * grid_w, i * grid_h),
                                   grid_w, grid_h,
                                   fill=False,
                                   edgecolor=self.colors['grid'],
                                   alpha=0.5,
                                   linewidth=1)
                    ax.add_patch(rect)
                    
                    # Add grid labels
                    # Use "middle" instead of "center" for row names to align with canonical naming
                    cell_names = [
                        ['top left', 'top center', 'top right'],
                        ['middle left', 'middle center', 'middle right'],
                        ['bottom left', 'bottom center', 'bottom right']
                    ]
                    ax.text(j * grid_w + grid_w//2, i * grid_h + grid_h//2, 
                           cell_names[i][j], 
                           ha='center', va='center', 
                           color='yellow', fontsize=6, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
                    
            # Highlight correct answer
            correct_cell = qa_result['answer']
            cell_map = {
                'top left': (0, 0), 'top center': (0, 1), 'top right': (0, 2),
                'middle left': (1, 0), 'middle center': (1, 1), 'middle right': (1, 2),
                'center left': (1, 0), 'center': (1, 1), 'center right': (1, 2),
                'bottom left': (2, 0), 'bottom center': (2, 1), 'bottom right': (2, 2)
            }
            
            if correct_cell in cell_map:
                i, j = cell_map[correct_cell]
                rect = Rectangle((j * grid_w, i * grid_h),
                               grid_w, grid_h,
                               fill=False,
                               edgecolor=self.colors['correct'],
                               linewidth=4)
                ax.add_patch(rect)
                
        elif category in self.comparison_categories:
            # Extract points from question text
            points = self.extract_points_from_text(qa_result['text'], img_shape)
            for i, (x, y) in enumerate(points):
                ax.plot(x, y, 'o', color=self.colors['comparison'], markersize=10, alpha=0.8)
                ax.annotate(f'Point {i+1}', (x, y), xytext=(10, 10), textcoords='offset points', 
                           color='white', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=self.colors['comparison'], alpha=0.8))
    
    def visualize_qa_result_multimodal(self, qa_result, save_path=None):
        """Visualize QA result with multiple modalities"""
        try:
            # Load all modalities with error handling
            try:
                rgb_img = self.load_image(qa_result['image'])
            except Exception as e:
                print(f"Error loading RGB image {qa_result['image']}: {e}")
                rgb_img = np.ones((512, 512, 3), dtype=np.uint8) * 128  # Gray placeholder
            
            svf_img = self.load_svf_image(qa_result['image'])
            seg_img = self.load_seg_image(qa_result['image'])
            
            # Count available modalities
            available_modalities = 1  # RGB is always available
            if svf_img is not None:
                available_modalities += 1
            if seg_img is not None:
                available_modalities += 1
            
            # Create subplot layout
            if available_modalities == 1:
                fig, axes = plt.subplots(1, 1, figsize=(12, 10))
                axes = [axes]
            elif available_modalities == 2:
                fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            else:
                fig, axes = plt.subplots(1, 3, figsize=(30, 10))
            
            # Plot RGB with overlays  
            # 修正: interpolation='nearest'でピクセル境界を明確化
            axes[0].imshow(rgb_img, interpolation='nearest')
            self.add_overlays_to_axis(axes[0], qa_result, rgb_img.shape)
            # Keep axis title concise like 4-modality views
            axes[0].set_title('RGB Image', fontsize=12)
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            
            current_axis = 1
            
            # Plot SVF if available
            if svf_img is not None and current_axis < len(axes):
                # Normalize SVF values for display (robust)
                svf_array = np.array(svf_img)
                if svf_array.ndim == 3 and svf_array.shape[2] >= 3:
                    svf_array = svf_array.mean(axis=2)
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
                # 修正: interpolation='nearest'を追加してピクセル境界を明確化
                im = axes[current_axis].imshow(svf_display, 
                                             cmap=self.svf_cmap, 
                                             vmin=0, vmax=1, 
                                             interpolation='nearest')
                self.add_overlays_to_axis(axes[current_axis], qa_result, svf_img.shape)
                axes[current_axis].set_title('SVF (Sky View Factor)', fontsize=12)
                axes[current_axis].set_xticks([])
                axes[current_axis].set_yticks([])
                
                # Add colorbar for SVF
                plt.colorbar(im, ax=axes[current_axis], fraction=0.046, pad=0.04, label='SVF Value')
                current_axis += 1
            
            # Plot Segmentation if available
            if seg_img is not None and current_axis < len(axes):
                # Create segmentation colormap
                seg_cmap, seg_norm, unique_vals = self.create_segmentation_colormap(seg_img)
                if seg_cmap is not None:
                    print(f"Segmentation visualization:")
                    print(f"  Using BoundaryNorm: discrete class to color 1:1 mapping")
                    print(f"  Image shape: {seg_img.shape}")
                    print(f"  Data type: {seg_img.dtype}")
                    
                    im = axes[current_axis].imshow(seg_img, cmap=seg_cmap, vmin=vmin, vmax=vmax)
                    self.add_overlays_to_axis(axes[current_axis], qa_result, seg_img.shape)
                    axes[current_axis].set_title('Segmentation', fontsize=12)
                    axes[current_axis].set_xticks([])
                    axes[current_axis].set_yticks([])
                    
                    # Add colorbar for segmentation with improved styling
                    cbar = plt.colorbar(im, ax=axes[current_axis], fraction=0.046, pad=0.04)
                    cbar.set_label('Land Cover Class', fontsize=10, fontweight='bold')
                    
                    # Show only appearing classes in the colorbar legend
                    if len(unique_vals) <= 11:
                        cbar.set_ticks(unique_vals)
                        # Class name mapping with improved readability
                        class_names = {
                            0: 'Background', 1: 'Forest', 2: 'Water', 3: 'Agricultural',
                            4: 'Urban', 5: 'Grassland', 6: 'Railway', 7: 'Highway',
                            8: 'Airport', 9: 'Roads', 10: 'Buildings'
                        }
                        labels = [class_names.get(int(val), f'Class {int(val)}') for val in unique_vals]
                        
                        # Improved font size calculation for better readability
                        num_classes = len(unique_vals)
                        if num_classes <= 4:
                            fontsize = 11
                        elif num_classes <= 7:
                            fontsize = 10
                        else:
                            fontsize = 9
                        
                        fontsize = max(8, fontsize)  # Minimum readable size
                        cbar.set_ticklabels(labels, fontsize=fontsize, fontweight='bold')
                else:
                    print("Failed to create segmentation colormap")
            else:
                if seg_img is None:
                    print("Segmentation image not loaded")
                if current_axis >= len(axes):
                    print("Insufficient axes for segmentation display")
            
            # Add overall title (shortened if too long)
            qid_text = self._shorten(str(qa_result.get('question_id', 'unknown')), 30)
            cat_text = self._shorten(str(qa_result.get('category', 'unknown')), 40)
            fig.suptitle(f'Q{qid_text} - {cat_text}', fontsize=16, y=0.95)
            
            # Add legend - 新しい視覚的区別を反映（色を統一）
            legend_elements = [
                plt.Line2D([0], [0], color='#00FF00', linewidth=6, label='✓ CORRECT ANSWER'),
                plt.Line2D([0], [0], color='#FF0000', linewidth=3, linestyle='--', label='✗ WRONG CHOICES'),
                plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=self.colors['comparison'], markersize=6, label='Comparison Points'),
                plt.Line2D([0], [0], marker='D', color='w', 
                           markerfacecolor=self.colors['correct'], markeredgecolor=self.colors['correct'], markersize=10, label='Point Answers')
            ]
            fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.9))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            print(f"Error visualizing question {qa_result.get('question_id', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()
            if save_path:
                plt.close()
    
    def visualize_qa_result(self, qa_result, save_path=None):
        """Backward compatibility - now uses multimodal visualization"""
        self.visualize_qa_result_multimodal(qa_result, save_path)
    
    def visualize_all_results(self, output_dir):
        """Visualize all QA results and save to output directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        for qa_result in self.qa_results:
            try:
                save_path = os.path.join(output_dir, 
                                       f"qa_{qa_result['question_id']:03d}_{qa_result['category']}.png")
                self.visualize_qa_result_multimodal(qa_result, save_path)
            except Exception as e:
                print(f"Failed to process question {qa_result.get('question_id', 'unknown')}: {e}")
                continue
                
    def create_summary_report(self, output_dir):
        """Create a summary report of all categories"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Group by category
        categories = {}
        for qa_result in self.qa_results:
            cat = qa_result['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(qa_result)
        
        # Create summary
        summary_path = os.path.join(output_dir, "summary_report.txt")
        with open(summary_path, 'w') as f:
            f.write("QA Results Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            for category, results in categories.items():
                f.write(f"Category: {category}\n")
                f.write(f"Number of questions: {len(results)}\n")
                f.write(f"Question IDs: {[r['question_id'] for r in results]}\n")
                
                # Check which modalities are used
                modalities = {'RGB': True}  # RGB is always available
                for result in results[:3]:  # Check first few
                    if self.load_svf_image(result['image']) is not None:
                        modalities['SVF'] = True
                    if self.load_seg_image(result['image']) is not None:
                        modalities['Segmentation'] = True
                        
                f.write(f"Used modalities: {', '.join(modalities.keys())}\n")
                f.write("-" * 30 + "\n")
        
        print(f"Summary report saved: {summary_path}")
            
def main():
    # Example usage
    qa_json_path = "svf_qa_output_revised0606/svf_10x_detailed_train_0608.json"
    image_dir = "../SynRS3D/GeoNRW_dsm"  # Update this path
    svf_dir = "../SynRS3D/GeoNRW_dsm/svf/skyview_umep_test"
    output_dir = "visualization_results_0608"
    
    visualizer = QAVisualizer(qa_json_path, image_dir, svf_dir)
    visualizer.visualize_all_results(output_dir)
    visualizer.create_summary_report(output_dir)
    
if __name__ == "__main__":
    main() 