"""Image converters for modality data (DSM/SVF/SEG) to RGB PIL.Image.

This module provides small, focused helpers:
- dsm_to_rgb: Convert DSM GeoTIFF to RGB with a matplotlib colormap.
- svf_to_rgb: Convert SVF GeoTIFF (0-1) to RGB with a colormap.
- seg_to_rgb: Convert segmentation GeoTIFF class ids to fixed RGB colors.

All functions return a PIL.Image.Image or None on failure.
"""

import os
from typing import Optional, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Optional heavy deps; guarded import to avoid hard failure at import time
try:
    import rasterio
    from rasterio.plot import reshape_as_image
except Exception:  # pragma: no cover
    rasterio = None  # type: ignore
    reshape_as_image = None  # type: ignore

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore


def _normalize_array(array: np.ndarray, min_val: Optional[float] = None, max_val: Optional[float] = None) -> np.ndarray:
    if min_val is None:
        min_val = float(np.nanmin(array))
    if max_val is None:
        max_val = float(np.nanmax(array))
    array = np.nan_to_num(array, nan=min_val)
    if min_val == max_val:
        return np.zeros_like(array)
    normalized = (array - min_val) / (max_val - min_val)
    return np.clip(normalized, 0.0, 1.0)


def dsm_to_rgb(
    dsm_path: str,
    colormap: str = "terrain",
    show_colorbar: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Optional[Image.Image]:
    if rasterio is None or plt is None:
        print("Warning: rasterio/matplotlib not available; cannot convert DSM to RGB.")
        return None
    try:
        with rasterio.open(dsm_path) as src:
            data = src.read(1)

        # Determine normalization bounds
        min_val = float(np.nanmin(data)) if vmin is None else float(vmin)
        max_val = float(np.nanmax(data)) if vmax is None else float(vmax)
        normalized = _normalize_array(data, min_val=min_val, max_val=max_val)

        # Apply colormap
        cm = plt.get_cmap(colormap)
        colored = cm(normalized)
        rgb = (colored[:, :, :3] * 255).astype(np.uint8)
        base_img = Image.fromarray(rgb)

        # Optionally append a vertical colorbar with min/max labels
        if not show_colorbar:
            return base_img

        height = base_img.height
        bar_width = 18
        label_width = 64
        margin = 6

        # Create vertical gradient (top=max -> bottom=min)
        gradient = np.linspace(1.0, 0.0, num=height, dtype=np.float32).reshape(height, 1)
        gradient = np.repeat(gradient, bar_width, axis=1)
        bar_rgba = cm(gradient)
        bar_rgb = (bar_rgba[:, :, :3] * 255).astype(np.uint8)
        bar_img = Image.fromarray(bar_rgb)

        # Label canvas
        label_img = Image.new("RGB", (label_width, height), (255, 255, 255))
        draw = ImageDraw.Draw(label_img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None  # Fallback to default

        # Draw labels: max at top, min at bottom
        top_text = f"{max_val:.2f} m"
        bot_text = f"{min_val:.2f} m"
        text_color = (0, 0, 0)

        # Measure text size if possible
        def _text_size(s: str) -> tuple:
            if font is None:
                return (len(s) * 6, 11)
            return draw.textlength(s, font=font), font.size if hasattr(font, "size") else 11

        top_w, top_h = _text_size(top_text)
        bot_w, bot_h = _text_size(bot_text)

        draw.text((max(0, (label_width - top_w) // 2), max(0, margin)), top_text, fill=text_color, font=font)
        draw.text((max(0, (label_width - bot_w) // 2), max(0, height - bot_h - margin)), bot_text, fill=text_color, font=font)

        # Title
        title = "DSM (m)"
        title_w, title_h = _text_size(title)
        draw.text((max(0, (label_width - title_w) // 2), max(0, height // 2 - title_h // 2)), title, fill=text_color, font=font)

        # Compose final image: [base_img | margin | bar_img | margin | label_img]
        total_width = base_img.width + margin + bar_img.width + margin + label_img.width
        combined = Image.new("RGB", (total_width, height), (255, 255, 255))
        x = 0
        combined.paste(base_img, (x, 0)); x += base_img.width + margin
        combined.paste(bar_img, (x, 0)); x += bar_img.width + margin
        combined.paste(label_img, (x, 0))
        return combined
    except Exception as e:  # pragma: no cover
        print(f"DSM conversion error: {e}, path: {dsm_path}")
        return None


def svf_to_rgb(
    svf_path: str,
    colormap: str = "plasma",
    show_colorbar: bool = False,
    vmin: Optional[float] = 0.0,
    vmax: Optional[float] = 1.0,
) -> Optional[Image.Image]:
    if rasterio is None or plt is None:
        print("Warning: rasterio/matplotlib not available; cannot convert SVF to RGB.")
        return None
    try:
        with rasterio.open(svf_path) as src:
            data = src.read(1)
        min_val = float(0.0 if vmin is None else vmin)
        max_val = float(1.0 if vmax is None else vmax)
        normalized = _normalize_array(data, min_val=min_val, max_val=max_val)
        cm = plt.get_cmap(colormap)
        colored = cm(normalized)
        rgb = (colored[:, :, :3] * 255).astype(np.uint8)
        base_img = Image.fromarray(rgb)

        if not show_colorbar:
            return base_img

        height = base_img.height
        bar_width = 18
        label_width = 64
        margin = 6

        # Create vertical gradient (top=max -> bottom=min)
        gradient = np.linspace(1.0, 0.0, num=height, dtype=np.float32).reshape(height, 1)
        gradient = np.repeat(gradient, bar_width, axis=1)
        bar_rgba = cm(gradient)
        bar_rgb = (bar_rgba[:, :, :3] * 255).astype(np.uint8)
        bar_img = Image.fromarray(bar_rgb)

        # Label canvas
        label_img = Image.new("RGB", (label_width, height), (255, 255, 255))
        draw = ImageDraw.Draw(label_img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        top_text = f"{max_val:.2f}"
        bot_text = f"{min_val:.2f}"
        text_color = (0, 0, 0)

        def _text_size(s: str) -> tuple:
            if font is None:
                return (len(s) * 6, 11)
            return draw.textlength(s, font=font), font.size if hasattr(font, "size") else 11

        top_w, top_h = _text_size(top_text)
        bot_w, bot_h = _text_size(bot_text)

        draw.text((max(0, (label_width - top_w) // 2), max(0, margin)), top_text, fill=text_color, font=font)
        draw.text((max(0, (label_width - bot_w) // 2), max(0, height - bot_h - margin)), bot_text, fill=text_color, font=font)

        title = "SVF"
        title_w, title_h = _text_size(title)
        draw.text((max(0, (label_width - title_w) // 2), max(0, height // 2 - title_h // 2)), title, fill=text_color, font=font)

        total_width = base_img.width + margin + bar_img.width + margin + label_img.width
        combined = Image.new("RGB", (total_width, height), (255, 255, 255))
        x = 0
        combined.paste(base_img, (x, 0)); x += base_img.width + margin
        combined.paste(bar_img, (x, 0)); x += bar_img.width + margin
        combined.paste(label_img, (x, 0))
        return combined
    except Exception as e:  # pragma: no cover
        print(f"SVF conversion error: {e}, path: {svf_path}")
        return None


def seg_to_rgb(seg_path: str) -> Optional[Image.Image]:
    if rasterio is None:
        print("Warning: rasterio not available; cannot convert SEG to RGB.")
        return None
    try:
        with rasterio.open(seg_path) as src:
            seg_data = src.read()
        if seg_data.shape[0] == 1:
            seg_ids = seg_data[0]
            height, width = seg_ids.shape
            colored = np.zeros((height, width, 3), dtype=np.uint8)
            # GeoNRW class colors
            colors: Dict[int, list] = {
                0: [0, 0, 0],
                1: [0, 100, 0],
                2: [0, 0, 255],
                3: [255, 255, 0],
                4: [128, 128, 128],
                5: [144, 238, 144],
                6: [165, 42, 42],
                7: [192, 192, 192],
                8: [255, 165, 0],
                9: [128, 128, 0],
                10: [255, 0, 0],
            }
            for cls_id, rgb in colors.items():
                mask = seg_ids == cls_id
                colored[mask] = rgb
            return Image.fromarray(colored)
        # Multi-band imagery fallback
        if reshape_as_image is not None:
            img = reshape_as_image(seg_data)
            if img.shape[2] > 3:
                img = img[:, :, :3]
            return Image.fromarray(img.astype(np.uint8))
        return None
    except Exception as e:  # pragma: no cover
        print(f"SEG conversion error: {e}, path: {seg_path}")
        return None


