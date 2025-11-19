"""Multimodal enricher for factory dataset items.

Roles:
- enrich_item_modalities: From single RGB item to multi-modality item (RGB/DSM/SVF/SEG)
- add_modality_description_to_user_message: prepend human guide to question content

This module reuses modality path resolution and description utilities to keep behavior
consistent with inference (`modalities_inference_api.py`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import os

try:
    # Package-relative import (when used inside package)
    from .modality_utils.path_resolver import get_related_image_paths
    from .modality_utils.messaging import generate_modality_description
except ImportError:
    # Fallback for script execution (when run from configs as top-level)
    from modality_utils.path_resolver import get_related_image_paths
    from modality_utils.messaging import generate_modality_description


def _extract_single_rgb_path(item: Dict[str, Any]) -> Optional[str]:
    images: List[str] = item.get("images") or []
    if not images:
        return None
    return images[0]


def enrich_item_modalities(
    item: Dict[str, Any],
    dsm_dir: Optional[str],
    svf_dir: Optional[str],
    seg_dir: Optional[str],
    desired_modalities: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Augment item with additional modalities if available.

    - Resolve related DSM/SVF/SEG using `get_related_image_paths`.
    - Update `images` list to include existing paths in the requested order.
    - Embed a `modalities` field for downstream clarity (optional, non-breaking).
    """
    rgb_path = _extract_single_rgb_path(item)
    if not rgb_path:
        return item

    modalities = desired_modalities[:] if desired_modalities else ["rgb", "svf", "dsm", "seg"]
    paths = get_related_image_paths(rgb_path, dsm_dir=dsm_dir, svf_dir=svf_dir, seg_dir=seg_dir)

    ordered_paths: List[str] = []
    for m in modalities:
        p = paths.get(m)
        if p and os.path.exists(p):
            ordered_paths.append(p)
    if paths.get("rgb") and paths["rgb"] not in ordered_paths:
        ordered_paths.insert(0, paths["rgb"])

    if ordered_paths:
        item = dict(item)
        item["images"] = ordered_paths
        item["modalities"] = _modalities_from_images_order(ordered_paths)
    return item


def add_modality_description_to_user_message(
    item: Dict[str, Any],
    modalities: List[str],
    dsm_colormap: str = "terrain",
    svf_colormap: str = "plasma",
) -> Dict[str, Any]:
    images: List[str] = item.get("images") or []
    ordered_modalities = _modalities_from_images_order(images) or []
    if modalities:
        for m in modalities:
            if m and m not in ordered_modalities:
                ordered_modalities.append(m)
    guide = generate_modality_description(ordered_modalities, dsm_colormap=dsm_colormap, svf_colormap=svf_colormap)

    # Append per-image DSM stats and color hint aligned with selected colormap
    try:
        if guide and "dsm" in ordered_modalities:
            dsm_max_val = item.get("dsm_max")
            dsm_min_val = item.get("dsm_min")
            if dsm_max_val is None or dsm_min_val is None:
                meta = item.get("meta") or {}
                dsm_max_val = dsm_max_val if dsm_max_val is not None else meta.get("dsm_max")
                dsm_min_val = dsm_min_val if dsm_min_val is not None else meta.get("dsm_min")
            try:
                # Color hint consistent with generate_modality_description
                cmap = (dsm_colormap or "").lower()
                if cmap == "terrain":
                    color_hint = "blue/green = lower; yellow/brown/white = higher"
                elif cmap == "jet":
                    color_hint = "blue = lower; red = higher"
                elif cmap == "viridis":
                    color_hint = "dark blue/purple = lower; yellow = higher"
                elif cmap == "plasma":
                    color_hint = "dark purple = lower; yellow = higher"
                elif cmap == "magma":
                    color_hint = "black/purple = lower; yellow/white = higher"
                elif cmap == "inferno":
                    color_hint = "black/purple = lower; yellow = higher"
                elif cmap == "turbo":
                    color_hint = "dark blue = lower; bright yellow = higher"
                else:
                    color_hint = "colorbar indicates low→high elevation"
                stats_parts = []
                if dsm_max_val is not None:
                    stats_parts.append(f"max={float(dsm_max_val):.2f}m")
                if dsm_min_val is not None:
                    stats_parts.append(f"min={float(dsm_min_val):.2f}m")
                stats_text = f" ({', '.join(stats_parts)} in this image)" if stats_parts else ""
                guide = guide + f"\n• DSM reading: {color_hint}.{stats_text}"
            except Exception:
                pass
    except Exception:
        pass
    if not guide:
        return item
    messages: List[Dict[str, Any]] = item.get("messages") or []
    if not messages:
        return item
    # Find first user message
    for idx, msg in enumerate(messages):
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            new_msg = dict(msg)
            original: str = new_msg["content"]
            # Strip all leading <image> tokens to avoid double counting
            stripped = _strip_leading_image_tokens(original)
            # Build exact number of <image> tokens matching images count
            images: List[str] = item.get("images") or []
            prefix = "".join(["<image>\n" for _ in range(len(images))])
            if guide and guide not in stripped:
                new_msg["content"] = f"{prefix}{guide}\n\n{stripped}"
            else:
                new_msg["content"] = f"{prefix}{stripped}"
            messages[idx] = new_msg
            break
    return item


def _infer_modality_from_path(path: str) -> Optional[str]:
    """Infer modality from file name pattern.

    Expected suffixes:
    - *_rgb.jp2 -> rgb
    - *_dem_svf_umep.tif -> svf
    - *_dem.tif -> dsm
    - *_seg.tif -> seg
    """
    p = (path or "").lower()
    name = os.path.basename(p)
    if name.endswith("_dem_svf_umep.tif"):
        return "svf"
    if name.endswith("_dem.tif"):
        return "dsm"
    if name.endswith("_seg.tif"):
        return "seg"
    if name.endswith("_rgb.jp2") or name.endswith(".jp2"):
        return "rgb"
    return None


def _modalities_from_images_order(images: List[str]) -> List[str]:
    """Return unique modalities in the same order as images list."""
    seen: set = set()
    ordered: List[str] = []
    for img in images:
        m = _infer_modality_from_path(img)
        if not m or m in seen:
            continue
        seen.add(m)
        ordered.append(m)
    return ordered


def _strip_leading_image_tokens(content: str) -> str:
    """Remove all leading occurrences of "<image>" tokens (with optional newline)."""
    s = content.lstrip()
    while s.startswith("<image>"):
        # Remove leading token and following single newline if present
        s = s[len("<image>") :]
        if s.startswith("\n"):
            s = s[1:]
        s = s.lstrip()
    return s


