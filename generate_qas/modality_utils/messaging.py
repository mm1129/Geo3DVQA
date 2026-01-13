"""Message builders and modality descriptions for Qwen VL inference.

This module provides:
- generate_modality_description: human-friendly guide for included modalities
- prepare_multimodal_message: build messages with multiple PIL images
"""

import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from PIL import Image

from .image_converters import dsm_to_rgb, svf_to_rgb, seg_to_rgb


def get_geonrw_class_names() -> Dict[int, str]:
    return {
        0: "Background", 1: "Forest", 2: "Water", 3: "Agricultural",
        4: "Residential/Commercial/Industrial", 5: "Grassland/Swamp/Shrubbery",
        6: "Railway/Train station", 7: "Highway/Squares", 8: "Airport/Shipyard",
        9: "Roads", 10: "Buildings"
    }


def get_geonrw_class_colors() -> Dict[int, list]:
    return {
        0: [0, 0, 0], 1: [0, 100, 0], 2: [0, 0, 255], 3: [255, 255, 0],
        4: [128, 128, 128], 5: [144, 238, 144], 6: [165, 42, 42],
        7: [192, 192, 192], 8: [255, 165, 0], 9: [128, 128, 0], 10: [255, 0, 0]
    }


def generate_modality_description(modalities: List[str], dsm_colormap: str = "terrain", svf_colormap: str = "plasma") -> str:
    descriptions: List[str] = []
    if modalities:
        order_note = ", ".join(modalities)
        descriptions.append(
            f"Images are shown in this order: {order_note}. Non-RGB images are visualizations (not raw), colorized for quick interpretation."
        )
    if "rgb" in modalities:
        descriptions.append("RGB: Standard aerial/satellite RGB image showing natural colors.")
    if "dsm" in modalities:
        # Provide color hint depending on the selected colormap, fall back to generic wording
        dsm_cmap = (dsm_colormap or "").lower()
        if dsm_cmap == "terrain":
            color_hint = "green = lowest elevation; brown/white = highest elevation"
        elif dsm_cmap == "jet":
            color_hint = "blue = lowest elevation; red = highest elevation"
        elif dsm_cmap == "viridis":
            color_hint = "dark blue/purple = lowest; yellow = highest"
        elif dsm_cmap == "plasma":
            color_hint = "dark purple = lowest; yellow = highest"
        elif dsm_cmap == "magma":
            color_hint = "black/purple = lowest; yellow/white = highest"
        elif dsm_cmap == "inferno":
            color_hint = "black/purple = lowest; yellow = highest"
        elif dsm_cmap == "turbo":
            color_hint = "dark blue = lowest; bright yellow = highest"
        else:
            color_hint = "colorbar shows low→high elevation per colormap"
        descriptions.append(
            f"DSM: Digital Surface Model visualized using '{dsm_colormap}'. {color_hint}. If a colorbar is visible, its labels are elevation in meters."
        )
    if "svf" in modalities:
        descriptions.append(
            f"SVF: Sky View Factor (0–1) visualized using '{svf_colormap}'. Dark blue/purple = low SVF (more obstruction), red/yellow = high SVF (more open sky)."
        )
    if "seg" in modalities:
        class_names = get_geonrw_class_names()
        colors = get_geonrw_class_colors()
        color_mappings = [f"{class_names[c]}=RGB{tuple(colors[c])}" for c in range(1, 11)]
        descriptions.append("SEG: Land-use segmentation (color-coded classes). " + "; ".join(color_mappings))
    if not descriptions:
        return ""
    if len(descriptions) == 1:
        return "**Image Analysis Guide:**\n• " + descriptions[0]
    return "**Image Analysis Guide:**\n" + "\n".join([f"• {d}" for d in descriptions])


def prepare_multimodal_message(
    image_paths: Dict[str, Optional[str]],
    text: str,
    modalities: List[str],
    dsm_colormap: str = "terrain",
    svf_colormap: str = "plasma",
    is_free: bool = False,
    dsm_show_colorbar: bool = False,
    dsm_vmin: Optional[float] = None,
    dsm_vmax: Optional[float] = None,
    svf_show_colorbar: bool = False,
    svf_vmin: Optional[float] = 0.0,
    svf_vmax: Optional[float] = 1.0,
) -> Tuple[list, list]:
    """Build a messages list with multiple modality images as PIL.Image.

    Returns (messages, used_modalities).
    """
    content: list = []
    used_modalities: List[str] = []

    # Add images in requested order
    for modality in modalities:
        path = image_paths.get(modality)
        if not path:
            continue
        img: Optional[Image.Image] = None
        if modality == "rgb":
            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:
                print(f"RGB open error: {e}, path: {path}")
        elif modality == "dsm":
            img = dsm_to_rgb(path, colormap=dsm_colormap, show_colorbar=dsm_show_colorbar, vmin=dsm_vmin, vmax=dsm_vmax)
        elif modality == "svf":
            img = svf_to_rgb(
                path,
                colormap=svf_colormap,
                show_colorbar=svf_show_colorbar,
                vmin=svf_vmin,
                vmax=svf_vmax,
            )
        elif modality == "seg":
            img = seg_to_rgb(path)
        if img is not None:
            content.append({"type": "image", "image": img})
            used_modalities.append(modality)

    # print("used_modalities: ", used_modalities)
    # Build text with optional guide
    modality_desc = generate_modality_description(used_modalities, dsm_colormap, svf_colormap)
    if is_free and modality_desc and not (len(used_modalities) == 1 and used_modalities[0] == "rgb"):
        enhanced_text = (
            f"{modality_desc}\n\n**Question:** {text}"
        )
    elif is_free:
        enhanced_text = (
            text + "\nPlease structure your response using only <OBSERVATION> and <CONCLUSION>:\n"
            "<OBSERVATION>Statistical data and image-grounded observations only (SVF values, elevation in meters, land cover %).</OBSERVATION>\n"
            "<CONCLUSION>Base the decision strictly on <OBSERVATION> and answer consisely to the question; please mention at least one grid for the answer.</CONCLUSION>."
        )
    else:
        if modality_desc and not (len(used_modalities) == 1 and used_modalities[0] == "rgb"):
            enhanced_text = (
                f"{modality_desc}\n\n**Question:** {text}"
            )
            enhanced_text = enhanced_text + "\nPlease provide only your final short answer to the question concisely."
        else:
            enhanced_text = text + "\nPlease provide only your final short answer to the question concisely."

    content.append({"type": "text", "text": enhanced_text})

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": content,
        },
    ]
    return messages, used_modalities


