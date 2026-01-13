"""Helpers to resolve related modality file paths from an RGB path.

Functions focus on simple, explicit path building based on dataset naming.
"""

import os
from pathlib import Path
from typing import Dict, Optional


def get_related_image_paths(rgb_path: str, dsm_dir: Optional[str], svf_dir: Optional[str], seg_dir: Optional[str]) -> Dict[str, Optional[str]]:
    """Resolve DSM/SVF/SEG paths from the given RGB path and base dirs.

    The function expects file naming like:
    - RGB: <base>_rgb.jp2
    - DSM: <base>_dem.tif
    - SVF: <base>_dem_svf_umep.tif
    - SEG: <base>_seg.tif
    """
    try:
        rgb_p = Path(rgb_path)
        if not rgb_p.exists():
            print(f"Warning: RGB image does not exist: {rgb_p}")
            return {"rgb": None, "dsm": None, "svf": None, "seg": None}

        base_dir = rgb_p.parent
        city = base_dir.name

        if "_rgb.jp2" in rgb_p.name:
            base_name = rgb_p.name.replace("_rgb.jp2", "")
        else:
            base_name = rgb_p.stem

        dsm_path = os.path.join(dsm_dir, city, f"{base_name}_dem.tif") if dsm_dir else None
        svf_path = os.path.join(svf_dir, city, f"{base_name}_dem_svf_umep.tif") if svf_dir else None
        seg_path = os.path.join(seg_dir, city, f"{base_name}_seg.tif") if seg_dir else None

        paths: Dict[str, Optional[str]] = {
            "rgb": str(rgb_p) if rgb_p.exists() else None,
            "dsm": dsm_path if dsm_path and os.path.exists(dsm_path) else None,
            "svf": svf_path if svf_path and os.path.exists(svf_path) else None,
            "seg": seg_path if seg_path and os.path.exists(seg_path) else None,
        }
        return paths
    except Exception as e:
        print(f"Path resolve error: {e}, rgb: {rgb_path}")
        return {"rgb": None, "dsm": None, "svf": None, "seg": None}


