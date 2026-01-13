"""CLI to convert a single-image factory dataset into a multimodal version."""

from __future__ import annotations

import argparse
import os
import re
from typing import List
from typing import Optional
import numpy as np
import json

try:
    # Package-relative imports
    from .data_io import read_json_array, write_json_array
    from .multimodal_enricher import (
        enrich_item_modalities,
        add_modality_description_to_user_message,
    )
    from .modality_utils.image_converters import dsm_to_rgb, svf_to_rgb, seg_to_rgb  # type: ignore
except ImportError:
    # Fallback for direct script execution
    from data_io import read_json_array, write_json_array
    from multimodal_enricher import (
        enrich_item_modalities,
        add_modality_description_to_user_message,
    )
    from modality_utils.image_converters import dsm_to_rgb, svf_to_rgb, seg_to_rgb  # type: ignore

try:
    from .modality_utils.path_resolver import get_related_image_paths  # type: ignore
except Exception:
    try:
        from modality_utils.path_resolver import get_related_image_paths  # type: ignore
    except Exception:
        get_related_image_paths = None  # type: ignore

try:
    import rasterio  # type: ignore
except Exception:
    rasterio = None  # type: ignore


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert factory dataset to multimodal version")
    parser.add_argument("--input", required=False, default="data/svf_15x_large_answers_train_mixed_hr0.3_0803_hr0.3_ft_factory.json", type=str, help="Path to source JSON array file")
    parser.add_argument("--output", required=False, type=str, help="Destination JSON array file")
    parser.add_argument("--modalities", default="rgb,svf,dsm,seg", type=str, help="Comma list of modalities order")
    parser.add_argument("--image_dir", default="/workspace/GeoNRW", type=str, help="Base dir for RGB")
    parser.add_argument("--dsm_dir", default="/workspace/GeoNRW", type=str, help="Base dir for DSM")
    parser.add_argument("--svf_dir", default="/workspace/GeoNRW/svf/skyview_umep_train/", type=str, help="Base dir for SVF")
    parser.add_argument("--seg_dir", default="/workspace/GeoNRW", type=str, help="Base dir for SEG")
    parser.add_argument("--dsm_colormap", default="terrain", type=str)
    parser.add_argument("--svf_colormap", default="plasma", type=str)
    parser.add_argument("--add_modality_guide", default=True, action="store_true", help="Prepend modality guide to question")
    parser.add_argument("--limit", type=int, default=-1, help="Process first N items only (-1 all)")
    parser.add_argument("--record_dsm_max", action="store_true", help="Record per-image DSM max value into item metadata")
    # Infer modalities per item based on the first sentence of the question
    parser.add_argument("--infer_modalities", default=True, action="store_true", help="Infer required modalities (svf/dsm/seg) from question prefix and union with RGB")
    parser.add_argument("--infer_mode", choices=["merge", "override"], default="override", help="merge (default): union with --modalities; override: use only inferred (fallback to --modalities if none)")
    parser.add_argument("--infer_cache_path", type=str, default="infer_cache.json", help="Optional JSON cache file to memoize question-signature -> modalities")
    parser.add_argument("--preview_dir", type=str, default=None, help="If set, export preview images and prompt for first K items")
    parser.add_argument("--preview_count", type=int, default=5, help="Number of preview items to export")
    # Half RGB+DSM-visualized and half RGB-only conversion mode
    parser.add_argument("--pair_rgb_dsmviz_half", action="store_true", help="For half items, set images=[rgb, dsm_viz_png]; others images=[rgb] only")
    parser.add_argument("--dsm_viz_dir", type=str, default="/workspace/GeoNRW/dsm_viz", help="Directory to store DSM visualization PNGs (used with --pair_rgb_dsmviz_half)")
    parser.add_argument("--dsm_viz_ratio", type=float, default=0.5, help="Ratio of items to include DSM viz (0.0-1.0); overrides strict half")
    parser.add_argument("--dsm_viz_format", type=str, default="jpeg", help="DSM viz image format: png or jpeg")
    parser.add_argument("--dsm_viz_quality", type=int, default=80, help="JPEG quality (if format=jpeg)")
    parser.add_argument("--dsm_viz_show_colorbar", action="store_false", dest="dsm_viz_show_colorbar", help="Hide colorbar on DSM viz (smaller file)")
    parser.set_defaults(dsm_viz_show_colorbar=False)
    parser.add_argument("--dsm_viz_max_edge", type=int, default=1024, help="Resize DSM viz so that the longer edge <= this pixels (0 to disable)")
    parser.add_argument("--dsm_viz_max_files", type=int, default=2000, help="Max number of DSM viz images to generate (-1 for unlimited)")
    parser.add_argument("--dsm_viz_colorbar_first_n", type=int, default=100, help="Show DSM colorbar (legend) for first N items when generating DSM visualizations")
    # Half RGB+SVF-visualized and half RGB-only conversion mode (SVF pairing similar to DSM)
    parser.add_argument("--pair_rgb_svfviz_half", action="store_true", help="For half items, set images=[rgb, svf_viz_png]; others images=[rgb] only")
    parser.add_argument("--svf_viz_dir", type=str, default="/workspace/GeoNRW/svf_viz", help="Directory to store SVF visualization PNGs (used with --pair_rgb_svfviz_half)")
    parser.add_argument("--svf_viz_ratio", type=float, default=0, help="Ratio of items to include SVF viz (0.0-1.0); overrides strict half")
    parser.add_argument("--svf_viz_format", type=str, default="jpeg", help="SVF viz image format: png or jpeg")
    parser.add_argument("--svf_viz_quality", type=int, default=90, help="JPEG quality (if format=jpeg)")
    parser.add_argument("--svf_viz_show_colorbar", action="store_false", dest="svf_viz_show_colorbar", help="Hide colorbar on SVF viz (smaller file)")
    parser.set_defaults(svf_viz_show_colorbar=False)
    parser.add_argument("--svf_viz_max_edge", type=int, default=1024, help="Resize SVF viz so that the longer edge <= this pixels (0 to disable)")
    parser.add_argument("--svf_viz_max_files", type=int, default=2000, help="Max number of SVF viz images to generate (-1 for unlimited)")
    # SEG visualization options (used when --infer_modalities)
    parser.add_argument("--seg_viz_dir", type=str, default="/workspace/GeoNRW/seg_viz", help="Directory to store SEG visualization PNGs (used when --infer_modalities)")
    parser.add_argument("--seg_viz_format", type=str, default="jpeg", help="SEG viz image format: png or jpeg")
    parser.add_argument("--seg_viz_quality", type=int, default=90, help="JPEG quality (if format=jpeg)")
    parser.add_argument("--seg_viz_max_edge", type=int, default=1024, help="Resize SEG viz so that the longer edge <= this pixels (0 to disable)")
    parser.add_argument("--seg_viz_max_files", type=int, default=2000, help="Max number of SEG viz images to generate (-1 for unlimited)")
    return parser.parse_args()


def _extract_first_question_sentence(item: dict) -> str:
    """Extract the first sentence of the user's question, stripping leading <image> tokens.

    We look for the first user message with string content, remove any leading
    "<image>" tokens and newlines, then return text up to the first question mark
    or the first line-break. If nothing is found, return an empty string.
    """
    try:
        messages = item.get("messages") or []
        for msg in messages:
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                s: str = msg.get("content") or ""
                s = s.lstrip()
                while s.startswith("<image>"):
                    s = s[len("<image>") :]
                    if s.startswith("\n"):
                        s = s[1:]
                    s = s.lstrip()
                qmark = s.find("?")
                if qmark >= 0:
                    s = s[: qmark + 1]
                else:
                    s = s.splitlines()[0] if s.splitlines() else s
                return s.strip()
    except Exception:
        return ""
    return ""


def _infer_modalities_from_question(first_sentence: str) -> List[str]:
    """Infer additional modalities required beyond RGB from the first question sentence.

    Heuristics follow our template question families:
    - Sun exposure / SVF questions -> svf
    - SVF variability/value/sky view -> svf
    - Height/elevation average/highest -> dsm
    - Visibility range / viewshed -> svf + dsm
    - Urban/building density -> seg + dsm + svf
    - Openness assessment (spatial openness) -> svf + dsm + seg
    - Land-cover / land-use recognition/comparison -> seg
    - Water accumulation -> dsm
    Returns a modality list in canonical order: ["rgb", "svf", "dsm", "seg"] without duplicates.
    """
    s = (first_sentence or "").lower()
    required: List[str] = []

    def need(mods: List[str]) -> None:
        for m in mods:
            if m not in required:
                required.append(m)

    if re.search(r"sun\s*exposure|solar\s*exposure|sunlight", s):
        need(["svf"])
    if re.search(r"svf\s*variability|svf\s*value|sky\s*view\s*factor|\bsvf\b", s):
        need(["svf"])
    if re.search(r"height|elevation|highest\s*region|highest\s*location|average\s*height|mean\s*elevation", s):
        need(["dsm"])
    if re.search(r"visibility\s*range|see\s*the\s*furthest|view\s*range|viewshed|line\s*of\s*sight", s):
        need(["svf", "dsm"])
    if re.search(r"sky\s*visibility|see\s*the\s*most\s*sky|skyward\s*view|upward\s*view|largest\s*portion\s*of\s*the\s*sky|open\s*sky\s*view|sky\s*access", s):
        need(["svf", "seg"])
    if re.search(r"urban\s*density|building\s*density|tightly\s*packed|densest", s):
        need(["seg", "dsm", "svf"])
    if re.search(r"openness|spatial\s*openness|open\s*sky", s):
        need(["svf", "dsm", "seg"])
    if re.search(r"land-?cover|land\s*use|segmentation", s):
        need(["seg"])
    if re.search(r"water\s*accumulation|flood|ponding", s):
        need(["dsm"])

    canonical = ["rgb", "svf", "dsm", "seg"]
    ordered = [m for m in canonical if m in required]
    return ordered


def _build_question_signature(first_sentence: str) -> str:
    """Build a stable, short signature for memoization.

    Lowercase, collapse spaces, keep only alnum and a few delimiters, and
    truncate to 160 chars. This is used as the key for the infer cache.
    """
    s = (first_sentence or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^a-z0-9 _-]", "", s)
    return s[:50]


def main() -> None:
    args = _parse_args()

    data = read_json_array(args.input)
    desired_modalities: List[str] = [m.strip() for m in (args.modalities or "rgb").split(",") if m.strip()]

    infer_cache: dict = {}
    if getattr(args, "infer_cache_path", None):
        try:
            if os.path.exists(args.infer_cache_path):
                with open(args.infer_cache_path, "r", encoding="utf-8") as f:
                    infer_cache = json.load(f)
        except Exception:
            infer_cache = {}

    if not args.output:
        base, ext = os.path.splitext(args.input)
        if args.pair_rgb_dsmviz_half:
            args.output = base + "_mm_dsmviz_half" + ext
        elif args.pair_rgb_svfviz_half:
            args.output = base + "_mm_svfviz_half" + ext
        elif args.record_dsm_max:
            args.output = base + "_mm_dsm_max" + ext
        else:
            args.output = base + "_mm" + ext

    out_data = []
    dsm_max_cache = {}
    dsm_min_cache = {}
    dsm_viz_cache = {}
    svf_viz_cache = {}
    seg_viz_cache = {}
    preview_emitted = 0
    dsm_viz_saved_count = 0
    svf_viz_saved_count = 0
    seg_viz_saved_count = 0
    items = data if args.limit is None or args.limit < 0 else data[: args.limit]
    for idx, item in enumerate(items):
        per_item_modalities: List[str] = list(desired_modalities)
        if getattr(args, "infer_modalities", False):
            try:
                q_first = _extract_first_question_sentence(item)
                sig = _build_question_signature(q_first)
                if sig and sig in infer_cache:
                    inferred_list = infer_cache.get(sig) or []
                else:
                    inferred_list = _infer_modalities_from_question(q_first)
                    if sig:
                        infer_cache[sig] = inferred_list
            except Exception:
                inferred_list = []
            mode = str(getattr(args, "infer_mode", "merge")).lower()
            if mode == "override" and inferred_list:
                per_item_modalities = ["rgb"] + [m for m in inferred_list if m != "rgb"]
            elif mode == "merge":
                canonical = ["rgb", "svf", "dsm", "seg"]
                base = [m for m in per_item_modalities if m in canonical]
                for m in canonical:
                    if m in inferred_list and m not in base:
                        base.append(m)
                per_item_modalities = base

        enriched = enrich_item_modalities(
            item,
            dsm_dir=args.dsm_dir,
            svf_dir=args.svf_dir,
            seg_dir=args.seg_dir,
            desired_modalities=per_item_modalities,
        )

        rgb_candidates: List[str] = enriched.get("images") or item.get("images") or []
        rgb_path: str = rgb_candidates[0] if rgb_candidates else ""
        paths = (
            get_related_image_paths(
                rgb_path,
                dsm_dir=args.dsm_dir,
                svf_dir=args.svf_dir,
                seg_dir=args.seg_dir,
            )
            if (get_related_image_paths and rgb_path)
            else None
        )
        dsm_path = paths.get("dsm") if paths else None
        svf_path = paths.get("svf") if paths else None

        if args.pair_rgb_dsmviz_half:
            if not args.dsm_viz_dir:
                try:
                    args.dsm_viz_dir = os.path.join(os.path.dirname(args.output), "dsm_viz")
                except Exception:
                    args.dsm_viz_dir = "dsm_viz"
            os.makedirs(args.dsm_viz_dir, exist_ok=True)

        # Determine SVF viz directory if needed
        if args.pair_rgb_svfviz_half:
            if not args.svf_viz_dir:
                try:
                    args.svf_viz_dir = os.path.join(os.path.dirname(args.output), "svf_viz")
                except Exception:
                    args.svf_viz_dir = "svf_viz"
            os.makedirs(args.svf_viz_dir, exist_ok=True)

        if args.infer_modalities and not args.pair_rgb_svfviz_half and not args.pair_rgb_dsmviz_half:
            if not args.seg_viz_dir:
                try:
                    args.seg_viz_dir = os.path.join(os.path.dirname(args.output), "seg_viz")
                except Exception:
                    args.seg_viz_dir = "seg_viz"
            os.makedirs(args.seg_viz_dir, exist_ok=True)

        if args.record_dsm_max:
            try:
                if dsm_path and os.path.exists(dsm_path) and rasterio is not None:
                    have_cached = dsm_path in dsm_max_cache and dsm_path in dsm_min_cache
                    if have_cached:
                        dsm_max_val = dsm_max_cache[dsm_path]
                        dsm_min_val = dsm_min_cache[dsm_path]
                    else:
                        with rasterio.open(dsm_path) as src:
                            arr = src.read(1)
                        dsm_max_val = float(np.nanmax(arr))
                        dsm_min_val = float(np.nanmin(arr))
                        dsm_max_cache[dsm_path] = dsm_max_val
                        dsm_min_cache[dsm_path] = dsm_min_val
                    enriched = dict(enriched)
                    meta = dict(enriched.get("meta") or {})
                    meta["dsm_max"] = dsm_max_val
                    meta["dsm_min"] = dsm_min_val
                    enriched["meta"] = meta
                    enriched["dsm_max"] = dsm_max_val
                    enriched["dsm_min"] = dsm_min_val
            except Exception:
                pass

        if args.pair_rgb_dsmviz_half and rgb_path:
            try:
                ratio = max(0.0, min(1.0, float(args.dsm_viz_ratio))) if args.dsm_viz_ratio is not None else None
                if ratio is not None and len(items) > 0:
                    use_viz = (float(idx) / float(len(items)) < ratio)
                else:
                    use_viz = (idx % 2 == 0)
                selected_images: List[str] = []
                selected_modalities: List[str] = []
                selected_images.append(rgb_path)
                selected_modalities.append("rgb")
                if use_viz and dsm_path and os.path.exists(dsm_path):
                    if args.dsm_viz_max_files is not None and int(args.dsm_viz_max_files) >= 0:
                        if dsm_path not in dsm_viz_cache and dsm_viz_saved_count >= int(args.dsm_viz_max_files):
                            viz_png = None
                            if viz_png:
                                selected_images.append(viz_png)
                                selected_modalities.append("dsm")
                            raise Exception("_SKIP_VIZ_NEW_")
                    if dsm_path in dsm_viz_cache and os.path.exists(dsm_viz_cache[dsm_path]):
                        viz_png = dsm_viz_cache[dsm_path]
                    else:
                        try:
                            show_cb = bool(args.dsm_viz_show_colorbar) or (isinstance(getattr(args, "dsm_viz_colorbar_first_n", 0), int) and idx < int(args.dsm_viz_colorbar_first_n))
                            img = dsm_to_rgb(dsm_path, colormap=args.dsm_colormap, show_colorbar=show_cb)
                            if img is not None and args.dsm_viz_max_edge and int(args.dsm_viz_max_edge) > 0:
                                try:
                                    max_edge = int(args.dsm_viz_max_edge)
                                    w, h = img.size
                                    if max(w, h) > max_edge:
                                        img.thumbnail((max_edge, max_edge))
                                except Exception:
                                    pass
                            ext = ".png" if (str(args.dsm_viz_format).lower() == "png") else ".jpg"
                            try:
                                base_dir = args.image_dir or "/"
                                rel = os.path.relpath(rgb_path, base_dir)
                            except Exception:
                                rel = os.path.basename(rgb_path)
                            rel = rel.replace(os.sep, "_")
                            rel = re.sub(r"[^A-Za-z0-9._-]+", "_", rel)
                            if "." in rel:
                                rel_noext = rel[: rel.rfind('.')]
                            else:
                                rel_noext = rel
                            if rel_noext.endswith("_rgb"):
                                rel_noext = rel_noext[:-4]
                            viz_basename = f"{rel_noext}_dsm{ext}"
                            viz_png = os.path.join(args.dsm_viz_dir, viz_basename)
                            if os.path.exists(viz_png):
                                dsm_viz_cache[dsm_path] = viz_png
                            elif img is not None:
                                save_kwargs = {}
                                if ext == ".jpg":
                                    try:
                                        img = img.convert("RGB")
                                    except Exception:
                                        pass
                                    save_kwargs.update(dict(quality=int(args.dsm_viz_quality or 80), optimize=True))
                                img.save(viz_png, **save_kwargs)
                                dsm_viz_cache[dsm_path] = viz_png
                                dsm_viz_saved_count += 1
                            else:
                                viz_png = None
                        except Exception:
                            viz_png = None
                    if viz_png:
                        selected_images.append(viz_png)
                        selected_modalities.append("dsm")
                if selected_images:
                    enriched = dict(enriched)
                    enriched["images"] = selected_images
                    enriched["modalities"] = selected_modalities
            except Exception:
                pass

        if args.pair_rgb_svfviz_half and rgb_path:
            try:
                ratio = max(0.0, min(1.0, float(args.svf_viz_ratio))) if args.svf_viz_ratio is not None else None
                if ratio is not None and len(items) > 0:
                    use_viz = (float(idx) / float(len(items)) < ratio)
                else:
                    use_viz = (idx % 2 == 0)
                selected_images: List[str] = []
                selected_modalities: List[str] = []
                selected_images.append(rgb_path)
                selected_modalities.append("rgb")
                if use_viz and svf_path and os.path.exists(svf_path):
                    if args.svf_viz_max_files is not None and int(args.svf_viz_max_files) >= 0:
                        if svf_path not in svf_viz_cache and svf_viz_saved_count >= int(args.svf_viz_max_files):
                            viz_png = None
                            if viz_png:
                                selected_images.append(viz_png)
                                selected_modalities.append("svf")
                            raise Exception("_SKIP_VIZ_NEW_")
                    if svf_path in svf_viz_cache and os.path.exists(svf_viz_cache[svf_path]):
                        viz_png = svf_viz_cache[svf_path]
                    else:
                        try:
                            img = svf_to_rgb(svf_path, colormap=args.svf_colormap, show_colorbar=bool(args.svf_viz_show_colorbar))
                            if img is not None and args.svf_viz_max_edge and int(args.svf_viz_max_edge) > 0:
                                try:
                                    max_edge = int(args.svf_viz_max_edge)
                                    w, h = img.size
                                    if max(w, h) > max_edge:
                                        img.thumbnail((max_edge, max_edge))
                                except Exception:
                                    pass
                            ext = ".png" if (str(args.svf_viz_format).lower() == "png") else ".jpg"
                            try:
                                base_dir = args.image_dir or "/"
                                rel = os.path.relpath(rgb_path, base_dir)
                            except Exception:
                                rel = os.path.basename(rgb_path)
                            rel = rel.replace(os.sep, "_")
                            rel = re.sub(r"[^A-Za-z0-9._-]+", "_", rel)
                            if "." in rel:
                                rel_noext = rel[: rel.rfind('.')]
                            else:
                                rel_noext = rel
                            if rel_noext.endswith("_rgb"):
                                rel_noext = rel_noext[:-4]
                            viz_basename = f"{rel_noext}_svf{ext}"
                            viz_png = os.path.join(args.svf_viz_dir, viz_basename)
                            if os.path.exists(viz_png):
                                svf_viz_cache[svf_path] = viz_png
                            elif img is not None:
                                save_kwargs = {}
                                if ext == ".jpg":
                                    try:
                                        img = img.convert("RGB")
                                    except Exception:
                                        pass
                                    save_kwargs.update(dict(quality=int(args.svf_viz_quality or 90), optimize=True))
                                img.save(viz_png, **save_kwargs)
                                svf_viz_cache[svf_path] = viz_png
                                svf_viz_saved_count += 1
                            else:
                                viz_png = None
                        except Exception:
                            viz_png = None
                    if viz_png:
                        selected_images.append(viz_png)
                        selected_modalities.append("svf")
                if selected_images:
                    enriched = dict(enriched)
                    enriched["images"] = selected_images
                    enriched["modalities"] = selected_modalities
            except Exception:
                pass

        if args.infer_modalities and not args.pair_rgb_dsmviz_half and not args.pair_rgb_svfviz_half and rgb_path:
            try:
                ordered_modalities: List[str] = enriched.get("modalities") or []
                if ordered_modalities:
                    selected_images: List[str] = []
                    selected_modalities: List[str] = []
                    for m in ordered_modalities:
                        if m == "rgb":
                            selected_images.append(rgb_path)
                            selected_modalities.append("rgb")
                            continue
                        if m == "dsm" and dsm_path and os.path.exists(dsm_path):
                            if args.dsm_viz_max_files is not None and int(args.dsm_viz_max_files) >= 0:
                                if dsm_path not in dsm_viz_cache and dsm_viz_saved_count >= int(args.dsm_viz_max_files):
                                    selected_images.append(dsm_path)
                                    selected_modalities.append("dsm")
                                    continue
                            if dsm_path in dsm_viz_cache and os.path.exists(dsm_viz_cache[dsm_path]):
                                viz_png = dsm_viz_cache[dsm_path]
                            else:
                                try:
                                    show_cb = bool(args.dsm_viz_show_colorbar) or (isinstance(getattr(args, "dsm_viz_colorbar_first_n", 0), int) and idx < int(args.dsm_viz_colorbar_first_n))
                                    img = dsm_to_rgb(dsm_path, colormap=args.dsm_colormap, show_colorbar=show_cb)
                                    if img is not None and args.dsm_viz_max_edge and int(args.dsm_viz_max_edge) > 0:
                                        try:
                                            max_edge = int(args.dsm_viz_max_edge)
                                            w, h = img.size
                                            if max(w, h) > max_edge:
                                                img.thumbnail((max_edge, max_edge))
                                        except Exception:
                                            pass
                                    ext = ".png" if (str(args.dsm_viz_format).lower() == "png") else ".jpg"
                                    try:
                                        base_dir = args.image_dir or "/"
                                        rel = os.path.relpath(rgb_path, base_dir)
                                    except Exception:
                                        rel = os.path.basename(rgb_path)
                                    rel = rel.replace(os.sep, "_")
                                    rel = re.sub(r"[^A-Za-z0-9._-]+", "_", rel)
                                    rel_noext = rel[: rel.rfind('.')] if "." in rel else rel
                                    if rel_noext.endswith("_rgb"):
                                        rel_noext = rel_noext[:-4]
                                    viz_basename = f"{rel_noext}_dsm{ext}"
                                    viz_png = os.path.join(args.dsm_viz_dir, viz_basename)
                                    if os.path.exists(viz_png):
                                        dsm_viz_cache[dsm_path] = viz_png
                                    elif img is not None:
                                        save_kwargs = {}
                                        if ext == ".jpg":
                                            try:
                                                img = img.convert("RGB")
                                            except Exception:
                                                pass
                                            save_kwargs.update(dict(quality=int(args.dsm_viz_quality or 80), optimize=True))
                                        img.save(viz_png, **save_kwargs)
                                        dsm_viz_cache[dsm_path] = viz_png
                                        dsm_viz_saved_count += 1
                                except Exception:
                                    viz_png = None  # type: ignore
                            selected_images.append(viz_png if (viz_png and os.path.exists(viz_png)) else dsm_path)
                            selected_modalities.append("dsm")
                            continue
                        if m == "svf" and svf_path and os.path.exists(svf_path):
                            if args.svf_viz_max_files is not None and int(args.svf_viz_max_files) >= 0:
                                if svf_path not in svf_viz_cache and svf_viz_saved_count >= int(args.svf_viz_max_files):
                                    selected_images.append(svf_path)
                                    selected_modalities.append("svf")
                                    continue
                            if svf_path in svf_viz_cache and os.path.exists(svf_viz_cache[svf_path]):
                                viz_png = svf_viz_cache[svf_path]
                            else:
                                try:
                                    img = svf_to_rgb(svf_path, colormap=args.svf_colormap, show_colorbar=bool(args.svf_viz_show_colorbar))
                                    if img is not None and args.svf_viz_max_edge and int(args.svf_viz_max_edge) > 0:
                                        try:
                                            max_edge = int(args.svf_viz_max_edge)
                                            w, h = img.size
                                            if max(w, h) > max_edge:
                                                img.thumbnail((max_edge, max_edge))
                                        except Exception:
                                            pass
                                    ext = ".png" if (str(args.svf_viz_format).lower() == "png") else ".jpg"
                                    try:
                                        base_dir = args.image_dir or "/"
                                        rel = os.path.relpath(rgb_path, base_dir)
                                    except Exception:
                                        rel = os.path.basename(rgb_path)
                                    rel = rel.replace(os.sep, "_")
                                    rel = re.sub(r"[^A-Za-z0-9._-]+", "_", rel)
                                    rel_noext = rel[: rel.rfind('.')] if "." in rel else rel
                                    if rel_noext.endswith("_rgb"):
                                        rel_noext = rel_noext[:-4]
                                    viz_basename = f"{rel_noext}_svf{ext}"
                                    viz_png = os.path.join(args.svf_viz_dir, viz_basename)
                                    if os.path.exists(viz_png):
                                        svf_viz_cache[svf_path] = viz_png
                                    elif img is not None:
                                        save_kwargs = {}
                                        if ext == ".jpg":
                                            try:
                                                img = img.convert("RGB")
                                            except Exception:
                                                pass
                                            save_kwargs.update(dict(quality=int(args.svf_viz_quality or 90), optimize=True))
                                        img.save(viz_png, **save_kwargs)
                                        svf_viz_cache[svf_path] = viz_png
                                        svf_viz_saved_count += 1
                                except Exception:
                                    viz_png = None  # type: ignore
                            selected_images.append(viz_png if (viz_png and os.path.exists(viz_png)) else svf_path)
                            selected_modalities.append("svf")
                            continue
                        if m == "seg":
                            seg_path = paths.get("seg") if paths else None
                            if seg_path and os.path.exists(seg_path):
                                if args.seg_viz_max_files is not None and int(args.seg_viz_max_files) >= 0:
                                    if seg_path not in seg_viz_cache and seg_viz_saved_count >= int(args.seg_viz_max_files):
                                        selected_images.append(seg_path)
                                        selected_modalities.append("seg")
                                        continue
                                if seg_path in seg_viz_cache and os.path.exists(seg_viz_cache[seg_path]):
                                    viz_png = seg_viz_cache[seg_path]
                                else:
                                    try:
                                        img = seg_to_rgb(seg_path)
                                        if img is not None and args.seg_viz_max_edge and int(args.seg_viz_max_edge) > 0:
                                            try:
                                                max_edge = int(args.seg_viz_max_edge)
                                                w, h = img.size
                                                if max(w, h) > max_edge:
                                                    img.thumbnail((max_edge, max_edge))
                                            except Exception:
                                                pass
                                        ext = ".png" if (str(args.seg_viz_format).lower() == "png") else ".jpg"
                                        try:
                                            base_dir = args.image_dir or "/"
                                            rel = os.path.relpath(rgb_path, base_dir)
                                        except Exception:
                                            rel = os.path.basename(rgb_path)
                                        rel = rel.replace(os.sep, "_")
                                        rel = re.sub(r"[^A-Za-z0-9._-]+", "_", rel)
                                        rel_noext = rel[: rel.rfind('.')] if "." in rel else rel
                                        if rel_noext.endswith("_rgb"):
                                            rel_noext = rel_noext[:-4]
                                        viz_basename = f"{rel_noext}_seg{ext}"
                                        viz_png = os.path.join(args.seg_viz_dir, viz_basename)
                                        if os.path.exists(viz_png):
                                            seg_viz_cache[seg_path] = viz_png
                                        elif img is not None:
                                            save_kwargs = {}
                                            if ext == ".jpg":
                                                try:
                                                    img = img.convert("RGB")
                                                except Exception:
                                                    pass
                                                save_kwargs.update(dict(quality=int(args.seg_viz_quality or 90), optimize=True))
                                            img.save(viz_png, **save_kwargs)
                                            seg_viz_cache[seg_path] = viz_png
                                            seg_viz_saved_count += 1
                                    except Exception:
                                        viz_png = None  # type: ignore
                                selected_images.append(viz_png if (viz_png and os.path.exists(viz_png)) else seg_path)
                                selected_modalities.append("seg")
                                continue
                        if m == "svf" and svf_path and os.path.exists(svf_path):
                            selected_images.append(svf_path)
                            selected_modalities.append("svf")
                        elif m == "dsm" and dsm_path and os.path.exists(dsm_path):
                            selected_images.append(dsm_path)
                            selected_modalities.append("dsm")
                        elif m == "seg" and paths and paths.get("seg") and os.path.exists(paths.get("seg")):
                            selected_images.append(paths.get("seg"))
                            selected_modalities.append("seg")
                    if selected_images:
                        enriched = dict(enriched)
                        enriched["images"] = selected_images
                        enriched["modalities"] = selected_modalities
            except Exception:
                pass
        if not args.record_dsm_max:
            try:
                if dsm_path and os.path.exists(dsm_path) and rasterio is not None:
                    with rasterio.open(dsm_path) as src:
                        arr = src.read(1)
                    dsm_max_val = float(np.nanmax(arr))
                    dsm_min_val = float(np.nanmin(arr))
                    enriched = dict(enriched)
                    enriched["dsm_max"] = dsm_max_val
                    enriched["dsm_min"] = dsm_min_val
            except Exception:
                pass

        if args.add_modality_guide:
            mods = enriched.get("modalities") or desired_modalities
            enriched = add_modality_description_to_user_message(
                enriched, mods, dsm_colormap=args.dsm_colormap, svf_colormap=args.svf_colormap
            )

        out_data.append(enriched)

        try:
            if args.preview_dir and preview_emitted < max(0, int(args.preview_count or 0)):
                os.makedirs(args.preview_dir, exist_ok=True)
                item_idx = preview_emitted
                ordered_modalities: List[str] = enriched.get("modalities") or []
                saved_rel_paths = []
                for m in ordered_modalities:
                    p = (paths.get(m) if paths else None)
                    if not p or not os.path.exists(p):
                        continue
                    out_path = os.path.join(args.preview_dir, f"{item_idx:04d}_{m}.png")
                    img_obj = None
                    if m == "rgb":
                        try:
                            from PIL import Image  # type: ignore
                            img_obj = Image.open(p).convert("RGB")
                        except Exception:
                            img_obj = None
                    elif m == "dsm":
                        img_obj = dsm_to_rgb(p, colormap=args.dsm_colormap, show_colorbar=True)
                    elif m == "svf":
                        img_obj = svf_to_rgb(p, colormap=args.svf_colormap, show_colorbar=True)
                    elif m == "seg":
                        img_obj = seg_to_rgb(p)
                    if img_obj is not None:
                        try:
                            img_obj.save(out_path)
                            saved_rel_paths.append({"modality": m, "file": out_path})
                        except Exception:
                            pass
                prompt_text = ""
                try:
                    for msg in (enriched.get("messages") or []):
                        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                            prompt_text = msg.get("content") or ""
                            break
                except Exception:
                    prompt_text = ""
                try:
                    meta_out = {
                        "index": item_idx,
                        "modalities": ordered_modalities,
                        "images": saved_rel_paths,
                        "dsm_min": enriched.get("dsm_min"),
                        "dsm_max": enriched.get("dsm_max"),
                        "prompt": prompt_text,
                    }
                    with open(os.path.join(args.preview_dir, f"{item_idx:04d}_preview.json"), "w", encoding="utf-8") as f:
                        import json  # type: ignore
                        f.write(json.dumps(meta_out, ensure_ascii=False, indent=2))
                except Exception:
                    pass
                preview_emitted += 1
        except Exception:
            pass

    write_json_array(args.output, out_data)
    print(f"Wrote multimodal dataset: {args.output} ({len(out_data)} items)")

    try:
        if getattr(args, "infer_cache_path", None):
            os.makedirs(os.path.dirname(args.infer_cache_path), exist_ok=True)
            with open(args.infer_cache_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(infer_cache, ensure_ascii=False, indent=2))
    except Exception:
        pass


if __name__ == "__main__":
    main()


