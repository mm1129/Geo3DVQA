#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper to scan a directory and yield prediction JSONL file paths.
This module isolates filesystem concerns from business logic.
"""

import os
from pathlib import Path
from typing import Iterable, List


def list_jsonl_files(target_dir: str, name_contains: str = "", recursive: bool = True) -> List[str]:
    """Return JSONL file paths under the directory, optionally filtered by substring.

    Args:
        target_dir: Directory to search under.
        name_contains: If provided, only include files whose names contain this substring.
        recursive: Recurse into subdirectories.

    Returns:
        List of absolute file paths sorted by filename.
    """
    base = Path(target_dir)
    if not base.exists():
        return []
    pattern = "**/*.jsonl" if recursive else "*.jsonl"
    files = [str(p.resolve()) for p in base.glob(pattern) if (name_contains in p.name)]
    files.sort()
    return files


def ensure_same_parent_directory(file_paths: Iterable[str]) -> str:
    """Return common parent directory if all files share same parent; else empty string.

    This is useful to ensure outputs are written alongside inputs when desired.
    """
    parent_dirs = {str(Path(p).parent.resolve()) for p in file_paths}
    if len(parent_dirs) == 1:
        return next(iter(parent_dirs))
    return ""


