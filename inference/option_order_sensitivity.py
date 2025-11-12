#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Option Order Sensitivity Utility

Purpose:
- Create a variant of a questions JSONL where the order of multiple-choice options
  in the user-visible "Please choose from:" block is shuffled (or fixed), to test
  sensitivity of model accuracy to option ordering.

Scope:
- Safe, text-only rewrite of the "Please choose from:" block.
- Does NOT modify ground-truth answers. Region definitions (A:, B:, ...) and
  point coordinates appear earlier in the prompt and remain unchanged, so the
  answer strings (e.g., "Region B", "Point (x%, y%)") stay valid.

Usage:
  python option_order_sensitivity.py \
      --input questions.jsonl \
      --output questions_shuffled.jsonl \
      --mode shuffle \
      --seed 42

Modes:
- shuffle: randomize option lines within "Please choose from:" block
- fixed  : keep original order (can also be used to normalize/strip spacing)
"""

import argparse
import json
import random
import re
from typing import List, Tuple, Optional

BLOCK_START_RE = re.compile(r"(?im)^\s*Please\s+choose\s+from:\s*$")
BLOCK_END_RE = re.compile(r"(?im)^\s*Answer\s+format\s*:", re.IGNORECASE)

# Option line formats we support
REGION_LINE_RE = re.compile(r"^\s*Region\s+[A-D]\s*$", re.IGNORECASE)
POINT_LINE_RE = re.compile(r"^\s*Point\s*\(\s*\d+(\.\d+)?\%\s*,\s*\d+(\.\d+)?\%\s*\)\s*$", re.IGNORECASE)


def _find_option_block(text: str) -> Optional[Tuple[int, int, List[str]]]:
    """
    Locate the lines between 'Please choose from:' and the next 'Answer format:' (or end of text).
    Return (start_idx, end_idx, option_lines) where indices are line indices in the split text.
    If block not found or no option-like lines, return None.
    """
    lines = text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if BLOCK_START_RE.match(line):
            start_idx = i + 1
            break
    if start_idx is None:
        return None

    end_idx = None
    for j in range(start_idx, len(lines)):
        if BLOCK_END_RE.match(lines[j]):
            end_idx = j
            break
    if end_idx is None:
        end_idx = len(lines)

    # Collect contiguous option-like lines until a blank or next section
    option_lines: List[str] = []
    for k in range(start_idx, end_idx):
        line = lines[k]
        if not line.strip():
            # allow empty lines inside block; keep them as separators
            option_lines.append(line)
            continue
        if REGION_LINE_RE.match(line) or POINT_LINE_RE.match(line):
            option_lines.append(line)
        else:
            # Non-option line within the block, keep as-is but do not break
            option_lines.append(line)
    if not any(REGION_LINE_RE.match(l) or POINT_LINE_RE.match(l) for l in option_lines):
        return None
    return start_idx, end_idx, option_lines


def _shuffle_preserving_structure(option_lines: List[str], rng: random.Random) -> List[str]:
    """
    Shuffle only the lines that match option formats. Keep other lines (including blanks)
    in their relative positions if they do not match REGION/POINT formats.
    """
    # indices of true options
    idx_and_opts = [
        (i, l) for i, l in enumerate(option_lines)
        if REGION_LINE_RE.match(l) or POINT_LINE_RE.match(l)
    ]
    opts_only = [l for _, l in idx_and_opts]
    if len(opts_only) <= 1:
        return option_lines[:]  # nothing to shuffle
    rng.shuffle(opts_only)
    # rebuild
    out = option_lines[:]
    for (i, _), new_line in zip(idx_and_opts, opts_only):
        out[i] = new_line
    return out


def rewrite_question_text(text: str, mode: str, rng: random.Random) -> str:
    """
    Rewrite the question `text` with shuffled (or fixed) option order in the 'Please choose from:' block.
    """
    found = _find_option_block(text)
    if not found:
        return text
    start_idx, end_idx, option_lines = found
    lines = text.splitlines()

    if mode == "shuffle":
        new_block = _shuffle_preserving_structure(option_lines, rng)
    elif mode == "fixed":
        new_block = option_lines[:]  # no-op, but normalizes whitespace by rejoin
    else:
        raise ValueError(f"Unknown mode: {mode}")

    new_lines = lines[:start_idx] + new_block + lines[end_idx:]
    return "\n".join(new_lines)


def process_file(input_path: str, output_path: str, mode: str, seed: Optional[int]) -> None:
    rng = random.Random(seed if seed is not None else random.randrange(0, 2**32 - 1))
    count_total = 0
    count_rewritten = 0
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            count_total += 1
            text = obj.get("text", "")
            if not isinstance(text, str) or not text.strip():
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue
            new_text = rewrite_question_text(text, mode, rng)
            if new_text != text:
                count_rewritten += 1
                obj["text"] = new_text
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Processed: {count_total} lines; Rewritten blocks: {count_rewritten}; Mode: {mode}; Seed: {seed}")


def main():
    parser = argparse.ArgumentParser(description="Shuffle (or fix) option order in 'Please choose from:' blocks for sensitivity testing.")
    parser.add_argument("--input", required=True, help="Input questions JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--mode", choices=["shuffle", "fixed"], default="shuffle", help="Rewrite mode")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for shuffling (optional)")
    args = parser.parse_args()
    process_file(args.input, args.output, args.mode, args.seed)


if __name__ == "__main__":
    main()


