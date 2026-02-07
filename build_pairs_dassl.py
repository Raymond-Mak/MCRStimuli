#!/usr/bin/env python3
"""
Build image-text pairs using dataset splits defined by Dassl-style dataset adapters.

This script instantiates the dataset class from `datasets/` (Emotion6, Emoset, FI, Twitter1, Twitter2, etc.)
to obtain the train/val/test splits, then matches each image with its caption from a CSV file
and writes JSONL pairs suitable for the multimodal pipeline.

Why a new builder?
- The user requested to base the split strictly on the dataset adapters in `datasets/` and then
  pair the resulting images with captions. This script does exactly that.

Usage examples:
- Emotion6 (stratified split by default):
  python build_pairs_dassl.py --dataset Emotion6 \
    --img_root data --caption_file caption/narracap_extended_Emotion6.csv \
    --out_dir cache --out_prefix Emotion6

- Emoset (uses predefined split dirs if the adapter does):
  python build_pairs_dassl.py --dataset Emoset \
    --img_root data --caption_file caption/narracap_extended_Emoset.csv \
    --out_dir cache --out_prefix Emoset

Notes:
- We keep image paths relative to `--img_root` in the output JSONL for consistency with loaders.
- Missing captions are skipped by default, but can be optionally filled with a fallback string.
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Dassl cfg for dataset adapters
from dassl.config import get_cfg_default

# Import dataset adapters dynamically
import importlib

SPLIT_TOKENS = ("train", "val", "test", "validation")


def _import_dataset_class(dataset_name: str):
    """Import the dataset class from datasets/<Name>.py and return the class object.

    The adapter is expected to be registered with DATASET_REGISTRY but we call the
    class directly for simplicity.
    """
    module_name = f"datasets.{dataset_name}"
    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        raise ImportError(f"Failed to import dataset module '{module_name}': {e}")

    # Heuristic: class name is identical to module (e.g., Emotion6.Emotion6)
    if hasattr(module, dataset_name):
        return getattr(module, dataset_name)

    # Fallback: find first class that is not 'Datum'
    for k, v in module.__dict__.items():
        if isinstance(v, type) and k.lower() == dataset_name.lower():
            return v

    raise ImportError(f"Cannot find dataset class '{dataset_name}' in module '{module_name}'")


def _normalize_path_key(path: str) -> str:
    """Normalize path separators and strip redundant prefixes like ./"""
    path = path.replace('\\', '/')
    if path.startswith("./"):
        path = path[2:]
    return path


def _strip_leading_data(path: str) -> str:
    """Remove a leading 'data/' prefix if present."""
    if path.startswith("data/"):
        return path[5:]
    return path


def _extract_split_tail(path: str) -> str:
    """Return the path portion starting from the first split token."""
    norm = _normalize_path_key(path)
    parts = norm.split('/')
    for idx, token in enumerate(parts):
        if token.lower() in SPLIT_TOKENS:
            return '/'.join(parts[idx:])
    return norm


def _build_tail_lookup(cap_map: Dict[str, str]) -> Dict[str, str]:
    """Create a lookup using only the split-relative tail of the path."""
    tail_map: Dict[str, str] = {}
    for key, text in cap_map.items():
        tail = _extract_split_tail(key)
        # Keep the first caption we see for a tail to avoid accidental overrides
        tail_map.setdefault(tail, text)
    return tail_map


def _read_captions(caption_file: str) -> Dict[str, str]:
    """Read caption CSV into a map: normalized_image_rel_path -> caption string.

    We normalize path separators to '/' and keep as given in CSV (usually prefixed with 'data/').
    """
    cap_map: Dict[str, str] = {}
    with open(caption_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Accept flexible header names
        img_key = None
        txt_key = None
        lower = [c.lower() for c in reader.fieldnames or []]
        for cand in ("image path", "image_path", "path", "filename"):
            if cand in lower:
                img_key = reader.fieldnames[lower.index(cand)]
                break
        for cand in ("caption", "text", "description"):
            if cand in lower:
                txt_key = reader.fieldnames[lower.index(cand)]
                break
        if img_key is None or txt_key is None:
            raise ValueError(f"Caption file missing required columns (have: {reader.fieldnames})")

        for row in reader:
            raw_path = str(row[img_key]).strip()
            if not raw_path:
                continue
            img_rel = _normalize_path_key(raw_path)
            if not img_rel:
                continue
            cap = str(row[txt_key]).strip()
            if not cap:
                continue
            # Store multiple keys for the same caption to improve matching robustness
            cap_map[img_rel] = cap
            stripped = _strip_leading_data(img_rel)
            if stripped != img_rel:
                cap_map.setdefault(stripped, cap)
    return cap_map


def _rel_from_root(path: str, img_root: str) -> str:
    """Return a path relative to img_root with forward slashes."""
    try:
        rel = os.path.relpath(path, img_root)
    except Exception:
        # If relpath fails, fallback to original
        rel = path
    return rel.replace('\\', '/')


def _match_items_with_captions(items, img_root: str, cap_map: Dict[str, str], dataset_name: str) -> List[Dict]:
    """Match dataset items to captions by relative path.

    Each item should have attributes: impath, label. We try several keys to
    look up the caption robustly: exact relative path, prefixed with 'data/'.
    """
    pairs: List[Dict] = []
    missing = 0
    tail_lookup = _build_tail_lookup(cap_map)
    tail_hits = 0
    for idx, it in enumerate(items):
        impath = getattr(it, 'impath', None) or getattr(it, '_impath', None)
        label = int(getattr(it, 'label', getattr(it, '_label', -1)))
        if not impath:
            continue

        # Normalize to relative path
        rel = _rel_from_root(impath, img_root)
        rel_norm = rel.replace('\\', '/')

        # Candidate keys to try in caption map
        keys = [rel_norm, f"data/{rel_norm}"]
        # When ROOT points to data/<DATASET>, captions may contain data/<DATASET>/<rel>
        if dataset_name:
            keys.append(f"{dataset_name}/{rel_norm}")
            keys.append(f"data/{dataset_name}/{rel_norm}")

        txt = None
        for k in keys:
            if k in cap_map:
                txt = cap_map[k]
                break

        if txt is None:
            tail_key = _extract_split_tail(rel_norm)
            if tail_key in tail_lookup:
                txt = tail_lookup[tail_key]
                tail_hits += 1

        if txt is None:
            missing += 1
            continue  # skip items without captions

        pairs.append({
            "image_path": rel_norm,
            "text": txt,
            "label": label,
            "id": f"{idx}"
        })

    if tail_hits:
        print(f"Info: matched {tail_hits} items via split-tail fallback")
    if missing:
        print(f"Warning: {missing} items had no matching caption and were skipped")
    return pairs


def _write_jsonl(items: List[Dict], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Wrote {len(items)} pairs -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name matching a module in datasets/ (e.g., Emotion6, Emoset, FI, Twitter1, Twitter2)')
    parser.add_argument('--img_root', type=str, default='data', help='Root directory for images')
    parser.add_argument('--caption_file', type=str, required=True, help='CSV with Image Path, Caption columns')
    parser.add_argument('--out_dir', type=str, default='cache', help='Directory to save JSONL pairs')
    parser.add_argument('--out_prefix', type=str, default='pairs', help='Output filename prefix')
    parser.add_argument('--split_strategy', type=str, default='auto', choices=['auto', 'predefined', 'stratified', 'random'],
                        help='Force a split strategy if the adapter supports it; auto lets the adapter decide')

    args = parser.parse_args()

    img_root = os.path.abspath(args.img_root)
    ds_cls = _import_dataset_class(args.dataset)

    # Build a minimal Dassl cfg for the dataset
    cfg = get_cfg_default()
    cfg.DATASET.ROOT = img_root
    cfg.DATASET.NAME = args.dataset
    # Sanitize seed: numpy requires non-negative seed
    if getattr(cfg, 'SEED', -1) is None or cfg.SEED < 0:
        cfg.SEED = 42
    # Some adapters read SPLIT_SEED from DATASET; keep consistent
    cfg.DATASET.SPLIT_SEED = cfg.SEED
    if args.split_strategy != 'auto':
        cfg.DATASET.SPLIT_STRATEGY = args.split_strategy

    # Instantiate the dataset adapter to get items; try fallback root if needed
    try:
        ds = ds_cls(cfg)
        effective_root = img_root
    except FileNotFoundError as e:
        alt_root = os.path.join(img_root, args.dataset)
        print(f"Primary ROOT failed ({e}); retrying with ROOT={alt_root}")
        cfg2 = get_cfg_default()
        cfg2.DATASET.ROOT = alt_root
        cfg2.DATASET.NAME = args.dataset
        if getattr(cfg2, 'SEED', -1) is None or cfg2.SEED < 0:
            cfg2.SEED = 42
        cfg2.DATASET.SPLIT_SEED = cfg2.SEED
        if args.split_strategy != 'auto':
            cfg2.DATASET.SPLIT_STRATEGY = args.split_strategy
        ds = ds_cls(cfg2)
        effective_root = alt_root
    train_items = getattr(ds, 'train_x', [])
    val_items = getattr(ds, 'val', [])
    test_items = getattr(ds, 'test', [])

    # Prefer val if present; otherwise use test as validation
    val_like = val_items if val_items else test_items

    # Read captions
    cap_map = _read_captions(args.caption_file)

    # Match and write pairs
    train_pairs = _match_items_with_captions(train_items, effective_root, cap_map, args.dataset)
    val_pairs = _match_items_with_captions(val_like, effective_root, cap_map, args.dataset)

    dataset_tag = Path(args.caption_file).stem  # keep for clarity
    train_out = os.path.join(args.out_dir, f"{args.out_prefix}_{args.dataset}_{dataset_tag}_train.jsonl")
    val_out = os.path.join(args.out_dir, f"{args.out_prefix}_{args.dataset}_{dataset_tag}_val.jsonl")

    _write_jsonl(train_pairs, train_out)
    _write_jsonl(val_pairs, val_out)

    # Quick stats
    def _stats(pairs: List[Dict]) -> Dict:
        hist: Dict[int, int] = {}
        for it in pairs:
            hist[it['label']] = hist.get(it['label'], 0) + 1
        return hist

    print("\nStats:")
    print(f"  Train: {len(train_pairs)} items | label dist: {_stats(train_pairs)}")
    print(f"  Val:   {len(val_pairs)} items | label dist: {_stats(val_pairs)}")
    print(f"Effective image root: {effective_root}")


if __name__ == '__main__':
    main()
