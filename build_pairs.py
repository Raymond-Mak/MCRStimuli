#!/usr/bin/env python3
"""
Build Image-Text Pairs from Caption CSVs

Utility script to convert caption CSV files and image directories
into paired datasets for multimodal training.

Usage:
    python build_pairs.py --img_root data --caption_dir caption/ --out cache/pairs.jsonl
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.pairs_dataset import PRIMARY_EMOTIONS, CLASSNAME_TO_LABEL, LABEL_TO_CLASSNAME


# Datasets that have predefined train/val/test splits
STRUCTURED_DATASETS = {
    "Emoset",
    "FI_new",
    "FI_Probing"
}


def is_structured_dataset(caption_file: str, img_root: str) -> bool:
    """
    Check if the dataset has predefined train/val/test directory structure.

    Args:
        caption_file: Path to caption file
        img_root: Root directory for images

    Returns:
        True if dataset has predefined structure, False otherwise
    """
    # Extract dataset name from caption file
    dataset_name = Path(caption_file).stem.replace("narracap_extended_", "")

    # Check if this is a known structured dataset
    if dataset_name in STRUCTURED_DATASETS:
        # Verify the directory structure exists
        dataset_path = Path(img_root) / dataset_name
        if dataset_path.exists():
            splits = ["train", "val", "test"]
            return all((dataset_path / split).exists() for split in splits)

    return False


def get_dataset_split_from_path(img_path: str) -> Optional[str]:
    """
    Extract dataset split (train/val/test) from image path.

    Args:
        img_path: Image path containing split information

    Returns:
        Split name (train/val/test) or None
    """
    path_parts = Path(img_path).parts
    for part in path_parts:
        if part.lower() in ["train", "val", "test"]:
            return part.lower()
    return None


def extract_label_from_image_path(img_path: str) -> Optional[int]:
    """
    Extract emotion label from image path.

    Args:
        img_path: Image path like "data/Emotion6/anger/xxx.jpg"

    Returns:
        Integer label or None if emotion not found
    """
    path_parts = Path(img_path).parts

    # Look for emotion in path components
    for part in path_parts:
        part_lower = part.lower()
        if part_lower in PRIMARY_EMOTIONS:
            return CLASSNAME_TO_LABEL[part_lower]

    # Alternative: check parent directory name
    parent_name = Path(img_path).parent.name.lower()
    if parent_name in PRIMARY_EMOTIONS:
        return CLASSNAME_TO_LABEL[parent_name]

    return None


def load_caption_file(caption_file: str) -> pd.DataFrame:
    """
    Load caption CSV/JSON file.

    Args:
        caption_file: Path to caption file

    Returns:
        DataFrame with columns: Image Path, Caption
    """
    if caption_file.endswith('.csv'):
        df = pd.read_csv(caption_file)
    elif caption_file.endswith('.json'):
        df = pd.read_json(caption_file, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {caption_file}")

    # Standardize column names
    column_mapping = {
        'image_path': 'Image Path',
        'path': 'Image Path',
        'filename': 'Image Path',
        'caption': 'Caption',
        'text': 'Caption',
        'description': 'Caption'
    }

    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})

    # Validate required columns
    required_columns = ['Image Path', 'Caption']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df


def validate_image_exists(img_path: str, img_root: str) -> bool:
    """Check if image file exists."""
    # Handle case where img_path already includes img_root
    if img_path.startswith(img_root):
        full_path = img_path
    else:
        full_path = os.path.join(img_root, img_path)
    return os.path.exists(full_path)


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if pd.isna(text) or text is None:
        return ""

    text = str(text).strip()
    # Remove extra quotes if present
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    elif text.startswith("'") and text.endswith("'"):
        text = text[1:-1].strip()

    return text


def build_pairs_from_structured_dataset(
    caption_file: str,
    img_root: str,
    output_file: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    validate_images: bool = True
) -> Tuple[List[Dict], List[Dict]]:
    """
    Build image-text pairs from datasets with predefined train/val/test splits.

    Args:
        caption_file: Path to caption CSV/JSON file
        img_root: Root directory for images
        output_file: Output JSONL file path
        split: Dataset split to process (train/val/test)
        max_samples: Maximum number of samples to process
        validate_images: Whether to check image existence

    Returns:
        Tuple of (split_pairs, empty_list) - second list empty for compatibility
    """
    print(f"Loading captions from {caption_file} for structured dataset...")
    df = load_caption_file(caption_file)
    print(f"Loaded {len(df)} caption entries")

    # Filter entries based on requested split
    filtered_df = df[df['Image Path'].str.contains(f'/{split}/', case=False, na=False)]
    print(f"Filtered to {len(filtered_df)} entries for '{split}' split")

    # Process entries
    pairs = []
    processed = 0
    skipped_no_label = 0
    skipped_no_image = 0
    skipped_empty_text = 0

    for idx, (_, row) in enumerate(filtered_df.iterrows()):
        if max_samples and processed >= max_samples:
            break

        img_path = row['Image Path']
        raw_caption = row['Caption']

        # Clean text
        caption = clean_text(raw_caption)
        if not caption:
            skipped_empty_text += 1
            continue

        # Extract label from path
        label = extract_label_from_image_path(img_path)
        if label is None:
            skipped_no_label += 1
            continue

        # Validate image exists
        if validate_images and not validate_image_exists(img_path, img_root):
            skipped_no_image += 1
            continue

        pair = {
            "image_path": img_path,
            "text": caption,
            "label": label,
            "id": f"{split}_{idx}"
        }

        pairs.append(pair)
        processed += 1

        if processed % 1000 == 0:
            print(f"Processed {processed} pairs for {split} split...")

    print(f"\nProcessing complete for {split} split:")
    print(f"  Total processed: {processed}")
    print(f"  Skipped (no label): {skipped_no_label}")
    print(f"  Skipped (no image): {skipped_no_image}")
    print(f"  Skipped (empty text): {skipped_empty_text}")
    print(f"  Final pairs: {len(pairs)}")

    return pairs, []  # Return empty list for compatibility with existing interface


def build_pairs_from_captions(
    caption_file: str,
    img_root: str,
    output_file: str,
    split: str = "train",
    split_ratio: float = 0.8,
    seed: int = 42,
    max_samples: Optional[int] = None,
    validate_images: bool = True
) -> Tuple[List[Dict], List[Dict]]:
    """
    Build image-text pairs from caption file.

    Args:
        caption_file: Path to caption CSV/JSON file
        img_root: Root directory for images
        output_file: Output JSONL file path
        split: Dataset split (train/val/test)
        split_ratio: Ratio for train/val split (if generating both)
        seed: Random seed for reproducibility
        max_samples: Maximum number of samples to process
        validate_images: Whether to check image existence

    Returns:
        Tuple of (train_pairs, val_pairs)
    """
    print(f"Loading captions from {caption_file}...")
    df = load_caption_file(caption_file)
    print(f"Loaded {len(df)} caption entries")

    # Set random seed
    random.seed(seed)

    # Process entries
    pairs = []
    processed = 0
    skipped_no_label = 0
    skipped_no_image = 0
    skipped_empty_text = 0

    for idx, row in df.iterrows():
        if max_samples and processed >= max_samples:
            break

        img_path = row['Image Path']
        raw_caption = row['Caption']

        # Clean text
        caption = clean_text(raw_caption)
        if not caption:
            skipped_empty_text += 1
            continue

        # Extract label from path
        label = extract_label_from_image_path(img_path)
        if label is None:
            skipped_no_label += 1
            continue

        # Validate image exists
        if validate_images and not validate_image_exists(img_path, img_root):
            skipped_no_image += 1
            continue

        pair = {
            "image_path": img_path,
            "text": caption,
            "label": label,
            "id": f"{split}_{idx}"
        }

        pairs.append(pair)
        processed += 1

        if processed % 1000 == 0:
            print(f"Processed {processed} pairs...")

    print(f"\nProcessing complete:")
    print(f"  Total processed: {processed}")
    print(f"  Skipped (no label): {skipped_no_label}")
    print(f"  Skipped (no image): {skipped_no_image}")
    print(f"  Skipped (empty text): {skipped_empty_text}")
    print(f"  Final pairs: {len(pairs)}")

    # Split into train/val if needed
    if split == "train" and split_ratio < 1.0:
        random.shuffle(pairs)
        split_idx = int(len(pairs) * split_ratio)
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]
    else:
        train_pairs = pairs
        val_pairs = []

    return train_pairs, val_pairs


def save_pairs(pairs: List[Dict], output_file: str):
    """Save pairs to JSONL file."""
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    print(f"Saved {len(pairs)} pairs to {output_file}")


def generate_statistics(pairs: List[Dict]) -> Dict:
    """Generate statistics for the pairs dataset."""
    if not pairs:
        return {}

    # Label distribution
    label_counts = {}
    for pair in pairs:
        label = pair['label']
        label_counts[label] = label_counts.get(label, 0) + 1

    # Text length statistics
    text_lengths = [len(pair['text'].split()) for pair in pairs]

    stats = {
        "total_pairs": len(pairs),
        "label_distribution": {
            LABEL_TO_CLASSNAME.get(label, f"class_{label}"): count
            for label, count in sorted(label_counts.items())
        },
        "text_length_stats": {
            "mean": sum(text_lengths) / len(text_lengths),
            "min": min(text_lengths),
            "max": max(text_lengths)
        }
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Build image-text pairs from caption files")

    # Input arguments
    parser.add_argument("--img_root", type=str, required=True,
                        help="Root directory for images")
    parser.add_argument("--caption_file", type=str, required=True,
                        help="Path to caption CSV/JSON file")
    parser.add_argument("--caption_dir", type=str, default="caption/",
                        help="Directory containing caption files (alternative to caption_file)")

    # Output arguments
    parser.add_argument("--out_dir", type=str, default="cache/",
                        help="Output directory for pairs files")
    parser.add_argument("--out_prefix", type=str, default="pairs",
                        help="Prefix for output files")

    # Processing arguments
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test"],
                        help="Dataset split")
    parser.add_argument("--split_ratio", type=float, default=0.8,
                        help="Train/val split ratio (only for split='train')")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--no_validate_images", action="store_true",
                        help="Skip image existence validation")

    args = parser.parse_args()

    # Determine caption file
    if args.caption_file:
        caption_files = [args.caption_file]
    else:
        # Find all CSV files in caption directory
        caption_dir = Path(args.caption_dir)
        caption_files = list(caption_dir.glob("*.csv"))
        if not caption_files:
            print(f"No CSV files found in {args.caption_dir}")
            return

    # Process each caption file
    for caption_file in caption_files:
        print(f"\nProcessing {caption_file}...")

        # Generate output filenames
        dataset_name = Path(caption_file).stem
        train_output = os.path.join(args.out_dir, f"{args.out_prefix}_{dataset_name}_train.jsonl")
        val_output = os.path.join(args.out_dir, f"{args.out_prefix}_{dataset_name}_val.jsonl")
        test_output = os.path.join(args.out_dir, f"{args.out_prefix}_{dataset_name}_{args.split}.jsonl")

        try:
            # Check if this is a structured dataset
            is_structured = is_structured_dataset(str(caption_file), args.img_root)

            if is_structured:
                print(f"Processing {dataset_name} as structured dataset with predefined splits")

                # For structured datasets, process each split separately
                splits_to_process = []
                if args.split == "train":
                    # For train split, generate train.jsonl from /train and val.jsonl from /test
                    splits_to_process = [("train", train_output), ("test", val_output)]
                else:
                    splits_to_process = [(args.split, test_output)]

                all_pairs = []

                for split_name, output_file in splits_to_process:
                    print(f"\nProcessing {split_name} split...")
                    split_pairs, _ = build_pairs_from_structured_dataset(
                        caption_file=str(caption_file),
                        img_root=args.img_root,
                        output_file=output_file,
                        split=split_name,
                        max_samples=args.max_samples,
                        validate_images=not args.no_validate_images
                    )

                    # Save the pairs for this split
                    if split_pairs:
                        save_pairs(split_pairs, output_file)
                        all_pairs.extend(split_pairs)

            else:
                print(f"Processing {dataset_name} as unstructured dataset (using random split)")

                # Original logic for unstructured datasets
                train_pairs, val_pairs = build_pairs_from_captions(
                    caption_file=str(caption_file),
                    img_root=args.img_root,
                    output_file="",  # We'll handle saving manually
                    split=args.split,
                    split_ratio=args.split_ratio,
                    seed=args.seed,
                    max_samples=args.max_samples,
                    validate_images=not args.no_validate_images
                )

                # Save based on split type
                if args.split == "train":
                    save_pairs(train_pairs, train_output)
                    if val_pairs:
                        save_pairs(val_pairs, val_output)
                else:
                    save_pairs(train_pairs, test_output)

                # Collect all pairs for statistics
                all_pairs = train_pairs + val_pairs

            # Generate and save statistics
            if all_pairs:
                stats = generate_statistics(all_pairs)

                stats_file = os.path.join(args.out_dir, f"{args.out_prefix}_{dataset_name}_stats.json")
                with open(stats_file, 'w') as f:
                    json.dump(stats, f, indent=2)

                print(f"\nStatistics for {dataset_name}:")
                print(f"  Total pairs: {stats.get('total_pairs', 0)}")
                print(f"  Label distribution: {stats.get('label_distribution', {})}")
            else:
                print(f"No pairs generated for {dataset_name}")

        except Exception as e:
            print(f"Error processing {caption_file}: {e}")
            continue

    print(f"\nProcessing complete! Check {args.out_dir} for output files.")


if __name__ == "__main__":
    main()