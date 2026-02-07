"""
Image-Text Pairs Dataset for Multimodal Training

This module provides dataset classes for loading image-text pairs
with strict one-to-one correspondence for multimodal training.
"""

import os
import json
import pickle
import random
from typing import Dict, List, Union, Optional
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer

# Define primary emotions for emotion datasets
# PRIMARY_EMOTIONS = ["neg", "pos"]
PRIMARY_EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
# PRIMARY_EMOTIONS = ["anger", "disgust", "fear", "awe", "amusement", "contentment", "sadness", "excitement"]
CLASSNAME_TO_LABEL = {name: i for i, name in enumerate(PRIMARY_EMOTIONS)}
LABEL_TO_CLASSNAME = {i: name for i, name in enumerate(PRIMARY_EMOTIONS)}


class ImageTextPairs(Dataset):
    """
    Dataset for loading image-text pairs with one-to-one correspondence.

    Returns (image, input_ids, attention_mask, label, img_path, txt_raw)
    """

    def __init__(
        self,
        pairs_file: str,
        img_root: str,
        tokenizer: AutoTokenizer,
        transform: Optional[transforms.Compose] = None,
        max_length: int = 128
    ):
        """
        Args:
            pairs_file: Path to JSONL file containing image-text pairs
            img_root: Root directory for images
            tokenizer: BERT tokenizer for text processing
            transform: Image transformations
            max_length: Maximum token length for text
        """
        self.pairs_file = pairs_file
        self.img_root = img_root
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Default image transformations
        if transform is None:
            # Default to CLIP-style preprocessing to match the CLIP visual encoder
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)
                )
            ])
        else:
            self.transform = transform

        # Load pairs
        self.items = self._load_pairs()

        print(f"Loaded {len(self.items)} image-text pairs from {pairs_file}")
        if len(self.items) > 0:
            self._validate_pairs()

    def _load_pairs(self) -> List[Dict]:
        """Load pairs from JSONL or PKL file."""
        if self.pairs_file.endswith('.jsonl'):
            return self._load_jsonl()
        elif self.pairs_file.endswith('.pkl'):
            return self._load_pkl()
        else:
            raise ValueError(f"Unsupported file format: {self.pairs_file}")

    def _load_jsonl(self) -> List[Dict]:
        """Load pairs from JSONL file."""
        items = []
        with open(self.pairs_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    if self._validate_item(item):
                        items.append(item)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON at line {line_num}: {e}")
                    continue
        return items

    def _load_pkl(self) -> List[Dict]:
        """Load pairs from PKL file."""
        with open(self.pairs_file, 'rb') as f:
            return pickle.load(f)

    def _validate_item(self, item: Dict) -> bool:
        """Validate individual item has required fields."""
        required_fields = ["image_path", "text", "label"]
        for field in required_fields:
            if field not in item:
                print(f"Warning: Missing required field '{field}' in item")
                return False

        # Check if image exists
        img_path = os.path.join(self.img_root, item["image_path"])
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            return False

        return True

    def _validate_pairs(self):
        """Validate data consistency."""
        if len(self.items) == 0:
            raise ValueError("No valid pairs loaded")

        # Print first few examples for verification
        print(f"\nFirst 3 examples for verification:")
        for i in range(min(3, len(self.items))):
            item = self.items[i]
            print(f"  {i+1}. Image: {item['image_path']}")
            print(f"     Text: {item['text'][:60]}...")
            print(f"     Label: {item['label']} ({LABEL_TO_CLASSNAME.get(item['label'], 'unknown')})")
            print()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            dict containing:
            - image: Tensor of shape (3, 224, 224)
            - input_ids: Tokenized text input IDs
            - attention_mask: Attention mask for text
            - label: Integer class label
            - img_path: Original image path
            - text_raw: Original text string
        """
        item = self.items[idx]

        # Load and transform image
        img_path = os.path.join(self.img_root, item["image_path"])
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            image = torch.zeros(3, 224, 224)

        # Tokenize text
        text = str(item["text"]).strip()
        if not text:
            text = "no caption available"  # Fallback for empty text

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            "image": image,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(item["label"], dtype=torch.long),
            "img_path": item["image_path"],
            "text_raw": text
        }


def create_sample_pairs(
    caption_csv: str,
    img_root: str,
    output_file: str,
    split: str = "train",
    max_samples: Optional[int] = None
):
    """
    Create sample pairs from caption CSV for testing.

    Args:
        caption_csv: Path to caption CSV file
        img_root: Root directory for images
        output_file: Output JSONL file path
        split: Dataset split (train/val/test)
        max_samples: Maximum number of samples to create
    """
    import pandas as pd

    # Read caption CSV
    df = pd.read_csv(caption_csv)
    print(f"Loaded {len(df)} captions from {caption_csv}")

    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    pairs = []

    for idx, row in df.iterrows():
        if max_samples and len(pairs) >= max_samples:
            break

        img_path = row["Image Path"]
        caption = str(row["Caption"]).strip()

        # Skip empty captions
        if not caption or caption.lower() == "nan":
            continue

        # Extract label from path for emotion datasets
        label = _extract_label_from_path(img_path)
        if label is None:
            continue

        # Check if image exists
        full_img_path = os.path.join(img_root, img_path)
        if not os.path.exists(full_img_path):
            continue

        pair = {
            "image_path": img_path,
            "text": caption,
            "label": label,
            "id": f"{split}_{idx}"
        }

        pairs.append(pair)

    # Save to JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    print(f"Created {len(pairs)} pairs and saved to {output_file}")
    return pairs


def _extract_label_from_path(img_path: str) -> Optional[int]:
    """
    Extract emotion label from image path.

    Args:
        img_path: Image path like "data/Emotion6/anger/xxx.jpg"

    Returns:
        Integer label or None if not found
    """
    # Extract folder name from path
    parts = Path(img_path).parts
    for part in parts:
        if part.lower() in PRIMARY_EMOTIONS:
            return CLASSNAME_TO_LABEL[part.lower()]

    # If no emotion folder found, try to infer from parent folder name
    parent_name = Path(img_path).parent.name.lower()
    if parent_name in PRIMARY_EMOTIONS:
        return CLASSNAME_TO_LABEL[parent_name]

    return None


def collate_fn(batch):
    """
    Collate function for DataLoader to ensure consistent batch formatting.
    """
    images = torch.stack([item["image"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_masks = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])

    return {
        "image": images,
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "label": labels,
        "img_path": [item["img_path"] for item in batch],
        "text_raw": [item["text_raw"] for item in batch]
    }


if __name__ == "__main__":
    # Example usage
    print("Testing ImageTextPairs dataset...")

    # Create sample pairs for testing
    caption_csv = "caption/narracap_extended_Emotion6_truncated.csv"
    img_root = "data"
    output_file = "cache/pairs_test.jsonl"

    if os.path.exists(caption_csv):
        pairs = create_sample_pairs(
            caption_csv=caption_csv,
            img_root=img_root,
            output_file=output_file,
            split="test",
            max_samples=10
        )

        # Test dataset loading
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        dataset = ImageTextPairs(
            pairs_file=output_file,
            img_root=img_root,
            tokenizer=tokenizer
        )

        print(f"Dataset created with {len(dataset)} samples")

        # Test loading a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Image shape: {sample['image'].shape}")
            print(f"Input IDs shape: {sample['input_ids'].shape}")
            print(f"Label: {sample['label']}")
    else:
        print(f"Caption file {caption_csv} not found")
