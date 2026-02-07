#!/usr/bin/env python3
"""
Simplified BERT Text Classification Training Script
Designed for local CSV data classification (e.g., emotion classification)
"""

import argparse
import json
import logging
import os

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_scheduler

ADAPTER_INITIALIZER = None
try:
    from transformers.adapters import AdapterConfig  # type: ignore
except Exception:
    try:
        # Newer AdapterHub releases expose the config via the standalone `adapters` package.
        from adapters import AdapterConfig  # type: ignore
        from adapters import init as adapters_init  # type: ignore

        ADAPTER_INITIALIZER = adapters_init
    except Exception:
        AdapterConfig = None
        ADAPTER_INITIALIZER = None

DEFAULT_ADAPTER_NAME = "task_adapter"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextClassificationDataset(Dataset):
    def __init__(self, tokenizer, texts, labels, max_length=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_csv_data(file_path, text_column='text', label_column='label'):
    """Load CSV data and extract text and labels"""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        # Assume JSON
        df = pd.read_json(file_path, lines=True)

    texts = df[text_column].astype(str).tolist()
    labels = df[label_column].tolist()

    # Create label mapping if labels are strings
    if isinstance(labels[0], str):
        unique_labels = sorted(list(set(labels)))
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        id_to_label = {idx: label for idx, label in enumerate(unique_labels)}
        labels = [label_to_id[label] for label in labels]
        num_labels = len(unique_labels)
    else:
        # Labels are already numeric
        unique_labels = sorted(list(set(labels)))
        label_to_id = None
        id_to_label = None
        num_labels = max(unique_labels) + 1

    return texts, labels, num_labels, label_to_id, id_to_label


def parse_args():
    parser = argparse.ArgumentParser(description="Train BERT on local text classification data")

    # Required arguments
    parser.add_argument("--train_file", type=str, required=True, help="Path to training CSV/JSON file")
    parser.add_argument("--validation_file", type=str, required=True, help="Path to validation CSV/JSON file")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Pretrained model name or path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model")

    # Data arguments
    parser.add_argument("--text_column", type=str, default="text", help="Column name for text data")
    parser.add_argument("--label_column", type=str, default="label", help="Column name for label data")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")

    # Training arguments
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", help="Evaluation strategy")
    parser.add_argument("--save_strategy", type=str, default="epoch", help="Save strategy")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Adapter options
    parser.add_argument("--use_adapter", action="store_true", help="Enable adapter-based fine-tuning")
    parser.add_argument("--adapter_type", type=str, default="pfeiffer", choices=["pfeiffer", "houlsby"],
                        help="Adapter structure (pfeiffer=FFN-only, houlsby=attn+FFN)")
    parser.add_argument("--adapter_reduction_factor", type=int, default=16, help="Adapter bottleneck size factor")
    parser.add_argument("--adapter_dropout", type=float, default=0.0, help="Adapter dropout probability")
    parser.add_argument("--lr_adapter", type=float, default=1e-4, help="Learning rate for adapter parameters")
    parser.add_argument("--lr_head", type=float, default=1e-4, help="Learning rate for classifier head parameters")

    # Freeze backbone option
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze all backbone parameters (embeddings, encoder layers), only train classifier head")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Load training data
    logger.info(f"Loading training data from {args.train_file}")
    train_texts, train_labels, num_labels, label_to_id, id_to_label = load_csv_data(
        args.train_file, args.text_column, args.label_column
    )
    logger.info(f"Training data: {len(train_texts)} samples, {num_labels} classes")

    # Load validation data
    logger.info(f"Loading validation data from {args.validation_file}")
    val_texts, val_labels, _, _, _ = load_csv_data(
        args.validation_file, args.text_column, args.label_column
    )
    logger.info(f"Validation data: {len(val_texts)} samples")

    # Create datasets
    train_dataset = TextClassificationDataset(tokenizer, train_texts, train_labels, args.max_length)
    val_dataset = TextClassificationDataset(tokenizer, val_texts, val_labels, args.max_length)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False)

    # Load model
    logger.info(f"Loading model: {args.model_name_or_path}")
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True
    )

    if args.use_adapter and ADAPTER_INITIALIZER is not None:
        # Inject adapter mixins when running with the standalone AdapterHub package.
        ADAPTER_INITIALIZER(model)

    model = model.to(device)

    # Setup optimizer and scheduler
    num_training_steps = len(train_loader) * args.num_train_epochs
    adapter_name = DEFAULT_ADAPTER_NAME
    optimizer = None

    if args.freeze_backbone:
        # ========== 线性探测模式：只训练分类器 ==========
        logger.info("=" * 60)
        logger.info("FREEZE BACKBONE MODE: Training ONLY classifier head")
        logger.info("=" * 60)

        head_params = []
        frozen_count = 0
        trainable_count = 0

        for name, param in model.named_parameters():
            if "classifier" in name or "score" in name:
                param.requires_grad = True
                head_params.append(param)
                trainable_count += param.numel()
                logger.info(f"  ✅ Trainable: {name} ({param.numel():,} params)")
            else:
                param.requires_grad = False
                frozen_count += param.numel()

        total_params = frozen_count + trainable_count
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable: {trainable_count:,} ({100*trainable_count/total_params:.2f}%)")
        logger.info(f"Frozen: {frozen_count:,} ({100*frozen_count/total_params:.2f}%)")
        logger.info(f"Classifier learning rate: {args.lr_head}")
        logger.info("=" * 60)

        if not head_params:
            raise RuntimeError("No classifier parameters found in model!")

        # 使用 lr_head 作为分类器学习率 (默认 1e-4)
        optimizer = optim.AdamW(head_params, lr=args.lr_head, weight_decay=args.weight_decay)

    elif args.use_adapter:
        if AdapterConfig is None:
            raise RuntimeError(
                "Adapter support is required for adapter training. Please install the AdapterHub "
                "packages (e.g., `pip install adapter-transformers` or `pip install adapters`)."
            )

        logger.info(f"Enabling adapter training ({args.adapter_type}, reduction={args.adapter_reduction_factor}, dropout={args.adapter_dropout})")
        adapter_config = AdapterConfig.load(
            args.adapter_type,
            reduction_factor=args.adapter_reduction_factor,
            dropout=args.adapter_dropout
        )

        existing_adapters = getattr(getattr(model.config, "adapters", None), "adapters", {})
        if existing_adapters and adapter_name in existing_adapters:
            model.delete_adapter(adapter_name)

        model.add_adapter(adapter_name, config=adapter_config)
        model.set_active_adapters(adapter_name)
        model.train_adapter(adapter_name)
        model = model.to(device)

        adapter_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "adapters" in name:
                adapter_params.append(param)
            elif "classifier" in name:
                head_params.append(param)
            else:
                param.requires_grad = False

        optimizer_groups = []
        if adapter_params:
            optimizer_groups.append({"params": adapter_params, "lr": args.lr_adapter})
        if head_params:
            optimizer_groups.append({"params": head_params, "lr": args.lr_head})

        if not optimizer_groups:
            raise RuntimeError("No parameters left to optimize when using adapters.")

        optimizer = optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Training loop
    logger.info("Starting training...")
    best_val_acc = 0.0

    for epoch in range(args.num_train_epochs):
        # Training
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")

        # Validation
        if args.evaluation_strategy == "epoch":
            model.eval()
            total_val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    loss = outputs.loss
                    logits = outputs.logits

                    total_val_loss += loss.item()
                    predictions = torch.argmax(logits, dim=-1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

            avg_val_loss = total_val_loss / len(val_loader)
            val_acc = correct / total
            logger.info(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                logger.info(f"New best validation accuracy: {best_val_acc:.4f}")
                model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)

                # Save label mappings
                if label_to_id:
                    with open(os.path.join(args.output_dir, 'label_mappings.json'), 'w') as f:
                        json.dump({
                            'label_to_id': label_to_id,
                            'id_to_label': id_to_label,
                            'num_labels': num_labels
                        }, f, indent=2)

    logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
