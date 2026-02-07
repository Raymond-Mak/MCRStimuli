"""
Multimodal Trainer for Image-Text Classification

This module provides training logic for the multimodal model with
proper parameter group management, metrics tracking, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import os
import json
import sys
import time
import math
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.mm_model import MultiModalModel
from utils.pairs_dataset import collate_fn


class MMTrainer:
    """
    Multimodal trainer for image-text classification.

    Handles training loop, optimization, validation, and model checkpointing.
    """

    def __init__(
        self,
        model: MultiModalModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        # Dataset configuration
        dataset_name: str = "Emotion6",
        # Optimization parameters
        vit_lr: float = 3e-5,
        bert_lr: float = 1e-5,
        head_lr: float = 1e-3,
        lr_adapter: Optional[float] = None,
        weight_decay: float = 0.01,
        # Training parameters
        epochs: int = 20,
        device: Optional[str] = None,
        mixed_precision: bool = True,
        gradient_clip_val: float = 1.0,
        # Checkpointing
        save_dir: str = "checkpoints/mm",
        save_freq: int = 5,  # Save every N epochs (now only affects logging)
        # Logging
        print_freq: int = 50,
        log_wandb: bool = False,
        project_name: str = "multimodal-training",
        # Monitoring
        monitor_modal_losses: bool = False,
        modal_loss_freq: Optional[int] = None
    ):
        """
        Args:
            model: Multimodal model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            dataset_name: Dataset name for binary classification mapping (Emotion6, FI_new, Emoset, etc.)
            vit_lr: Learning rate for ViT encoder
            bert_lr: Learning rate for BERT encoder
            head_lr: Learning rate for fusion head
            lr_adapter: Optional learning rate override for adapter parameters (defaults to bert_lr)
            weight_decay: Weight decay for optimizer
            epochs: Number of training epochs
            device: Device to use (auto-detected if None)
            mixed_precision: Whether to use mixed precision training
            gradient_clip_val: Gradient clipping value
            save_dir: Directory to save checkpoints (now default to checkpoints/mm)
            save_freq: Save checkpoint every N epochs (now only affects logging frequency)
            print_freq: Print metrics every N steps
            log_wandb: Whether to log to Weights & Biases
            project_name: WandB project name
            monitor_modal_losses: Whether to compute per-modality monitoring losses
            modal_loss_freq: Frequency (in steps) for monitoring loss computation (default: print_freq)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.dataset_name = dataset_name

        # Training parameters
        self.epochs = epochs
        self.vit_lr = vit_lr
        self.bert_lr = bert_lr
        self.head_lr = head_lr
        self.lr_adapter = lr_adapter
        self.weight_decay = weight_decay
        self.print_freq = print_freq
        self.save_freq = save_freq

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Move model to device
        self.model.to(self.device)

        # Setup optimizer with parameter groups
        self.optimizer = self._setup_optimizer()

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Mixed precision
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler() if mixed_precision else None

        # Gradient clipping
        self.gradient_clip_val = gradient_clip_val

        # Checkpointing
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.log_wandb = log_wandb
        self.monitor_modal_losses = monitor_modal_losses
        self.modal_loss_freq = modal_loss_freq or print_freq
        if log_wandb:
            try:
                import wandb
                wandb.init(
                    project=project_name,
                    config={
                        "vit_lr": vit_lr,
                        "bert_lr": bert_lr,
                        "head_lr": head_lr,
                        "weight_decay": weight_decay,
                        "epochs": epochs,
                        "mixed_precision": mixed_precision,
                        "device": str(self.device),
                        "save_dir": str(self.save_dir)
                    }
                )
                self.wandb = wandb
                print(f"WandB logging enabled for project: {project_name}")
            except ImportError:
                print("Warning: wandb not installed, skipping wandb logging")
                self.log_wandb = False

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_accuracies = []

        print(f"=== MULTIMODAL TRAINER INITIALIZED ===")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.mixed_precision}")
        print(f"Gradient clipping: {self.gradient_clip_val}")
        print(f"Checkpoints directory: {self.save_dir}")
        print(f"Print frequency: {self.print_freq}")
        print(f"Save strategy: Best model only (to save disk space)")
        print("="*60)

    def _get_binary_mapping(self, label: int) -> int:
        """
        将原始标签映射为二分类标签（0=neg, 1=pos）

        Args:
            label: 原始类别标签

        Returns:
            二分类标签（0=neg, 1=pos）
        """
        from utils.sentiment_mapping import get_binary_mapping
        return get_binary_mapping(self.dataset_name, label)

    def _compute_grad_norm(self, module: nn.Module) -> float:
        """Compute L2 norm of gradients for parameters in the given module."""
        if module is None:
            return 0.0

        sq_sum = 0.0
        for param in module.parameters():
            if param.grad is None:
                continue
            param_norm = param.grad.data.norm(2)
            sq_sum += param_norm.item() ** 2

        return math.sqrt(sq_sum) if sq_sum > 0 else 0.0

    def _setup_optimizer(self) -> optim.AdamW:
        """Setup optimizer with parameter-specific learning rates."""
        params = []

        # ViT parameters
        vit_params = []
        for name, param in self.model.vision_encoder.named_parameters():
            if param.requires_grad:
                vit_params.append(param)
        if vit_params:
            params.append({
                "params": vit_params,
                "lr": self.vit_lr,
                "weight_decay": self.weight_decay,
                "name": "vit"
            })

        # BERT parameters
        bert_params = []
        adapter_params = []
        text_encoder = getattr(self.model, "text_encoder", None)
        if text_encoder is not None:
            for name, param in text_encoder.named_parameters():
                if not param.requires_grad:
                    continue
                name_lower = name.lower()
                if "adapter" in name_lower:
                    adapter_params.append(param)
                else:
                    bert_params.append(param)

        if bert_params:
            params.append({
                "params": bert_params,
                "lr": self.bert_lr,
                "weight_decay": self.weight_decay,
                "name": "bert"
            })

        if adapter_params:
            adapter_lr = self.lr_adapter if self.lr_adapter is not None else self.bert_lr
            params.append({
                "params": adapter_params,
                "lr": adapter_lr,
                "weight_decay": self.weight_decay,
                "name": "bert_adapter"
            })

        # Fusion head parameters
        head_params = []
        for name, param in self.model.fusion_head.named_parameters():
            if param.requires_grad:
                head_params.append(param)
        if head_params:
            params.append({
                "params": head_params,
                "lr": self.head_lr,
                "weight_decay": self.weight_decay,
                "name": "fusion"
            })

        optimizer = optim.AdamW(params)

        # Log parameter group info
        logger.info(f"Optimizer setup with {len(params)} parameter groups:")
        for i, group in enumerate(params):
            param_count = len(group["params"])
            logger.info(f"  Group {i+1} ({group['name']}): {param_count} parameters, lr={group['lr']}")

        return optimizer

    def _compute_modal_losses(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Compute monitoring losses for individual modalities."""
        modal_losses: Dict[str, float] = {}

        if not self.monitor_modal_losses:
            return modal_losses

        # Require multimodal components
        if (
            not hasattr(self.model, "vision_encoder")
            or not hasattr(self.model, "text_encoder")
            or getattr(self.model, "vision_encoder", None) is None
            or getattr(self.model, "text_encoder", None) is None
            or getattr(self.model, "fusion_head", None) is None
        ):
            return modal_losses

        with torch.no_grad():
            v_features = self.model.vision_encoder.get_image_feature(images)
            t_features = self.model.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

            zero_text = torch.zeros_like(t_features)
            zero_vision = torch.zeros_like(v_features)

            image_only_logits = self.model.fusion_head(v_features, zero_text)
            text_only_logits = self.model.fusion_head(zero_vision, t_features)

            modal_losses["image_loss"] = self.criterion(image_only_logits, labels).item()
            modal_losses["text_loss"] = self.criterion(text_only_logits, labels).item()

        return modal_losses

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        # Initialize loss and accuracy tracking
        epoch_losses = []
        correct_predictions = 0
        total_samples = 0
        modal_image_losses = []
        modal_text_losses = []
        vision_grad_norms = []
        text_grad_norms = []
        
        # Data timing
        data_time = 0
        batch_time = 0
        end = time.time()

        print(f'Epoch: {self.current_epoch}')
        
        for batch_idx, batch in enumerate(self.train_loader):
            data_time = time.time() - end
            
            # Move data to device
            images = batch["image"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass with mixed precision
            vision_grad_norm = None
            text_grad_norm = None
            grad_ratio = None

            if self.mixed_precision:
                with autocast():
                    logits = self.model(images, input_ids, attention_mask)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(images, input_ids, attention_mask)
                loss = self.criterion(logits, labels)

            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.monitor_modal_losses:
                    vision_grad_norm = self._compute_grad_norm(getattr(self.model, "vision_encoder", None))
                    text_grad_norm = self._compute_grad_norm(getattr(self.model, "text_encoder", None))
                    if text_grad_norm is not None and text_grad_norm > 0:
                        grad_ratio = vision_grad_norm / (text_grad_norm + 1e-12)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.monitor_modal_losses:
                    vision_grad_norm = self._compute_grad_norm(getattr(self.model, "vision_encoder", None))
                    text_grad_norm = self._compute_grad_norm(getattr(self.model, "text_encoder", None))
                    if text_grad_norm is not None and text_grad_norm > 0:
                        grad_ratio = vision_grad_norm / (text_grad_norm + 1e-12)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.optimizer.step()

            batch_time = time.time() - end

            # Calculate accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)
                batch_acc = (preds == labels).float().mean().item() * 100

            # Update metrics
            epoch_losses.append(loss.item())
            self.global_step += 1

            if self.monitor_modal_losses and vision_grad_norm is not None:
                vision_grad_norms.append(vision_grad_norm)
                text_grad_norms.append(text_grad_norm or 0.0)

            if self.monitor_modal_losses and (batch_idx % self.modal_loss_freq == 0):
                modal_loss_values = self._compute_modal_losses(images, input_ids, attention_mask, labels)
                if "image_loss" in modal_loss_values:
                    modal_image_losses.append(modal_loss_values["image_loss"])
                if "text_loss" in modal_loss_values:
                    modal_text_losses.append(modal_loss_values["text_loss"])

            # Print batch information following LaFTer style
            if batch_idx % self.print_freq == 0:
                current_acc = correct_predictions / total_samples * 100 if total_samples > 0 else 0.0
                grad_log = ""
                if (
                    self.monitor_modal_losses
                    and vision_grad_norm is not None
                    and text_grad_norm is not None
                ):
                    ratio_str = f"\tgrad_ratio {grad_ratio:.3f}" if grad_ratio is not None else ""
                    grad_log = f"\tvision_grad {vision_grad_norm:.3e}\ttext_grad {text_grad_norm:.3e}{ratio_str}"
                print(
                    "epoch [{0}/{1}][{2}/{3}]\t"
                    "loss {losses:.4f}\t"
                    "training_acc {training_acc:.2f}\t"
                    "lr {lr:.6e}{grad_log}".format(
                        self.current_epoch + 1,
                        self.epochs,
                        batch_idx + 1,
                        len(self.train_loader),
                        losses=loss.item(),
                        training_acc=current_acc,
                        lr=self.optimizer.param_groups[0]["lr"],
                        grad_log=grad_log
                    ))
                
                # Log parameter group learning rates
                for i, group in enumerate(self.optimizer.param_groups):
                    if i > 0:  # Skip first group (already shown above)
                        print(f"    lr_{group['name']}: {group['lr']:.6e}")

            # Log to wandb if enabled
            if self.log_wandb and grad_ratio is not None:
                self.wandb.log({"train/grad_ratio": grad_ratio}, step=self.global_step)

            if self.log_wandb and batch_idx % self.print_freq == 0:
                step_metrics = {
                    "train/loss": loss.item(),
                    "train/accuracy": current_acc,
                    "train/learning_rate": self.optimizer.param_groups[0]["lr"]
                }
                
                # Log parameter group learning rates
                for i, group in enumerate(self.optimizer.param_groups):
                    step_metrics[f"lr/{group['name']}"] = group['lr']
                
                self.wandb.log(step_metrics, step=self.global_step)

            end = time.time()

        # Calculate epoch metrics
        avg_loss = np.mean(epoch_losses)
        epoch_acc = correct_predictions / total_samples * 100 if total_samples > 0 else 0.0

        # Print epoch summary following LaFTer style
        print(f'Epoch [{self.current_epoch+1}/{self.epochs}] completed:')
        print(f'  Average Training Loss: {avg_loss:.4f}')
        print(f'  Training Accuracy: {epoch_acc:.2f}%')
        if self.monitor_modal_losses and modal_image_losses:
            avg_image_loss = np.mean(modal_image_losses)
            avg_text_loss = np.mean(modal_text_losses) if modal_text_losses else 0.0
            print(f'  Image-only loss (monitor): {avg_image_loss:.4f}')
            print(f'  Text-only loss (monitor): {avg_text_loss:.4f}')
        
        # Display learning rates for each parameter group
        print(f'  Learning Rates:')
        for i, group in enumerate(self.optimizer.param_groups):
            param_count = len(group["params"])
            trainable_params = sum(p.numel() for p in group["params"])
            print(f'    {group["name"].capitalize()}: lr={group["lr"]:.6e}, params={trainable_params:,}')

        metrics = {
            "train/epoch_loss": avg_loss,
            "train/epoch_accuracy": epoch_acc,
            "train/epoch_time": 0  # Time tracking removed to match LaFTer style
        }
        if self.monitor_modal_losses and modal_image_losses:
            metrics["train/image_loss_monitor"] = np.mean(modal_image_losses)
            metrics["train/text_loss_monitor"] = np.mean(modal_text_losses) if modal_text_losses else 0.0

        return metrics

    def validate(self):
        """Validate the model."""
        if self.val_loader is None:
            print("Warning: No validation loader provided")
            return {}

        self.model.eval()
        correct_predictions = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        print(f'Validation: {self.current_epoch}')
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                images = batch["image"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                # Forward pass
                if self.mixed_precision:
                    with autocast():
                        logits = self.model(images, input_ids, attention_mask)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(images, input_ids, attention_mask)
                    loss = self.criterion(logits, labels)

                # Calculate metrics
                preds = logits.argmax(dim=1)
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)
                batch_acc = (preds == labels).float().mean().item() * 100

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate validation metrics
        avg_acc = correct_predictions / total_samples * 100 if total_samples > 0 else 0.0

        print(f'TOP-1 Accuracy: {avg_acc:.2f}%')

        # Calculate per-class accuracy if possible
        try:
            from sklearn.metrics import classification_report
            # emotion_names = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
            # emotion_names = ["neg", "pos"]
            emotion_names = ["anger", "disgust", "fear", "awe", "amusement", "contentment", "sadness", "excitement"]
            unique_labels = sorted(set(all_labels + all_preds))
            class_names = [emotion_names[i] for i in unique_labels]
            report = classification_report(all_labels, all_preds, labels=unique_labels, target_names=class_names, output_dict=True)
            
            print(f'Per-class Results:')
            for class_name, metrics in report.items():
                if isinstance(metrics, dict) and 'f1-score' in metrics:
                    print(f'  {class_name}: F1={metrics["f1-score"]:.3f}, Precision={metrics["precision"]:.3f}, Recall={metrics["recall"]:.3f}')

        except ImportError:
            print("Note: sklearn not available, skipping per-class metrics")

        # ===== Binary Classification Evaluation =====
        print(f'\n=== Binary Classification Results ===')
        binary_acc = 0.0
        try:
            from utils.sentiment_mapping import get_binary_class_names

            # 将原始预测和标签转换为二分类
            binary_labels = [self._get_binary_mapping(label) for label in all_labels]
            binary_preds = [self._get_binary_mapping(pred) for pred in all_preds]

            # 计算二分类准确率
            binary_correct = sum(1 for bl, bp in zip(binary_labels, binary_preds) if bl == bp)
            binary_acc = binary_correct / len(binary_labels) * 100 if binary_labels else 0.0
            print(f'Binary Accuracy: {binary_acc:.2f}%')

            # 计算二分类指标
            binary_report = classification_report(
                binary_labels,
                binary_preds,
                labels=[0, 1],
                target_names=get_binary_class_names(),
                output_dict=True
            )

            # 打印每类指标
            print(f'Binary Per-class Results:')
            for class_name in get_binary_class_names():
                metrics = binary_report[class_name]
                print(f'  {class_name}: F1={metrics["f1-score"]:.3f}, '
                      f'Precision={metrics["precision"]:.3f}, Recall={metrics["recall"]:.3f}')

            # 打印 macro 和 weighted 平均
            macro_avg = binary_report["macro avg"]
            weighted_avg = binary_report["weighted avg"]
            print(f'  Macro Avg: F1={macro_avg["f1-score"]:.3f}, '
                  f'P={macro_avg["precision"]:.3f}, R={macro_avg["recall"]:.3f}')
            print(f'  Weighted Avg: F1={weighted_avg["f1-score"]:.3f}, '
                  f'P={weighted_avg["precision"]:.3f}, R={weighted_avg["recall"]:.3f}')

        except Exception as e:
            print(f"Binary classification error: {e}")
            import traceback
            traceback.print_exc()

        # ===== Binary Classification Evaluation End =====

        # Save confusion matrix for validation
        try:
            from sklearn.metrics import confusion_matrix
            import numpy as np
            import json

            # Create save directory: log/multimodal_new/{dataset_name}/confusion/
            save_dir = Path(f"/root/autodl-tmp/LaFTer-masterTEXT/log/multimodal_new/{self.dataset_name}/confusion")
            save_dir.mkdir(parents=True, exist_ok=True)

            # Compute and save confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            cm_path = save_dir / f"confusion_matrix_val_epoch_{self.current_epoch}.npy"
            np.save(cm_path, cm)

            # Save class names metadata
            metadata = {
                "phase": "val",
                "epoch": self.current_epoch,
                "dataset": self.dataset_name,
                "class_names": class_names
            }
            metadata_path = save_dir / f"metadata_val_epoch_{self.current_epoch}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"Confusion matrix saved: {cm_path}")

            # Save binary confusion matrix
            try:
                from sklearn.metrics import confusion_matrix

                # Calculate binary confusion matrix
                binary_labels = [self._get_binary_mapping(label) for label in all_labels]
                binary_preds = [self._get_binary_mapping(pred) for pred in all_preds]

                binary_cm = confusion_matrix(binary_labels, binary_preds)

                # Save binary confusion matrix
                binary_cm_path = save_dir / f"confusion_matrix_binary_val_epoch_{self.current_epoch}.npy"
                np.save(binary_cm_path, binary_cm)

                # Save binary confusion matrix metadata
                binary_metadata = {
                    "phase": "val_binary",
                    "epoch": self.current_epoch,
                    "dataset": self.dataset_name,
                    "class_names": ["neg", "pos"]
                }
                binary_metadata_path = save_dir / f"metadata_binary_val_epoch_{self.current_epoch}.json"
                with open(binary_metadata_path, 'w') as f:
                    json.dump(binary_metadata, f, indent=2)

                print(f"Binary confusion matrix saved: {binary_cm_path}")
            except Exception as e:
                print(f"Warning: Failed to save binary confusion matrix: {e}")
        except Exception as e:
            print(f"Warning: Failed to save confusion matrix: {e}")

        return {
            "val/accuracy": avg_acc,
            "val/binary_accuracy": binary_acc
        }

    def train(self):
        """Main training loop."""
        print("=== MULTIMODAL TRAINING PIPELINE ===")
        print(f"Training configuration:")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.train_loader.batch_size}")
        print(f"  ViT LR: {self.vit_lr}")
        print(f"  BERT LR: {self.bert_lr}")
        print(f"  Head LR: {self.head_lr}")
        print(f"  Weight decay: {self.weight_decay}")
        print(f"  Mixed precision: {self.mixed_precision}")
        print(f"  Device: {self.device}")
        
        # Display model parameter information
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Model parameters: {total_params:,} (trainable: {trainable_params:,})")
        
        # Display parameter group information
        print(f"  Parameter groups: {len(self.optimizer.param_groups)}")
        for i, group in enumerate(self.optimizer.param_groups):
            param_count = len(group["params"])
            trainable_count = sum(p.numel() for p in group["params"])
            print(f"    Group {i+1} ({group['name']}): {param_count} tensors, {trainable_count:,} parameters, lr={group['lr']:.6e}")
        
        print("="*60)

        all_acc = list()
        train_history = {"train_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # Training
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics["train/epoch_loss"])
            train_history["train_loss"].append(train_metrics["train/epoch_loss"])
            train_history["train_acc"].append(train_metrics["train/epoch_accuracy"])

            # Validation
            if self.val_loader is not None:
                val_metrics = self.validate()
                self.val_accuracies.append(val_metrics["val/accuracy"])
                train_history["val_acc"].append(val_metrics["val/accuracy"])
                all_acc.append(val_metrics["val/accuracy"])

                # Check for best model
                current_val_acc = val_metrics["val/accuracy"]
                if current_val_acc > self.best_val_acc:
                    self.best_val_acc = current_val_acc
                    self.save_checkpoint(is_best=True)
                    print(f'*** New best validation accuracy: {current_val_acc:.2f}% ***')

            # No longer save regular checkpoints to save disk space
            # Only best model is saved automatically when validation accuracy improves

            # Log epoch metrics to wandb
            if self.log_wandb:
                epoch_log = {**train_metrics, **(val_metrics if self.val_loader else {})}
                self.wandb.log(epoch_log, step=self.global_step)

        print(f'-------------------------------- Best Accuracy: {max(all_acc) if all_acc else 0:.2f} --------------------------------')
        print("Multimodal training completed!")
        
        return train_history

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint - only save best model to save disk space."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "train_losses": self.train_losses,
            "val_accuracies": self.val_accuracies,
            "train_config": {
                "vit_lr": self.vit_lr,
                "bert_lr": self.bert_lr,
                "head_lr": self.head_lr,
                "weight_decay": self.weight_decay,
                "epochs": self.epochs
            }
        }

        if self.mixed_precision and self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Only save best model to save disk space
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
        else:
            # Don't save regular checkpoints to save disk space
            print(f"Epoch {self.current_epoch+1} completed - checkpoint not saved (only best model is stored)")
            return

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_acc = checkpoint["best_val_acc"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_accuracies = checkpoint.get("val_accuracies", [])

        if self.mixed_precision and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")

    def test(self):
        """Test the model."""
        if self.test_loader is None:
            print("Warning: No test loader provided")
            return {}

        print("=== FINAL TEST EVALUATION ===")
        self.model.eval()

        test_losses = []
        correct_predictions = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.test_loader:
                # Move data to device
                images = batch["image"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                # Forward pass
                if self.mixed_precision:
                    with autocast():
                        logits = self.model(images, input_ids, attention_mask)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(images, input_ids, attention_mask)
                    loss = self.criterion(logits, labels)

                # Calculate metrics
                preds = logits.argmax(dim=1)
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)

                test_losses.append(loss.item())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate test metrics
        avg_loss = np.mean(test_losses)
        avg_acc = correct_predictions / total_samples * 100 if total_samples > 0 else 0.0

        print(f'Test Results:')
        print(f'  Average Test Loss: {avg_loss:.4f}')
        print(f'  Test Accuracy: {avg_acc:.2f}%')

        # Calculate per-class accuracy if possible
        try:
            from sklearn.metrics import classification_report, confusion_matrix
            emotion_names = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
            unique_labels = sorted(set(all_labels + all_preds))
            class_names = [emotion_names[i] for i in unique_labels]
            report = classification_report(all_labels, all_preds, labels=unique_labels, target_names=class_names, output_dict=True)
            
            print(f'Per-class Test Results:')
            for class_name, metrics in report.items():
                if isinstance(metrics, dict) and 'f1-score' in metrics:
                    print(f'  {class_name}: F1={metrics["f1-score"]:.3f}, Precision={metrics["precision"]:.3f}, Recall={metrics["recall"]:.3f}')
                    
            # Confusion matrix summary
            cm = confusion_matrix(all_labels, all_preds)
            print(f'Confusion Matrix Summary:')
            for i, class_name in enumerate(class_names):
                correct = cm[i, i] if i < len(cm) else 0
                total = sum(cm[i, :]) if i < len(cm) else 0
                print(f'  {class_name}: {correct}/{total} correct ({correct/total*100 if total>0 else 0:.1f}%)')

        except ImportError:
            print("Note: sklearn not available, skipping detailed metrics")

        # ===== Binary Test Results =====
        print(f'\n=== Binary Test Results ===')
        binary_acc = 0.0
        try:
            from utils.sentiment_mapping import get_binary_class_names

            # 将原始预测和标签转换为二分类
            binary_labels = [self._get_binary_mapping(label) for label in all_labels]
            binary_preds = [self._get_binary_mapping(pred) for pred in all_preds]

            # 计算二分类准确率
            binary_correct = sum(1 for bl, bp in zip(binary_labels, binary_preds) if bl == bp)
            binary_acc = binary_correct / len(binary_labels) * 100 if binary_labels else 0.0
            print(f'Binary Test Accuracy: {binary_acc:.2f}%')

            # 计算二分类指标
            binary_report = classification_report(
                binary_labels,
                binary_preds,
                labels=[0, 1],
                target_names=get_binary_class_names(),
                output_dict=True
            )

            # 打印每类指标
            print(f'Binary Test Results:')
            for class_name in get_binary_class_names():
                metrics = binary_report[class_name]
                print(f'  {class_name}: F1={metrics["f1-score"]:.3f}, '
                      f'P={metrics["precision"]:.3f}, R={metrics["recall"]:.3f}')

            # 打印 macro 和 weighted 平均
            macro_avg = binary_report["macro avg"]
            weighted_avg = binary_report["weighted avg"]
            print(f'  Macro Avg: F1={macro_avg["f1-score"]:.3f}, '
                  f'P={macro_avg["precision"]:.3f}, R={macro_avg["recall"]:.3f}')
            print(f'  Weighted Avg: F1={weighted_avg["f1-score"]:.3f}, '
                  f'P={weighted_avg["precision"]:.3f}, R={weighted_avg["recall"]:.3f}')

        except Exception as e:
            print(f"Binary test error: {e}")
            import traceback
            traceback.print_exc()

        # ===== Binary Test Results End =====

        print("="*60)

        return {
            "test/loss": avg_loss,
            "test/accuracy": avg_acc,
            "test/binary_accuracy": binary_acc
        }


if __name__ == "__main__":
    # Test the trainer (basic functionality)
    print("Testing MMTrainer...")

    # This would require actual data loaders and model
    # For now, just print initialization info
    print("MMTrainer module loaded successfully!")
    print("Use MMTrainer with proper data loaders for training.")
