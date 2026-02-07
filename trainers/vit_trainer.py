"""
ViT Trainer Implementation for LaFTer-ViT Ensemble Architecture
Based on LaFTer_ViT_Ensemble_Implementation_Plan.md Phase 1

This module implements:
1. ViTModel entity with HuggingFace ViT-base-patch16-224 integration
2. ViTTrainingService for training loop management
3. Device management and classification head replacement
4. Model persistence and evaluation functionality

Key Features:
- GPU/CPU automatic device adaptation
- Configurable training parameters (lr=3e-5, weight_decay=1e-5)
- Probability distribution output for ensemble integration
- Model save/load functionality with state management
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Optional, Union
import logging

# Set up logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HuggingFace transformers import - required for local ViT model
try:
    from transformers import ViTForImageClassification, ViTConfig, ViTFeatureExtractor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    raise ImportError("HuggingFace transformers is required for local ViT model. Please install: pip install transformers")

# Local ViT model path (manually downloaded)
LOCAL_VIT_PATH = r"/root/autodl-tmp/LaFTer-masterTEXT/models/google/vit-base-patch16-224"
VIT_NAME = "google/vit-base-patch16-224"
VIT_HIDDEN_SIZE = 768

class ViTModel(nn.Module):
    """
    ViT Model Entity - Uses local HuggingFace transformers ViT model only
    
    Loads ViT-base-patch16-224 from local directory with pretrained ImageNet weights.
    """
    
    def __init__(self, num_labels: int, model_name: str = VIT_NAME, device: str = 'cuda'):
        super(ViTModel, self).__init__()
        self.num_labels = num_labels
        self.model_name = model_name
        self.device = device
        
        # Load local transformers model
        self.vit_model = self._create_local_transformers_model()
        
        # Move to device
        self.vit_model.to(device)
        
        logger.info(f"ViT model initialized with {num_labels} classes on {device} using local transformers")
    
    def _initialize_model(self):
        """Initialize ViT model with local transformers only (force local model usage)"""
        
        # Force use local transformers model
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(f"HuggingFace transformers is required to use local ViT model at {LOCAL_VIT_PATH}")
        
        logger.info("Forcing local transformers ViT model usage...")
        return self._create_local_transformers_model()
    
    def _create_local_transformers_model(self):
        """Create ViT model using locally downloaded transformers model"""
        # Validate local model path exists
        if not os.path.exists(LOCAL_VIT_PATH):
            raise FileNotFoundError(f"Local ViT model not found at: {LOCAL_VIT_PATH}")
            
        # Load from local directory with pretrained weights
        logger.info(f"Loading local ViT model from: {LOCAL_VIT_PATH}")
        
        # Load config and update num_labels
        config = ViTConfig.from_pretrained(LOCAL_VIT_PATH)
        config.num_labels = self.num_labels
        
        # Load model with pretrained weights
        vit_model = ViTForImageClassification.from_pretrained(
            LOCAL_VIT_PATH,
            config=config,
            ignore_mismatched_sizes=True
        )
        
        # Replace classification head to match our classes (keeping pretrained features)
        vit_model.classifier = nn.Linear(VIT_HIDDEN_SIZE, self.num_labels)
        
        logger.info(f"✅ Successfully loaded local transformers ViT with {self.num_labels} classes")
        logger.info(f"   Model has {sum(p.numel() for p in vit_model.parameters()):,} parameters")
        logger.info(f"   Using ImageNet pretrained weights from: {LOCAL_VIT_PATH}")
        logger.info(f"   Classification head updated for {self.num_labels} classes")
        self.use_timm = False
        return vit_model
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ViT model
        
        Args:
            images: Input images tensor [batch_size, channels, height, width]
            
        Returns:
            logits: Classification logits [batch_size, num_labels]
        """
        outputs = self.vit_model(images)
        # HuggingFace transformers models return object with .logits attribute
        return outputs.logits
    
    def get_probabilities(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get probability distribution from model output
        
        Args:
            images: Input images tensor
            
        Returns:
            probabilities: Softmax probabilities [batch_size, num_labels]
        """
        with torch.no_grad():
            logits = self.forward(images)
            probabilities = torch.softmax(logits, dim=-1)
        return probabilities
    
    def freeze_backbone(self) -> None:
        """Freeze ViT backbone parameters, only train classifier"""
        # For transformers models
        for param in self.vit_model.vit.parameters():
            param.requires_grad = False
        logger.info("ViT backbone frozen, only classifier trainable")
    
    def unfreeze_all(self) -> None:
        """Unfreeze all parameters for full fine-tuning"""
        for param in self.vit_model.parameters():
            param.requires_grad = True
        logger.info("All ViT parameters unfrozen")


class ViTTrainingService:
    """
    ViT Training Service - Domain service for managing ViT training workflow
    
    Handles training loop implementation, optimizer configuration, and performance monitoring
    as specified in the LaFTer ensemble architecture plan.
    """
    
    def __init__(
        self,
        model: ViTModel,
        device: str = 'cuda',
        learning_rate: float = 3e-5,
        weight_decay: float = 1e-5,
        batch_size: int = 10,
        num_epochs: int = 10
    ):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_history = {
            'train_losses': [],
            'train_accuracies': [],
            'val_losses': [],
            'val_accuracies': []
        }
        
        logger.info(f"ViT Training Service initialized - LR: {learning_rate}, WD: {weight_decay}")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int = 0, print_freq: int = 10) -> Tuple[float, float]:
        """
        Train model for one epoch with enhanced accuracy monitoring
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            print_freq: Frequency of printing training progress
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\n=== ViT Training Epoch {epoch + 1}/{self.num_epochs} ===")
        
        for batch_idx, batch in enumerate(train_loader):
            # Extract images and labels from batch
            if isinstance(batch, dict):
                images = batch["img"].to(self.device)
                labels = batch["label"].to(self.device)
            else:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Calculate current batch and running accuracy
            batch_correct = predicted.eq(labels).sum().item()
            batch_acc = 100.0 * batch_correct / labels.size(0)
            running_acc = 100.0 * correct / total
            
            # Print training progress in enhanced LaFTer format
            if batch_idx % print_freq == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(
                    "epoch [{0}/{1}][{2}/{3}]\t"
                    "loss {loss:.4f}\t"
                    "batch_acc {batch_acc:.2f}%\t"
                    "running_acc {running_acc:.2f}%\t"
                    "lr {lr:.6e}".format(
                        epoch + 1,
                        self.num_epochs,
                        batch_idx + 1,
                        len(train_loader),
                        loss=loss.item(),
                        batch_acc=batch_acc,
                        running_acc=running_acc,
                        lr=current_lr,
                    )
                )
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        # Print epoch summary
        print(f"Epoch {epoch + 1} Summary: Loss = {avg_loss:.4f}, Training Acc = {accuracy:.2f}% ({correct}/{total})")
        
        return avg_loss, accuracy
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate model on validation/test data
        
        Args:
            data_loader: Validation/test data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(data_loader, desc="Evaluating")
            for batch in pbar:
                # Extract images and labels from batch
                if isinstance(batch, dict):
                    images = batch["img"].to(self.device)
                    labels = batch["label"].to(self.device)
                else:
                    images, labels = batch
                    images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                accuracy = 100.0 * correct / total
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{accuracy:.2f}%'
                })
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None
    ) -> Dict:
        """
        Complete training workflow with comprehensive accuracy monitoring
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of epochs (override default)
            
        Returns:
            Training history dictionary
        """
        if num_epochs is not None:
            self.num_epochs = num_epochs
        
        logger.info(f"Starting ViT training for {self.num_epochs} epochs")
        print("=" * 80)
        print("ViT MODEL TRAINING - LaFTer Ensemble Stage 1")
        print("=" * 80)
        
        best_val_acc = 0.0
        best_train_acc = 0.0
        best_model_state = None
        
        for epoch in range(self.num_epochs):
            print(f"\n>> EPOCH {epoch + 1}/{self.num_epochs}")
            print("-" * 50)
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader, epoch, print_freq=10)
            self.train_history['train_losses'].append(train_loss)
            self.train_history['train_accuracies'].append(train_acc)
            
            # Track best training accuracy
            if train_acc > best_train_acc:
                best_train_acc = train_acc
            
            # Validation phase
            if val_loader is not None:
                print(f"\n>> VALIDATION - Epoch {epoch + 1}")
                val_loss, val_acc = self.evaluate(val_loader)
                
                self.train_history['val_losses'].append(val_loss)
                self.train_history['val_accuracies'].append(val_acc)
                
                # Calculate train-val gap
                train_val_gap = train_acc - val_acc
                
                # Print comprehensive results
                print(f"\n>> EPOCH {epoch + 1} RESULTS:")
                print(f"   Training Loss:      {train_loss:.4f}")
                print(f"   Training Accuracy:  {train_acc:.2f}%")
                print(f"   Validation Loss:    {val_loss:.4f}")
                print(f"   Validation Accuracy: {val_acc:.2f}%")
                print(f"   Train-Val Gap:      {train_val_gap:.2f}%")
                
                # Overfitting warning (similar to gate trainer)
                if train_val_gap > 10:
                    print(f"   >> WARNING: Large train-val gap ({train_val_gap:.1f}%) - possible overfitting")
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = self.model.state_dict().copy()
                    print(f"   >> NEW BEST VALIDATION ACCURACY: {best_val_acc:.2f}%")
            else:
                # No validation data, just print training results
                print(f"\n>> EPOCH {epoch + 1} RESULTS:")
                print(f"   Training Loss:      {train_loss:.4f}")
                print(f"   Training Accuracy:  {train_acc:.2f}%")
            
            print("-" * 50)
        
        # Print final training summary
        print(f"\n>> TRAINING COMPLETED!")
        print("=" * 80)
        print(f"   Best Training Accuracy:    {best_train_acc:.2f}%")
        if val_loader is not None:
            print(f"   Best Validation Accuracy:  {best_val_acc:.2f}%")
            final_gap = best_train_acc - best_val_acc
            print(f"   Final Train-Val Gap:       {final_gap:.2f}%")
            
            # Overfitting assessment (similar to gate trainer)
            print(f"\n   Overfitting Assessment:")
            if final_gap <= 3:
                print(f"     Train-Val Gap: {final_gap:.1f}% - EXCELLENT (healthy training)")
            elif final_gap <= 8:
                print(f"     Train-Val Gap: {final_gap:.1f}% - GOOD (acceptable)")
            elif final_gap <= 15:
                print(f"     Train-Val Gap: {final_gap:.1f}% - WARNING (mild overfitting)")
            else:
                print(f"     Train-Val Gap: {final_gap:.1f}% - CRITICAL (severe overfitting)")
        
        print("=" * 80)
        print("ViT TRAINING COMPLETED")
        print("=" * 80)
        
        return self.train_history
    
    def get_predictions(self, data_loader: DataLoader) -> torch.Tensor:
        """
        Get probability distributions for ensemble integration
        
        Args:
            data_loader: Data loader for prediction
            
        Returns:
            Probability distributions tensor [num_samples, num_classes]
        """
        self.model.eval()
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Getting predictions"):
                if isinstance(batch, dict):
                    images = batch["img"].to(self.device)
                else:
                    images = batch[0].to(self.device)
                
                probs = self.model.get_probabilities(images)
                all_probs.append(probs.cpu())
        
        return torch.cat(all_probs, dim=0)
    
    def evaluate_test(self, test_loader: DataLoader) -> Tuple[float, float]:
        """
        Final test evaluation
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Tuple of (test_loss, test_accuracy)
        """
        test_loss, test_acc = self.evaluate(test_loader)
        logger.info(f"[ViT Test] Final Results: Loss {test_loss:.4f}, Acc {test_acc:.4f}")
        return test_loss, test_acc


class ViTModelPersistence:
    """
    Model Persistence Service - Infrastructure for multi-model state management
    
    Handles saving/loading ViT models with metadata for ensemble coordination.
    """
    
    @staticmethod
    def save_model(
        model: ViTModel,
        trainer: ViTTrainingService,
        save_path: str,
        dataset_name: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save ViT model with training history and metadata
        
        Args:
            model: ViT model instance
            trainer: Training service with history
            save_path: Base save directory
            dataset_name: Dataset name for path construction
            metadata: Additional metadata to save
            
        Returns:
            Full path to saved model
        """
        os.makedirs(save_path, exist_ok=True)
        
        model_filename = "best_model.pth"
        full_path = os.path.join(save_path, model_filename)
        
        # Prepare save data
        save_data = {
            'model_state_dict': model.state_dict(),
            'model_config': {
                'num_labels': model.num_labels,
                'model_name': model.model_name,
                'hidden_size': VIT_HIDDEN_SIZE
            },
            'training_config': {
                'learning_rate': trainer.learning_rate,
                'weight_decay': trainer.weight_decay,
                'batch_size': trainer.batch_size,
                'num_epochs': trainer.num_epochs
            },
            'training_history': trainer.train_history,
            'dataset_name': dataset_name,
            'metadata': metadata or {}
        }
        
        torch.save(save_data, full_path)
        logger.info(f"ViT model saved to {full_path}")
        
        return full_path
    
    @staticmethod
    def load_model(
        load_path: str,
        device: str = 'cuda',
        num_labels: Optional[int] = None
    ) -> Tuple[ViTModel, Dict]:
        """
        Load ViT model from saved checkpoint
        
        Args:
            load_path: Path to saved model
            device: Target device
            num_labels: Number of labels (if different from saved)
            
        Returns:
            Tuple of (loaded_model, metadata)
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        # Load checkpoint
        checkpoint = torch.load(load_path, map_location=device)
        
        # Extract config
        model_config = checkpoint['model_config']
        if num_labels is None:
            num_labels = model_config['num_labels']
        
        # Create model
        model = ViTModel(
            num_labels=num_labels,
            model_name=model_config['model_name'],
            device=device
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"ViT model loaded from {load_path}")
        
        return model, checkpoint
    
    @staticmethod
    def save_training_config(config: Dict, save_path: str) -> str:
        """
        Save training configuration as JSON
        
        Args:
            config: Configuration dictionary
            save_path: Save directory
            
        Returns:
            Path to saved config file
        """
        os.makedirs(save_path, exist_ok=True)
        config_path = os.path.join(save_path, "vit_training_config.json")
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Training config saved to {config_path}")
        return config_path


# Convenience functions for easy integration with LaFTer pipeline

def get_model(modelname: str, num_labels: int, device: str = 'cuda') -> ViTModel:
    """
    Model factory function - ALWAYS uses local ViT model
    
    Args:
        modelname: Model name ('ViT' for ViT-base-patch16-224)
        num_labels: Number of classification labels
        device: Target device
        
    Returns:
        Initialized ViT model using local pretrained weights
    """
    if modelname == 'ViT':
        model = ViTModel(num_labels=num_labels, device=device)
        logger.info(f"Created local ViT model with {num_labels} labels on {device}")
        logger.info(f"Model loaded from: {LOCAL_VIT_PATH}")
        return model
    else:
        raise ValueError(f"Unsupported model name: {modelname}")


def get_vit_feature_extractor() -> Optional[ViTFeatureExtractor]:
    """Get ViT feature extractor from local model"""
    try:
        if TRANSFORMERS_AVAILABLE:
            feature_extractor = ViTFeatureExtractor.from_pretrained(LOCAL_VIT_PATH)
            logger.info(f"✅ Successfully loaded ViTFeatureExtractor from {LOCAL_VIT_PATH}")
            return feature_extractor
        else:
            logger.warning("Transformers not available, cannot create ViTFeatureExtractor")
            return None
    except Exception as e:
        logger.warning(f"Failed to load ViTFeatureExtractor: {e}")
        return None


def create_vit_trainer(
    num_labels: int,
    device: str = 'cuda',
    **kwargs
) -> Tuple[ViTModel, ViTTrainingService]:
    """
    Factory function to create ViT model and trainer - ALWAYS uses local model
    
    Args:
        num_labels: Number of classification labels
        device: Target device
        **kwargs: Additional training parameters
        
    Returns:
        Tuple of (local_vit_model, trainer)
    """
    model = get_model('ViT', num_labels, device)
    trainer = ViTTrainingService(model, device=device, **kwargs)
    
    logger.info(f"Created ViT trainer using LOCAL model from: {LOCAL_VIT_PATH}")
    
    return model, trainer


def train_vit_standalone(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_labels: int,
    dataset_name: str,
    save_dir: str,
    device: str = 'cuda',
    **training_kwargs
) -> Tuple[ViTModel, Dict]:
    """
    Standalone ViT training function for integration with LaFTer pipeline
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader  
        test_loader: Test data loader
        num_labels: Number of classes
        dataset_name: Name of dataset
        save_dir: Directory to save model
        device: Training device
        **training_kwargs: Additional training parameters
        
    Returns:
        Tuple of (trained_model, results)
    """
    logger.info(f"Starting standalone ViT training for {dataset_name}")
    
    # Create model and trainer
    model, trainer = create_vit_trainer(
        num_labels=num_labels,
        device=device,
        **training_kwargs
    )
    
    # Train model
    history = trainer.train(train_loader, val_loader)
    
    # Final test evaluation
    test_loss, test_acc = trainer.evaluate_test(test_loader)
    
    # Save model
    model_path = ViTModelPersistence.save_model(
        model=model,
        trainer=trainer,
        save_path=save_dir,
        dataset_name=dataset_name,
        metadata={'final_test_acc': test_acc, 'final_test_loss': test_loss}
    )
    
    results = {
        'model_path': model_path,
        'training_history': history,
        'test_accuracy': test_acc,
        'test_loss': test_loss
    }
    
    logger.info(f"ViT training completed for {dataset_name}. Test accuracy: {test_acc:.4f}")
    
    return model, results


if __name__ == "__main__":
    # Example usage and testing
    logger.info("Testing ViT Trainer implementation...")
    
    # Test model creation
    try:
        model = get_model('ViT', num_labels=6, device='cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("✓ ViT model creation successful")
        
        # Test trainer creation
        trainer = ViTTrainingService(model)
        logger.info("✓ ViT trainer creation successful")
        
        # Test save/load
        save_path = "test_vit_model.pth"
        ViTModelPersistence.save_model(model, trainer, ".", "test")
        logger.info("✓ Model save successful")
        
        loaded_model, metadata = ViTModelPersistence.load_model("vit_test_model.pth")
        logger.info("✓ Model load successful")
        
        logger.info("All ViT Trainer tests passed!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise