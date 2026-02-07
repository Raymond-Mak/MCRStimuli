#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ViT Standalone Training Script
Based on LaFTer framework structure for Phase 1 ViT model testing
"""

import argparse
import torch
import datetime
import sys
import os
import wandb
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.data import DataManager

# Import datasets
import datasets.Emotion6
import datasets.Emoset
import datasets.caltech101
import datasets.cifar
import datasets.FI

# Import ViT trainer
from trainers.vit_trainer import create_vit_trainer, ViTModelPersistence
from utils.utils import lafter_datasets


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def get_num_classes(dataset_name):
    """Get number of classes for each dataset"""
    class_map = {
        'Emotion6': 6,
        'Emoset': 8,
        'FI': 8,
        'abstract': 8,
        'CIFAR10_local': 10,
        'CIFAR100_local': 100,
        'Caltech101': 101,
        'FGVCAircraft': 100,
        'OxfordFlowers': 102,
        'SUN397': 397,
        'DescribableTextures': 47,
        'EuroSAT': 10,
        'UCF101': 101
    }
    return class_map.get(dataset_name, 6)  # Default to 6


def setup_cfg(args):
    """Setup configuration"""
    cfg = get_cfg_default()
    
    # Dataset configuration
    cfg.DATASET.NAME = args.dataset
    cfg.DATASET.ROOT = "data"
    
    # Model configuration  
    cfg.MODEL.BACKBONE.NAME = "ViT-B/32"
    
    # Training configuration
    cfg.OPTIM.NAME = "adam"
    cfg.OPTIM.LR = args.lr
    cfg.TRAIN.MAX_EPOCH = args.epochs
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.batch_size
    
    # Output configuration - use provided output_dir or default
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    else:
        cfg.OUTPUT_DIR = f"output/vit/{args.dataset}"
    
    # Set random seed
    cfg.SEED = 7777
    
    return cfg


def test_vit_accuracy(model, trainer, test_loader, dataset_name, num_classes):
    """Test ViT classification accuracy"""
    print("\n" + "="*60)
    print("ViT CLASSIFICATION ACCURACY TEST")
    print("="*60)
    
    # Test evaluation
    test_loss, test_acc = trainer.evaluate_test(test_loader)
    
    # Print detailed results
    print(f"Dataset: {dataset_name}")
    print(f"Number of Classes: {num_classes}")
    print(f"Model: ViT-base-patch16-224 (timm)")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc:.2f}%)")
    print("="*60)
    
    return test_acc


def main():
    parser = argparse.ArgumentParser(description='ViT Standalone Training and Testing')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--test_only', action='store_true', help='Test only mode')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for saving model and results')
    
    # WandB related arguments - Complete parameter set
    parser.add_argument('--wandb_project', type=str, default='lafter-vit-training',
                       help='WandB project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='WandB run name (auto-generated if not specified)')
    parser.add_argument('--wandb_tags', type=str, nargs='*', default=[],
                       help='Additional WandB tags')
    parser.add_argument('--wandb_notes', type=str, default=None,
                       help='WandB run notes/description')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'],
                       help='WandB logging mode')
    parser.add_argument('--wandb_group', type=str, default=None,
                       help='WandB group name for organizing runs')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    # Setup configuration
    cfg = setup_cfg(args)
    set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)
    
    # Get number of classes
    num_classes = get_num_classes(args.dataset)
    
    # Initialize WandB with comprehensive parameter tracking
    wandb_config = {
        # Stage identification
        "training_stage": "vit_training",
        "stage_number": 1,
        "pipeline_component": "vision_transformer",
        
        # Basic experiment info
        "dataset_name": args.dataset,
        "num_classes": num_classes,
        "model_type": "vit_base_patch16_224",
        "model_architecture": "ViT-B/16",
        "pretrained_model": "timm/vit_base_patch16_224",
        
        # Training hyperparameters
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "optimizer": "Adam",
        
        # Model architecture parameters
        "patch_size": 16,
        "image_size": 224,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_layers": 12,
        
        # Data parameters
        "train_samples": None,  # Will be filled after data loading
        "test_samples": None,   # Will be filled after data loading
        "val_samples": None,    # Will be filled after data loading
        
        # System and environment
        "device": args.device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "pytorch_version": torch.__version__,
        
        # Training configuration
        "test_only": args.test_only,
        "output_dir": cfg.OUTPUT_DIR,
        "seed": cfg.SEED,
        
        # Local model configuration
        "local_model_path": "D:\\Deep Learning\\LLM\\FuXian\\vit-base-patch16-224",
        "use_local_pretrained": True,
        
        # Pipeline context (for ensemble training)
        "pipeline_mode": "ensemble_stage1",
        "next_stage": "text_classifier_training",
        
        # Experiment metadata
        "timestamp": datetime.datetime.now().isoformat(),
        "script_name": "vit.py",
        "framework": "timm + torch",
    }
    
    # Set WandB mode
    os.environ["WANDB_MODE"] = args.wandb_mode
    
    # Initialize WandB run
    wandb_run_name = args.wandb_run_name if args.wandb_run_name else f"vit_{args.dataset}_e{args.epochs}_{datetime.datetime.now().strftime('%m%d_%H%M')}"
    wandb_notes = args.wandb_notes if args.wandb_notes else f"ViT training on {args.dataset} - Stage 1 of LaFTer ensemble pipeline"
    
    wandb_tags = [
        "vit_training",
        "stage1", 
        "ensemble_component",
        args.dataset,
        "vision_transformer",
        f"epochs_{args.epochs}",
        f"bs_{args.batch_size}",
        f"lr_{args.lr}"
    ] + args.wandb_tags
    
    if not args.test_only:
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config=wandb_config,
            tags=wandb_tags,
            notes=wandb_notes,
            group=args.wandb_group if args.wandb_group else f"ensemble_{args.dataset}"
        )
        
        print(f"WandB initialized - Project: {args.wandb_project}")
        print(f"Run name: {wandb.run.name}")
        print(f"Run URL: {wandb.run.url}")
    
    # Print configuration
    print_args(args, cfg)
    print(f"\nTraining ViT on {args.dataset} with {num_classes} classes")
    
    # Create data manager
    print("Loading dataset...")
    dm = DataManager(cfg)
    
    print(f"Train samples: {len(dm.dataset.train_x)}")
    print(f"Test samples: {len(dm.dataset.test)}")
    
    # Update wandb config with actual data sizes
    if not args.test_only:
        wandb.config.update({
            "train_samples": len(dm.dataset.train_x),
            "test_samples": len(dm.dataset.test),
            "val_samples": len(dm.dataset.val) if hasattr(dm.dataset, 'val') and dm.dataset.val else 0
        }, allow_val_change=True)
    
    # Create ViT model and trainer - FORCE local model usage
    print("Creating ViT model and trainer using LOCAL pretrained model...")
    
    model, trainer = create_vit_trainer(
        num_labels=num_classes,
        device=args.device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_epochs=args.epochs
    )
    
    if not args.test_only:
        # Training phase with wandb logging
        print(f"Starting ViT training for {args.epochs} epochs...")
        
        # Log model parameters count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.log({
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/parameter_efficiency": trainable_params / total_params
        })
        
        history = trainer.train(
            train_loader=dm.train_loader_x,
            val_loader=dm.val_loader,
            num_epochs=args.epochs
        )
        
        # Log training history
        if 'train_losses' in history and history['train_losses']:
            for epoch, (train_loss, train_acc, val_loss, val_acc) in enumerate(zip(
                history['train_losses'], 
                history['train_accuracies'],
                history['val_losses'], 
                history['val_accuracies']
            )):
                wandb.log({
                    "vit/train_loss": train_loss,
                    "vit/train_accuracy": train_acc,
                    "vit/val_loss": val_loss,
                    "vit/val_accuracy": val_acc,
                    "vit/epoch": epoch + 1
                })
        
        # Save model
        print("Saving trained model...")
        model_path = ViTModelPersistence.save_model(
            model=model,
            trainer=trainer,
            save_path=cfg.OUTPUT_DIR,
            dataset_name=args.dataset
        )
        print(f"Model saved to: {model_path}")
        
        # Log model artifact
        artifact = wandb.Artifact(f"vit_model_{args.dataset}", type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        
        # Print and log training results
        if history['val_accuracies']:
            best_val_acc = max(history['val_accuracies'])
            best_train_acc = max(history['train_accuracies']) if history['train_accuracies'] else 0
            final_val_acc = history['val_accuracies'][-1] if history['val_accuracies'] else 0
            
            wandb.log({
                "vit/best_validation_accuracy": best_val_acc,
                "vit/best_training_accuracy": best_train_acc,
                "vit/final_validation_accuracy": final_val_acc,
                "vit/training_completed": True
            })
            
            print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    else:
        print("Skipping training (test_only mode)")
    
    # Testing phase - ViT classification accuracy
    print("Starting ViT classification accuracy testing...")
    final_accuracy = test_vit_accuracy(
        model, trainer, dm.test_loader, 
        args.dataset, num_classes
    )
    
    # Log final test results
    if not args.test_only:
        wandb.log({
            "vit/final_test_accuracy": final_accuracy,
            "vit/test_accuracy_percentage": final_accuracy * 100,
            "vit/model_path": model_path if 'model_path' in locals() else "test_only_mode"
        })
    
    # Save test results
    results_file = os.path.join(cfg.OUTPUT_DIR, f"vit_test_results_{args.dataset}.txt")
    with open(results_file, 'w') as f:
        f.write(f"ViT Classification Test Results\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Classes: {num_classes}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Test Accuracy: {final_accuracy:.4f}\n")
        f.write(f"Test Time: {datetime.datetime.now()}\n")
    
    print(f"\nResults saved to: {results_file}")
    
    # Finish wandb run
    if not args.test_only:
        # Log results file as artifact
        results_artifact = wandb.Artifact(f"vit_results_{args.dataset}", type="results")
        results_artifact.add_file(results_file)
        wandb.log_artifact(results_artifact)
        
        print(f"\nViT training completed. WandB run URL: {wandb.run.url}")
        wandb.finish()
    
    return final_accuracy


if __name__ == "__main__":
    main()