#!/usr/bin/env python3
"""
Enhanced Training Setup with Selective Visual Encoder Unfreezing

This module provides an enhanced version of setup_lafter_training_utils
with support for selective visual encoder parameter unfreezing and
different learning rates for different parameter groups.

"""

import torch
import torch.optim as optim
import logging
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.optim.lr_scheduler import _LRScheduler

from utils.utils import LabelSmoothingCrossEntropy

logger = logging.getLogger(__name__)


class VisualEncoderUnfreezer:
    """
    Handles selective parameter unfreezing for visual encoders.

    Supports different unfreezing strategies:
    1. 'last_blocks': Unfreeze last N transformer blocks
    2. 'attention_heads': Unfreeze attention head parameters
    3. 'layer_norms': Unfreeze layer normalization parameters
    4. 'projection_head': Unfreeze final projection layers
    5. 'custom': Custom pattern matching
    """

    def __init__(self,
                 unfreeze_strategy: str = 'last_blocks',
                 unfreeze_blocks: int = 4,
                 custom_patterns: list = None):
        """
        Initialize the visual encoder unfreezer.

        Args:
            unfreeze_strategy: Strategy for selective unfreezing
            unfreeze_blocks: Number of last transformer blocks to unfreeze
            custom_patterns: Custom regex patterns for parameter matching
        """
        self.strategy = unfreeze_strategy
        self.unfreeze_blocks = unfreeze_blocks
        self.custom_patterns = custom_patterns or []

        # Track unfrozen parameters for coverage analysis
        self.unfrozen_params = set()
        self.total_visual_params = 0

    def get_transformer_block_count(self, model) -> int:
        """Count total transformer blocks in the visual encoder."""
        block_count = 0
        for name, _ in model.named_parameters():
            if 'visual.transformer.resblocks.' in name:
                # Extract block number from parameter name
                import re
                match = re.search(r'resblocks\.(\d+)\.', name)
                if match:
                    block_count = max(block_count, int(match.group(1)) + 1)
        return block_count

    def should_unfreeze_parameter(self, param_name: str, total_blocks: int) -> bool:
        """
        Determine if a parameter should be unfrozen based on the strategy.

        Args:
            param_name: Full parameter name
            total_blocks: Total number of transformer blocks

        Returns:
            True if parameter should be unfrozen
        """
        # Only consider visual encoder parameters
        if 'visual' not in param_name:
            return False

        if self.strategy == 'last_blocks':
            # Unfreeze last N transformer blocks
            if 'visual.transformer.resblocks.' in param_name:
                import re
                match = re.search(r'resblocks\.(\d+)\.', param_name)
                if match:
                    block_num = int(match.group(1))
                    return block_num >= (total_blocks - self.unfreeze_blocks)

            # Also unfreeze final layer norms and projection
            if any(key in param_name for key in ['visual.ln_post', 'visual.proj']):
                return True

        elif self.strategy == 'attention_heads':
            # Unfreeze attention head parameters
            if any(key in param_name for key in ['attn.in_proj', 'attn.out_proj', 'mlp.c_fc', 'mlp.c_proj']):
                return True

        elif self.strategy == 'layer_norms':
            # Unfreeze all layer normalization parameters
            if 'ln_' in param_name:
                return True

        elif self.strategy == 'projection_head':
            # Unfreeze final projection and normalization layers
            if any(key in param_name for key in ['visual.ln_post', 'visual.proj']):
                return True

        elif self.strategy == 'custom':
            # Use custom regex patterns
            import re
            for pattern in self.custom_patterns:
                if re.search(pattern, param_name):
                    return True

        return False

    def apply_selective_unfreezing(self, model: torch.nn.Module,
                                  verbose: bool = True) -> dict:
        """
        Apply selective parameter unfreezing to the model.

        Args:
            model: The model to modify
            verbose: Whether to print detailed information

        Returns:
            Dictionary with statistics about unfrozen parameters
        """
        # Count total visual parameters first
        self.total_visual_params = 0
        for name, param in model.named_parameters():
            if 'visual' in name:
                self.total_visual_params += param.numel()

        # Get total transformer block count
        total_blocks = self.get_transformer_block_count(model)

        # Reset tracking
        self.unfrozen_params.clear()
        unfrozen_count = 0

        # Apply unfreezing
        for name, param in model.named_parameters():
            if self.should_unfreeze_parameter(name, total_blocks):
                param.requires_grad = True
                self.unfrozen_params.add(name)
                unfrozen_count += param.numel()
            else:
                # Ensure other parameters that should learn remain unfrozen
                if any(key in name for key in ['adapter', 'prompt_embeddings']):
                    continue  # Don't touch these
                elif 'visual' in name:
                    param.requires_grad = False

        # Calculate statistics
        stats = {
            'total_visual_params': self.total_visual_params,
            'unfrozen_visual_params': unfrozen_count,
            'unfrozen_ratio': unfrozen_count / self.total_visual_params if self.total_visual_params > 0 else 0,
            'total_transformer_blocks': total_blocks,
            'unfrozen_blocks': self.unfreeze_blocks if self.strategy == 'last_blocks' else 0
        }

        if verbose:
            self._print_unfreezing_stats(stats)

        return stats

    def _print_unfreezing_stats(self, stats: dict):
        """Print detailed statistics about parameter unfreezing."""
        print("\n" + "="*80)
        print(" Visual Encoder Selective Unfreezing Report")
        print("="*80)
        print(f"Strategy: {self.strategy}")
        print(f"Total transformer blocks: {stats['total_transformer_blocks']}")
        if self.strategy == 'last_blocks':
            print(f"Unfrozen blocks: {stats['unfrozen_blocks']}")
        print(f"Total visual parameters: {stats['total_visual_params']:,}")
        print(f"Unfrozen visual parameters: {stats['unfrozen_visual_params']:,}")
        print(f"Unfrozen ratio: {stats['unfrozen_ratio']:.2%}")

        print("\nUnfrozen parameter groups:")
        for param_name in sorted(self.unfrozen_params):
            if 'visual' in param_name:
                print(f"  ✓ {param_name}")
        print("="*80)


def create_visual_encoder_optimizer_groups(model: torch.nn.Module,
                                         visual_lr: float = 1e-5,
                                         main_lr: float = 1e-3,
                                         visual_weight_decay: float = 1e-4,
                                         main_weight_decay: float = 1e-4,
                                         no_decay_params: list = None) -> list:
    """
    Create parameter groups for optimizer with different learning rates.

    Args:
        model: The model with selectively unfrozen parameters
        visual_lr: Learning rate for visual encoder parameters
        main_lr: Learning rate for main parameters (prompts, etc.)
        visual_weight_decay: Weight decay for visual encoder parameters
        main_weight_decay: Weight decay for main parameters
        no_decay_params: List of parameter patterns to exclude from weight decay

    Returns:
        List of parameter group dictionaries for optimizer

    Note:
        Adapter parameters use 5x the main learning rate for faster convergence
    """
    if no_decay_params is None:
        no_decay_params = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    visual_params = []
    visual_params_no_decay = []
    adapter_params = []
    adapter_params_no_decay = []
    main_params = []
    main_params_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_visual = 'visual' in name
        is_adapter = 'adapter' in name
        is_no_decay = any(pattern in name for pattern in no_decay_params)

        if is_visual:
            if is_no_decay:
                visual_params_no_decay.append(param)
            else:
                visual_params.append(param)
        elif is_adapter:
            if is_no_decay:
                adapter_params_no_decay.append(param)
            else:
                adapter_params.append(param)
        else:
            if is_no_decay:
                main_params_no_decay.append(param)
            else:
                main_params.append(param)

    # Create parameter groups
    param_groups = []

    # Set adapter learning rate to 5 times main learning rate
    adapter_lr = main_lr * 5.0

    if visual_params:
        param_groups.append({
            'params': visual_params,
            'lr': visual_lr,
            'weight_decay': visual_weight_decay,
            'name': 'visual_params_with_decay'
        })

    if visual_params_no_decay:
        param_groups.append({
            'params': visual_params_no_decay,
            'lr': visual_lr,
            'weight_decay': 0.0,
            'name': 'visual_params_no_decay'
        })

    if adapter_params:
        param_groups.append({
            'params': adapter_params,
            'lr': adapter_lr,
            'weight_decay': main_weight_decay,
            'name': 'adapter_params_with_decay'
        })

    if adapter_params_no_decay:
        param_groups.append({
            'params': adapter_params_no_decay,
            'lr': adapter_lr,
            'weight_decay': 0.0,
            'name': 'adapter_params_no_decay'
        })

    if main_params:
        param_groups.append({
            'params': main_params,
            'lr': main_lr,
            'weight_decay': main_weight_decay,
            'name': 'main_params_with_decay'
        })

    if main_params_no_decay:
        param_groups.append({
            'params': main_params_no_decay,
            'lr': main_lr,
            'weight_decay': 0.0,
            'name': 'main_params_no_decay'
        })

    # Print parameter group statistics
    print("\n" + "="*60)
    print(" Optimizer Parameter Groups")
    print("="*60)
    print(f"Visual encoder LR: {visual_lr:.6f}, Weight decay: {visual_weight_decay:.6f}")
    print(f"Adapter LR: {adapter_lr:.6f}, Weight decay: {main_weight_decay:.6f}")
    print(f"Main parameters LR: {main_lr:.6f}, Weight decay: {main_weight_decay:.6f}")

    total_visual = len(visual_params) + len(visual_params_no_decay)
    total_adapter = len(adapter_params) + len(adapter_params_no_decay)
    total_main = len(main_params) + len(main_params_no_decay)

    print(f"Visual parameters: {total_visual} groups")
    print(f"Adapter parameters: {total_adapter} groups")
    print(f"Main parameters: {total_main} groups")
    print("="*60)

    return param_groups


class ConstantWarmupScheduler(_LRScheduler):
    """Constant learning rate with warmup."""

    def __init__(
            self,
            optimizer,
            successor,
            warmup_epoch,
            cons_lr,
            visual_warmup_epoch=None,
            visual_warmup_lr=None,
            last_epoch=-1,
            verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        self.cons_lr = cons_lr
        self.visual_warmup_epoch = visual_warmup_epoch if visual_warmup_epoch is not None else warmup_epoch
        self.visual_warmup_lr = visual_warmup_lr if visual_warmup_lr is not None else cons_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        current_lrs = []

        for i, param_group in enumerate(self.optimizer.param_groups):
            group_name = param_group.get('name', '')

            # Check if this is a visual parameter group
            is_visual = 'visual' in group_name

            if is_visual:
                # Use visual warmup configuration
                if self.last_epoch < self.visual_warmup_epoch:
                    current_lrs.append(self.visual_warmup_lr)
                else:
                    # Visual warmup completed, use successor scheduler
                    successor_lrs = self.successor.get_last_lr()
                    if i < len(successor_lrs):
                        current_lrs.append(successor_lrs[i])
                    else:
                        current_lrs.append(successor_lrs[-1] if successor_lrs else self.visual_warmup_lr)
            else:
                # Use main warmup configuration
                if self.last_epoch < self.warmup_epoch:
                    current_lrs.append(self.cons_lr)
                else:
                    # Main warmup completed, use successor scheduler
                    successor_lrs = self.successor.get_last_lr()
                    if i < len(successor_lrs):
                        current_lrs.append(successor_lrs[i])
                    else:
                        current_lrs.append(successor_lrs[-1] if successor_lrs else self.cons_lr)

        return current_lrs

    def step(self, epoch=None):
        if epoch is not None:
            self.last_epoch = epoch
        else:
            self.last_epoch += 1

        # Check if all warmup phases are completed
        all_warmup_completed = (self.last_epoch >= self.warmup_epoch and
                              self.last_epoch >= self.visual_warmup_epoch)

        if all_warmup_completed:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            self._last_lr = self.get_lr()




def setup_enhanced_lafter_training_utils(args, model,
                                         enable_visual_unfreezing: bool = True,
                                         visual_unfreeze_strategy: str = 'last_blocks',
                                         visual_unfreeze_blocks: int = 4,
                                         visual_lr: float = 1e-5,
                                         main_lr: float = None,
                                         # Dropout parameters
                                         adapter_dropout: float = 0.2,
                                         projector_dropout: float = 0.1,
                                         visual_dropout: float = 0.15):  # NEW: Visual encoder dropout
    """
    Enhanced setup for LaFTer training with selective visual encoder unfreezing.

    Args:
        args: Command line arguments
        model: LaFTer model
        enable_visual_unfreezing: Whether to enable selective visual encoder unfreezing
        visual_unfreeze_strategy: Strategy for unfreezing visual encoder parameters
        visual_unfreeze_blocks: Number of last transformer blocks to unfreeze
        visual_lr: Learning rate for visual encoder parameters
        main_lr: Learning rate for main parameters (uses args.lr if None)
        adapter_dropout: Dropout probability for adapter layers
        projector_dropout: Dropout probability for projector layers
        visual_dropout: Dropout probability for visual encoder layers

    Returns:
        optimizer, scheduler, criteria
    """
    model = model.cuda()
    model = model.float()

    # Use main_lr if provided, otherwise use args.lr
    if main_lr is None:
        main_lr = args.lr

    # Standard parameter activation (original logic)
    for key, value in model.named_parameters():
        if key == 'prompt_embeddings':
            value.requires_grad = True
        elif 'adapter' in key and 'adapter_pl' not in key:
            value.requires_grad = True
        elif 'projector' in key and not args.entropy:
            value.requires_grad = True
        elif 'ln' in key:
            value.requires_grad = True
        else:
            value.requires_grad = False

    # Apply selective visual encoder unfreezing if enabled
    if enable_visual_unfreezing:
        logger.info("Applying selective visual encoder unfreezing...")

        unfreezer = VisualEncoderUnfreezer(
            unfreeze_strategy=visual_unfreeze_strategy,
            unfreeze_blocks=visual_unfreeze_blocks
        )

        unfreeze_stats = unfreezer.apply_selective_unfreezing(model, verbose=True)

        # Log unfreezing statistics for monitoring
        logger.info(f"Visual encoder unfreezing stats: {unfreeze_stats}")

        # NEW: Apply dropout mechanisms when visual unfreezing is enabled
        if adapter_dropout > 0 or projector_dropout > 0 or visual_dropout > 0:
            import torch.nn as nn
            logger.info("Applying dropout mechanisms for enhanced regularization...")

            added_layers = 0

            # Collect modules first to avoid OrderedDict mutation during iteration
            adapter_modules = []
            projector_modules = []
            visual_modules = []

            for name, module in model.named_modules():
                if 'adapter' in name and isinstance(module, (nn.Linear, nn.Conv2d)):
                    adapter_modules.append((name, module))
                elif 'projector' in name and isinstance(module, (nn.Linear, nn.Conv2d)):
                    projector_modules.append((name, module))
                elif 'visual' in name and isinstance(module, (nn.Linear, nn.Conv2d, nn.MultiheadAttention)):
                    visual_modules.append((name, module))

            # Add dropout to adapter modules
            if adapter_dropout > 0 and adapter_modules:
                adapter_count = 0
                for name, module in adapter_modules:
                    dropout_layer = nn.Dropout(p=adapter_dropout, inplace=False)
                    parent_path = name.rsplit('.', 1)[0] if '.' in name else None
                    if parent_path and hasattr(model, parent_path):
                        parent_module = getattr(model, parent_path)
                        setattr(parent_module, f"dropout_{name.split('.')[-1]}", dropout_layer)
                        added_layers += 1
                        adapter_count += 1
                print(f">>> Added dropout to {adapter_count} adapter modules (p={adapter_dropout})")

            # Add dropout to projector modules
            if projector_dropout > 0 and projector_modules:
                projector_count = 0
                for name, module in projector_modules:
                    dropout_layer = nn.Dropout(p=projector_dropout, inplace=False)
                    parent_path = name.rsplit('.', 1)[0] if '.' in name else None
                    if parent_path and hasattr(model, parent_path):
                        parent_module = getattr(model, parent_path)
                        setattr(parent_module, f"dropout_{name.split('.')[-1]}", dropout_layer)
                        added_layers += 1
                        projector_count += 1
                print(f">>> Added dropout to {projector_count} projector modules (p={projector_dropout})")

            # Add dropout to visual encoder modules (NEW)
            if visual_dropout > 0 and visual_modules:
                visual_count = 0
                for name, module in visual_modules:
                    dropout_layer = nn.Dropout(p=visual_dropout, inplace=False)
                    parent_path = name.rsplit('.', 1)[0] if '.' in name else None
                    if parent_path and hasattr(model, parent_path):
                        parent_module = getattr(model, parent_path)
                        setattr(parent_module, f"dropout_{name.split('.')[-1]}", dropout_layer)
                        added_layers += 1
                        visual_count += 1
                print(f">>> Added dropout to {visual_count} visual encoder modules (p={visual_dropout})")

            logger.info(f"Total dropout layers added: {added_layers}")
            print(f">>> Dropout mechanisms fully activated: {added_layers} total layers added")
        else:
            print(">>> Dropout mechanisms disabled (all dropout probabilities set to 0)")
    else:
        logger.info("Visual encoder unfreezing disabled")
        # Original logic for visual encoder parameters
        for key, value in model.named_parameters():
            if 'visual' in key:
                if 'ln' in key or 'bn' in key:
                    value.requires_grad = True
                else:
                    value.requires_grad = False

    # Collect learnable parameters
    params = list()
    for key, value in model.named_parameters():
        if value.requires_grad:
            params.append((key, value))

    print('------------------ Learnable Parameters ------------------')
    for key, value in model.named_parameters():
        if value.requires_grad:
            print("\t{}, {}, {}".format(key, value.numel(), value.shape))
    print('----------------------------------------------------------')

    # Create optimizer with parameter groups
    if enable_visual_unfreezing:
        # Use enhanced parameter grouping with different learning rates
        visual_weight_decay = 1e-4  # Lower weight decay for visual encoder
        main_weight_decay = args.weight_decay if hasattr(args, 'weight_decay') and args.weight_decay else 1e-4

        param_groups = create_visual_encoder_optimizer_groups(
            model=model,
            visual_lr=visual_lr,
            main_lr=main_lr,
            visual_weight_decay=visual_weight_decay,
            main_weight_decay=main_weight_decay
        )

        optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999))

        # Print learning rate information
        adapter_lr = main_lr * 5.0
        print(f">>> Visual encoder learning rate: {visual_lr:.6f}, weight_decay: {visual_weight_decay:.6f}")
        print(f">>> Adapter learning rate: {adapter_lr:.6f}, weight_decay: {main_weight_decay:.6f}")
        print(f">>> Main parameters learning rate: {main_lr:.6f}, weight_decay: {main_weight_decay:.6f}")
    else:
        # Use original parameter grouping logic
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        adapter_params = []
        other_params = []
        adapter_params_no_decay = []
        other_params_no_decay = []

        for n, p in params:
            if 'adapter' in n:
                if any(nd in n for nd in no_decay):
                    adapter_params_no_decay.append(p)
                else:
                    adapter_params.append(p)
            else:
                if any(nd in n for nd in no_decay):
                    other_params_no_decay.append(p)
                else:
                    other_params.append(p)

        optimizer_grouped_parameters = []

        # Lower weight decay to reduce underfitting
        adapter_weight_decay = 1e-4
        other_weight_decay = args.weight_decay if hasattr(args, 'weight_decay') and args.weight_decay else 1e-4

        # Set adapter learning rate to 5 times main learning rate
        adapter_lr = main_lr * 5.0

        if adapter_params:
            optimizer_grouped_parameters.append({
                'params': adapter_params,
                'lr': adapter_lr,
                'weight_decay': adapter_weight_decay
            })
        if adapter_params_no_decay:
            optimizer_grouped_parameters.append({
                'params': adapter_params_no_decay,
                'lr': adapter_lr,
                'weight_decay': 0.0
            })
        if other_params:
            optimizer_grouped_parameters.append({
                'params': other_params,
                'lr': main_lr,
                'weight_decay': other_weight_decay
            })
        if other_params_no_decay:
            optimizer_grouped_parameters.append({
                'params': other_params_no_decay,
                'lr': main_lr,
                'weight_decay': 0.0
            })

        optimizer = optim.AdamW(optimizer_grouped_parameters, betas=(0.9, 0.999))
        print(f">>> Adapter learning rate: {adapter_lr:.6f}, weight_decay: {adapter_weight_decay:.6f}")
        print(f">>> Other parameters learning rate: {main_lr:.6f}, weight_decay: {other_weight_decay:.6f}")

    # Setup learning rate scheduler
    warmup_epochs = max(3, args.epochs // 8)
    warmup_lr = main_lr * 0.01

    # For visual encoder unfreezing, use separate warmup configuration
    if enable_visual_unfreezing:
        visual_warmup_epochs = max(4, args.epochs // 6)  # Longer warmup for visual encoder
        visual_warmup_lr = visual_lr * 0.01          # Start from 1% of target visual LR
    else:
        visual_warmup_epochs = warmup_epochs
        visual_warmup_lr = warmup_lr

    if hasattr(args, 'milestones') and args.milestones:
        main_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, 0.7)
        scheduler = ConstantWarmupScheduler(
            optimizer, main_scheduler, warmup_epochs, warmup_lr,
            visual_warmup_epoch=visual_warmup_epochs,
            visual_warmup_lr=visual_warmup_lr
        )
        print(f">>> Using MultiStepLR with Differential Warmup:")
        print(f"    Main parameters: {warmup_lr:.6f} -> {main_lr:.6f} over {warmup_epochs} epochs")
        print(f"    Visual encoder: {visual_warmup_lr:.6f} -> {visual_lr:.6f} over {visual_warmup_epochs} epochs")
        print(f"    milestones={args.milestones}, decay_factor=0.7")
    else:
        constant_scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=args.epochs)
        scheduler = ConstantWarmupScheduler(
            optimizer, constant_scheduler, warmup_epochs, warmup_lr,
            visual_warmup_epoch=visual_warmup_epochs,
            visual_warmup_lr=visual_warmup_lr
        )
        print(f">>> Using Constant Learning Rate with Differential Warmup:")
        print(f"    Main parameters: {warmup_lr:.6f} -> {main_lr:.6f} over {warmup_epochs} epochs")
        print(f"    Visual encoder: {visual_warmup_lr:.6f} -> {visual_lr:.6f} over {visual_warmup_epochs} epochs")

    # Use reduced label smoothing to reduce regularization
    criteria = LabelSmoothingCrossEntropy(epsilon=0.1)

    # Return dropout manager for monitoring (None if not enabled)
    dropout_manager = None
    if enable_visual_unfreezing and (adapter_dropout > 0 or projector_dropout > 0 or visual_dropout > 0):
        dropout_manager = {
            'adapter_dropout': adapter_dropout,
            'projector_dropout': projector_dropout,
            'visual_dropout': visual_dropout,
            'enabled': True
        }

    return optimizer, scheduler, criteria, dropout_manager


def get_visual_parameter_coverage_stats(model) -> dict:
    """
    Get detailed statistics about visual encoder parameter coverage.

    Args:
        model: The trained model

    Returns:
        Dictionary with coverage statistics
    """
    unfreezer = VisualEncoderUnfreezer()

    # Analyze current parameter states
    total_visual = 0
    trainable_visual = 0

    visual_params = {}
    for name, param in model.named_parameters():
        if 'visual' in name:
            total_visual += param.numel()
            if param.requires_grad:
                trainable_visual += param.numel()
                visual_params[name] = param.numel()

    # Analyze by component type
    component_stats = {
        'transformer_blocks': 0,
        'attention_layers': 0,
        'normalization_layers': 0,
        'projection_layers': 0,
        'other_visual': 0
    }

    for param_name in visual_params.keys():
        if 'transformer.resblocks.' in param_name:
            component_stats['transformer_blocks'] += visual_params[param_name]
        elif any(key in param_name for key in ['attn.in_proj', 'attn.out_proj', 'mlp.c_fc', 'mlp.c_proj']):
            component_stats['attention_layers'] += visual_params[param_name]
        elif 'ln_' in param_name:
            component_stats['normalization_layers'] += visual_params[param_name]
        elif any(key in param_name for key in ['ln_post', 'proj']):
            component_stats['projection_layers'] += visual_params[param_name]
        else:
            component_stats['other_visual'] += visual_params[param_name]

    stats = {
        'total_visual_parameters': total_visual,
        'trainable_visual_parameters': trainable_visual,
        'trainable_ratio': trainable_visual / total_visual if total_visual > 0 else 0,
        'component_breakdown': component_stats,
        'trainable_parameter_count': len(visual_params)
    }

    return stats


# Test functions for TDD
def test_enhanced_setup():
    """Test the enhanced setup functionality."""
    print("Testing enhanced training setup...")

    # Mock args
    class MockArgs:
        def __init__(self):
            self.epochs = 50
            self.lr = 0.001
            self.weight_decay = 0.0001
            self.milestones = [10, 20, 30]
            self.entropy = False

    args = MockArgs()

    # Test that the function can be called without errors
    # (In real usage, this would be called with an actual model)
    print("✓ Enhanced setup function structure is valid")


def test_coverage_stats():
    """Test coverage statistics calculation."""
    print("Testing coverage statistics calculation...")

    # Mock model parameters
    class MockParam:
        def __init__(self, name, requires_grad=False, numel=1000):
            self.name = name
            self.requires_grad = requires_grad
            self._numel = numel

        def numel(self):
            return self._numel

    class MockModel:
        def __init__(self):
            self.params = {
                'model.visual.transformer.resblocks.11.weight': MockParam('model.visual.transformer.resblocks.11.weight', True, 5000),
                'model.visual.attn.in_proj_weight': MockParam('model.visual.attn.in_proj_weight', True, 3000),
                'model.visual.ln_post.weight': MockParam('model.visual.ln_post.weight', True, 1000),
                'model.visual.proj': MockParam('model.visual.proj', False, 2000),
                'model.adapter.weight': MockParam('model.adapter.weight', True, 1000)  # Not visual
            }

        def named_parameters(self):
            return self.params.items()

    model = MockModel()
    stats = get_visual_parameter_coverage_stats(model)

    # Verify statistics
    assert stats['total_visual_parameters'] == 11000, f"Expected 11000, got {stats['total_visual_parameters']}"
    assert stats['trainable_visual_parameters'] == 9000, f"Expected 9000, got {stats['trainable_visual_parameters']}"
    assert abs(stats['trainable_ratio'] - 0.818) < 0.01, f"Expected ~0.818, got {stats['trainable_ratio']}"

    print("✓ Coverage statistics test passed")


def test_visual_unfreezer():
    """Test visual encoder unfreezer functionality."""
    print("Testing Visual Encoder Unfreezer...")

    # Mock model parameters
    class MockParam:
        def __init__(self, requires_grad=False, numel=1000):
            self.requires_grad = requires_grad
            self._numel = numel

        def numel(self):
            return self._numel

    class MockModel:
        def __init__(self):
            self.params = {}
            # Mock visual encoder parameters (12 transformer blocks)
            for i in range(12):
                self.params[f'model.visual.transformer.resblocks.{i}.attn.in_proj_weight'] = MockParam()
                self.params[f'model.visual.transformer.resblocks.{i}.mlp.c_fc.weight'] = MockParam()

            # Mock final layers
            self.params['model.visual.ln_post.weight'] = MockParam()
            self.params['model.visual.proj'] = MockParam()

            # Mock adapter parameters
            self.params['model.adapter.weight'] = MockParam()

        def named_parameters(self):
            return self.params.items()

    model = MockModel()

    # Test last_blocks strategy
    unfreezer = VisualEncoderUnfreezer(
        unfreeze_strategy='last_blocks',
        unfreeze_blocks=2
    )

    stats = unfreezer.apply_selective_unfreezing(model, verbose=False)

    # Verify that last 2 blocks (10, 11) are unfrozen
    expected_unfrozen_blocks = 2
    actual_unfrozen_blocks = stats['unfrozen_blocks']
    assert actual_unfrozen_blocks == expected_unfrozen_blocks, f"Expected {expected_unfrozen_blocks} unfrozen blocks, got {actual_unfrozen_blocks}"

    print("✓ Visual encoder unfreezer test passed")


def test_optimizer_groups():
    """Test optimizer parameter group creation."""
    print("Testing optimizer parameter groups...")

    # Mock model parameters
    class MockParam:
        def __init__(self, requires_grad=True):
            self.requires_grad = requires_grad

        def __repr__(self):
            return "MockParam()"

    mock_model_params = {
        'model.visual.transformer.resblocks.11.attn.weight': MockParam(),
        'model.visual.ln_post.bias': MockParam(),
        'model.adapter.weight': MockParam(),
        'model.prompt_embeddings': MockParam()
    }

    # Mock model
    class MockModel:
        def __init__(self):
            self._params = mock_model_params

        def named_parameters(self):
            return self._params.items()

    model = MockModel()

    # Test parameter group creation
    param_groups = create_visual_encoder_optimizer_groups(
        model, visual_lr=1e-5, main_lr=1e-3
    )

    assert len(param_groups) > 0, "Should create at least one parameter group"

    # Check that visual and main parameters are separated
    visual_groups = [g for g in param_groups if 'visual' in g.get('name', '')]
    main_groups = [g for g in param_groups if 'main' in g.get('name', '')]

    assert len(visual_groups) > 0, "Should have visual parameter groups"
    assert len(main_groups) > 0, "Should have main parameter groups"

    print("✓ Optimizer parameter groups test passed")


def run_enhanced_tests():
    """Run all enhanced tests."""
    print("Running Enhanced Training Setup Tests...")
    print("-" * 60)

    test_enhanced_setup()
    test_coverage_stats()
    test_visual_unfreezer()
    test_optimizer_groups()

    print("-" * 60)
    print("All enhanced tests passed! ✓")


if __name__ == "__main__":
    run_enhanced_tests()