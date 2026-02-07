"""
BERT Text Encoder for Multimodal Training

This module provides a BERT wrapper for extracting text features
in multimodal training pipelines.
"""

import os
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

ADAPTER_INITIALIZER = None
try:
    from transformers.adapters import AdapterConfig
except Exception:
    try:
        from adapters import AdapterConfig  # type: ignore
        from adapters import init as adapters_init  # type: ignore

        ADAPTER_INITIALIZER = adapters_init
    except Exception:
        AdapterConfig = None
        ADAPTER_INITIALIZER = None

DEFAULT_ADAPTER_NAME = "task_adapter"


class BertTextEncoder(nn.Module):
    """
    BERT text encoder wrapper that can be frozen or fine-tuned.

    Extracts CLS token features from BERT for multimodal fusion.
    """

    def __init__(
        self,
        model_dir: str,
        freeze: bool = True,
        proj_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_adapter: bool = False,
        adapter_type: str = "pfeiffer",
        adapter_reduction_factor: int = 16,
        adapter_dropout: float = 0.0,
        adapter_trainable: bool = True
    ):
        """
        Args:
            model_dir: Path to trained BERT model directory
            freeze: Whether to freeze BERT parameters
            proj_dim: Optional projection dimension (if None, use BERT hidden size)
            dropout: Dropout rate for projection layer
        """
        super().__init__()

        self.model_dir = model_dir
        self.freeze = freeze
        self.adapter_type = adapter_type
        self.adapter_reduction = adapter_reduction_factor
        self.adapter_dropout = adapter_dropout
        self.adapter_trainable = adapter_trainable
        self.adapter_name = DEFAULT_ADAPTER_NAME
        self.use_adapter = use_adapter

        # Load BERT model and config
        try:
            self.config = AutoConfig.from_pretrained(model_dir)
            self.backbone = AutoModel.from_pretrained(model_dir)
        except Exception as e:
            print(f"Error loading BERT from {model_dir}: {e}")
            print("Falling back to bert-base-uncased")
            self.config = AutoConfig.from_pretrained("bert-base-uncased")
            self.backbone = AutoModel.from_pretrained("bert-base-uncased")

        self.out_dim = self.config.hidden_size  # Typically 768 for bert-base

        # Setup projection layer if needed
        if proj_dim is not None and proj_dim != self.out_dim:
            self.proj = nn.Sequential(
                nn.Linear(self.out_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.out_dim = proj_dim
        else:
            self.proj = nn.Identity()

        # Freeze parameters if requested
        if freeze:
            self._freeze_parameters()

        # Automatically enable adapters if the loaded checkpoint already contains one
        if not self.use_adapter:
            self.use_adapter = self._has_pretrained_adapter()

        if self.use_adapter:
            self._initialize_adapter_runtime()
            self._setup_adapter()

        print(f"BERT Text Encoder initialized:")
        print(f"  Model: {model_dir}")
        print(f"  Hidden size: {self.config.hidden_size}")
        print(f"  Output dim: {self.out_dim}")
        print(f"  Frozen: {freeze}")
        if self.use_adapter:
            print(f"  Adapter active: {self.adapter_name}")
            print(f"    Type: {self.adapter_type}")
            print(f"    Reduction factor: {self.adapter_reduction}")
            print(f"    Dropout: {self.adapter_dropout}")
            print(f"    Trainable: {self.adapter_trainable}")

    def _freeze_parameters(self):
        """Freeze BERT backbone parameters."""
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        print(f"Frozen {sum(1 for _ in self.backbone.parameters())} BERT parameters")

    def _has_pretrained_adapter(self) -> bool:
        """Check if the loaded checkpoint already contains an adapter with the default name."""
        adapters_cfg = getattr(getattr(self.backbone, "config", None), "adapters", None)
        try:
            return self.adapter_name in getattr(adapters_cfg, "adapters", {})
        except Exception:
            return False

    def _setup_adapter(self):
        """Attach (or activate) adapter layers."""
        if AdapterConfig is None:
            print("Warning: transformers.adapters is unavailable; cannot enable adapters.")
            self.use_adapter = False
            return

        try:
            adapter_exists = self.adapter_name in self.backbone.config.adapters.adapters
        except Exception:
            adapter_exists = False

        if not adapter_exists:
            adapter_config = AdapterConfig.load(
                self.adapter_type,
                reduction_factor=self.adapter_reduction,
                dropout=self.adapter_dropout
            )
            self.backbone.add_adapter(self.adapter_name, config=adapter_config)

        self.backbone.set_active_adapters(self.adapter_name)
        if self.adapter_trainable:
            try:
                self.backbone.train_adapter(self.adapter_name)
            except Exception:
                for name, param in self.backbone.named_parameters():
                    if f"adapters.{self.adapter_name}" in name:
                        param.requires_grad = True
        else:
            for name, param in self.backbone.named_parameters():
                if f"adapters.{self.adapter_name}" in name:
                    param.requires_grad = False

    def _initialize_adapter_runtime(self):
        """Ensure adapter mixins are initialized when using standalone adapters package."""
        if ADAPTER_INITIALIZER is None:
            return
        try:
            ADAPTER_INITIALIZER(self.backbone)
        except Exception as exc:
            print(f"Warning: Failed to initialize adapter runtime: {exc}")

    def unfreeze_last_layers(self, num_layers: int = 2):
        """
        Unfreeze the last N transformer layers.

        Args:
            num_layers: Number of last layers to unfreeze
        """
        if not self.freeze:
            print("BERT is already not frozen")
            return

        # Get total number of encoder layers
        total_layers = self.config.num_hidden_layers

        # Unfreeze last N layers
        for i in range(total_layers - num_layers, total_layers):
            layer_param_names = [f"encoder.layer.{i}.{name}"
                                for name in ["attention.self.query", "attention.self.key",
                                           "attention.self.value", "attention.output.dense",
                                           "intermediate.dense", "output.dense"]]

            for param_name in layer_param_names:
                try:
                    param = dict(self.backbone.named_parameters())[param_name]
                    param.requires_grad = True
                except KeyError:
                    # Parameter name might be slightly different
                    continue

        print(f"Unfroze last {num_layers} transformer layers")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through BERT encoder.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)

        Returns:
            Text features of shape (batch_size, out_dim)
        """
        # Pass through BERT backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False
        )

        # Extract CLS token representation (first token)
        cls_features = outputs.last_hidden_state[:, 0]  # (batch_size, hidden_size)

        # Apply projection if needed
        features = self.proj(cls_features)  # (batch_size, out_dim)

        return features

    def get_feature_dim(self) -> int:
        """Get output feature dimension."""
        return self.out_dim

    def get_tokenizer(self) -> AutoTokenizer:
        """Get the corresponding tokenizer for this model."""
        try:
            return AutoTokenizer.from_pretrained(self.model_dir)
        except Exception:
            print(f"Could not load tokenizer from {self.model_dir}, using bert-base-uncased")
            return AutoTokenizer.from_pretrained("bert-base-uncased")

    def save_pretrained(self, save_directory: str):
        """Save the encoder to directory."""
        self.backbone.save_pretrained(save_directory)
        self.config.save_pretrained(save_directory)

        # Save projection layer separately if it's not identity
        if not isinstance(self.proj, nn.Identity):
            torch.save(self.proj.state_dict(),
                      f"{save_directory}/projection_state.pt")

    @classmethod
    def from_pretrained(cls, model_dir: str, **kwargs):
        """Load encoder from pretrained model directory."""
        return cls(model_dir=model_dir, **kwargs)
