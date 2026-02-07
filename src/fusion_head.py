"""
Fusion Head for Multimodal Training

This module provides the fusion MLP that combines visual and text features
for classification in multimodal training pipelines.
"""

import torch
import torch.nn as nn
from typing import Optional


class FusionHead(nn.Module):
    """
    Two-layer MLP fusion head for multimodal feature combination.

    Concatenates visual and text features, then passes through
    a 2-layer MLP before final classification.
    """

    def __init__(
        self,
        d_v: int = 768,      # Visual feature dimension
        d_t: int = 768,      # Text feature dimension
        d_fuse: int = 768,   # Fusion hidden dimension
        num_classes: int = 6, # Number of output classes
        dropout: float = 0.1, # Dropout rate
        activation: str = "gelu"  # Activation function
    ):
        """
        Args:
            d_v: Visual feature dimension (from ViT)
            d_t: Text feature dimension (from BERT)
            d_fuse: Hidden dimension for fusion layers
            num_classes: Number of output classes
            dropout: Dropout rate
            activation: Activation function ('gelu', 'relu', 'leaky_relu')
        """
        super().__init__()

        self.d_v = d_v
        self.d_t = d_t
        self.d_fuse = d_fuse
        self.num_classes = num_classes

        # Choose activation function
        if activation.lower() == "gelu":
            act_fn = nn.GELU()
        elif activation.lower() == "relu":
            act_fn = nn.ReLU()
        elif activation.lower() == "leaky_relu":
            act_fn = nn.LeakyReLU(negative_slope=0.1)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Two-layer MLP for fusion
        self.mlp = nn.Sequential(
            nn.Linear(d_v + d_t, d_fuse),  # Concatenate visual + text
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_fuse, d_fuse),
            act_fn,
            nn.Dropout(dropout)
        )

        # Final classification layer
        self.classifier = nn.Linear(d_fuse, num_classes)

        # Initialize weights
        self._initialize_weights()

        print(f"FusionHead initialized:")
        print(f"  Visual dim: {d_v}")
        print(f"  Text dim: {d_t}")
        print(f"  Fusion dim: {d_fuse}")
        print(f"  Num classes: {num_classes}")
        print(f"  Activation: {activation}")
        print(f"  Dropout: {dropout}")

    def _initialize_weights(self):
        """Initialize MLP weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, v_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through fusion head.

        Args:
            v_feat: Visual features of shape (batch_size, d_v)
            t_feat: Text features of shape (batch_size, d_t)

        Returns:
            Classification logits of shape (batch_size, num_classes)
        """
        # Ensure features have correct shapes
        assert v_feat.dim() == 2, f"Visual features should be 2D, got {v_feat.dim()}"
        assert t_feat.dim() == 2, f"Text features should be 2D, got {t_feat.dim()}"
        assert v_feat.size(0) == t_feat.size(0), "Batch sizes must match"
        assert v_feat.size(1) == self.d_v, f"Visual feature dimension mismatch: {v_feat.size(1)} != {self.d_v}"
        assert t_feat.size(1) == self.d_t, f"Text feature dimension mismatch: {t_feat.size(1)} != {self.d_t}"

        # Concatenate features along feature dimension
        z = torch.cat([v_feat, t_feat], dim=1)  # (batch_size, d_v + d_t)

        # Pass through fusion MLP
        h = self.mlp(z)  # (batch_size, d_fuse)

        # Final classification
        logits = self.classifier(h)  # (batch_size, num_classes)

        return logits

    def get_fused_features(self, v_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        """
        Get fused features before final classification.

        Args:
            v_feat: Visual features of shape (batch_size, d_v)
            t_feat: Text features of shape (batch_size, d_t)

        Returns:
            Fused features of shape (batch_size, d_fuse)
        """
        z = torch.cat([v_feat, t_feat], dim=1)
        h = self.mlp(z)
        return h

    def get_probabilities(self, v_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        """
        Get classification probabilities.

        Args:
            v_feat: Visual features of shape (batch_size, d_v)
            t_feat: Text features of shape (batch_size, d_t)

        Returns:
            Classification probabilities of shape (batch_size, num_classes)
        """
        logits = self.forward(v_feat, t_feat)
        return torch.softmax(logits, dim=-1)

    def print_model_info(self):
        """Print model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\nFusionHead Info:")
        print(f"  Input dimensions: Visual={self.d_v}, Text={self.d_t}")
        print(f"  Fusion dimension: {self.d_fuse}")
        print(f"  Output classes: {self.num_classes}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

    def save_model(self, save_path: str):
        """Save the fusion head state."""
        import os
        import json

        os.makedirs(save_path, exist_ok=True)

        # Save model state
        torch.save(self.state_dict(), f"{save_path}/fusion_head.pt")

        # Save metadata
        metadata = {
            "d_v": self.d_v,
            "d_t": self.d_t,
            "d_fuse": self.d_fuse,
            "num_classes": self.num_classes,
            "model_class": "FusionHead"
        }

        with open(f"{save_path}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"FusionHead saved to {save_path}")

    @classmethod
    def load_model(cls, load_path: str):
        """Load fusion head from saved state."""
        import json

        # Load metadata
        with open(f"{load_path}/metadata.json", "r") as f:
            metadata = json.load(f)

        # Create model
        model = cls(
            d_v=metadata["d_v"],
            d_t=metadata["d_t"],
            d_fuse=metadata["d_fuse"],
            num_classes=metadata["num_classes"]
        )

        # Load state
        model.load_state_dict(torch.load(f"{load_path}/fusion_head.pt"))
        print(f"FusionHead loaded from {load_path}")

        return model


class LightweightFusionHead(nn.Module):
    """
    Lightweight fusion head with single layer for faster training.
    """

    def __init__(
        self,
        d_v: int = 768,
        d_t: int = 768,
        num_classes: int = 6,
        dropout: float = 0.1
    ):
        """
        Args:
            d_v: Visual feature dimension
            d_t: Text feature dimension
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()

        self.d_v = d_v
        self.d_t = d_t
        self.num_classes = num_classes

        # Single layer fusion (more efficient)
        self.fusion = nn.Sequential(
            nn.Linear(d_v + d_t, d_v + d_t),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Classification layer
        self.classifier = nn.Linear(d_v + d_t, num_classes)

        self._initialize_weights()

        print(f"LightweightFusionHead initialized:")
        print(f"  Visual dim: {d_v}")
        print(f"  Text dim: {d_t}")
        print(f"  Combined dim: {d_v + d_t}")
        print(f"  Num classes: {num_classes}")

    def _initialize_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, v_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        z = torch.cat([v_feat, t_feat], dim=1)
        h = self.fusion(z)
        logits = self.classifier(h)
        return logits

    def get_fused_features(self, v_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        """Get fused features before classification."""
        z = torch.cat([v_feat, t_feat], dim=1)
        return self.fusion(z)


if __name__ == "__main__":
    # Test the fusion head
    print("Testing FusionHead...")

    # Create fusion head
    fusion_head = FusionHead(
        d_v=768,
        d_t=512,
        d_fuse=768,
        num_classes=6,
        dropout=0.1
    )

    fusion_head.print_model_info()

    # Test with dummy inputs
    batch_size = 8
    visual_features = torch.randn(batch_size, 768)
    text_features = torch.randn(batch_size, 512)

    print(f"\nTesting with:")
    print(f"  Visual features shape: {visual_features.shape}")
    print(f"  Text features shape: {text_features.shape}")

    # Forward pass
    with torch.no_grad():
        logits = fusion_head(visual_features, text_features)
        print(f"Classification logits shape: {logits.shape}")

        # Test fused features
        fused_features = fusion_head.get_fused_features(visual_features, text_features)
        print(f"Fused features shape: {fused_features.shape}")

        # Test probabilities
        probs = fusion_head.get_probabilities(visual_features, text_features)
        print(f"Probabilities shape: {probs.shape}")
        print(f"Sum of probabilities: {probs.sum(dim=1)}")

    print("\nTesting LightweightFusionHead...")
    lightweight_head = LightweightFusionHead(
        d_v=768,
        d_t=512,
        num_classes=6
    )

    with torch.no_grad():
        logits_lite = lightweight_head(visual_features, text_features)
        print(f"Lightweight logits shape: {logits_lite.shape}")

    print("\nFusionHead test completed successfully!")