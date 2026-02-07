"""
Vision Encoder for Multimodal Training

This module provides vision encoder wrappers for extracting visual features
in multimodal training pipelines. Supports both ViT and CLIP backends
through a unified interface.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import Optional, Dict

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trainers.vit_trainer import ViTModel, VIT_HIDDEN_SIZE


class VitVisionEncoder(nn.Module):
    """
    Universal vision encoder wrapper that supports both ViT and CLIP backends.

    Extracts pooled features from vision encoder for multimodal training.
    Maintains backward compatibility while adding CLIP support.
    """

    def __init__(
        self,
        num_classes: int = 6,  # Default for Emotion6
        encoder_type: str = "vit",  # "vit" or "clip"
        model_name: str = "ViT-B/32",  # Model name for both ViT and CLIP
        model_path: Optional[str] = None,
        freeze: bool = False,
        proj_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Args:
            num_classes: Number of output classes for classification
            encoder_type: Type of encoder - "vit" or "clip"
            model_name: Model name (ViT variant or CLIP variant)
            model_path: Path to custom model (if None, uses default)
            freeze: Whether to freeze backbone parameters
            proj_dim: Optional projection dimension (if None, use default)
            dropout: Dropout rate for projection layer
        """
        super().__init__()

        self.num_classes = num_classes
        self.encoder_type = encoder_type.lower()
        self.model_name = model_name
        self.model_path = model_path
        self.freeze = freeze

        # Initialize the appropriate encoder
        if self.encoder_type == "clip":
            # Import CLIP encoder dynamically
            try:
                from .clip_vision_encoder import ClipVisionEncoder
                self._encoder = ClipVisionEncoder(
                    num_classes=num_classes,
                    model_name=model_name,
                    model_path=model_path,
                    freeze=freeze,
                    proj_dim=proj_dim,
                    dropout=dropout
                )
                self.device = self._encoder.device
                self.out_dim = self._encoder.get_feature_dim()
                self.proj = nn.Identity()  # CLIP encoder handles its own projection
                print(f"Using CLIP vision encoder: {model_name}")
            except ImportError as e:
                raise ImportError(f"Failed to import CLIP encoder: {e}")
        else:
            # Use original ViT implementation
            self.vit_model = ViTModel(num_labels=num_classes)

            # Determine device
            self.device = next(self.vit_model.parameters()).device

            # Output dimension (ViT hidden size)
            self.out_dim = VIT_HIDDEN_SIZE  # 768 for ViT-base

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

            print(f"Using ViT vision encoder: {model_name}")

        print(f"Vision Encoder initialized:")
        print(f"  Type: {self.encoder_type}")
        print(f"  Model: {model_name}")
        print(f"  Num classes: {num_classes}")
        print(f"  Output dim: {self.out_dim}")
        print(f"  Frozen: {freeze}")

    def _freeze_parameters(self):
        """Freeze backbone parameters."""
        if self.encoder_type == "clip":
            self._encoder._freeze_parameters()
        else:
            self.vit_model.freeze_backbone()

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        if self.encoder_type == "clip":
            self._encoder.unfreeze_all()
        else:
            self.vit_model.unfreeze_all()
        self.freeze = False
        print(f"All {self.encoder_type} parameters unfrozen")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through vision encoder.

        Args:
            images: Input images of shape (batch_size, 3, 224, 224)

        Returns:
            Classification logits of shape (batch_size, num_classes)
        """
        if self.encoder_type == "clip":
            return self._encoder(images)
        else:
            return self.vit_model(images)

    def get_image_feature(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract pooled visual features for multimodal fusion.

        Args:
            images: Input images of shape (batch_size, 3, 224, 224)

        Returns:
            Visual features of shape (batch_size, out_dim)
        """
        if self.encoder_type == "clip":
            return self._encoder.get_image_feature(images)
        else:
            # Get ViT model outputs
            with torch.set_grad_enabled(not self.freeze):
                # Get hidden states from ViT
                vit_outputs = self.vit_model.vit_model(
                    images,
                    output_hidden_states=True,
                    output_attentions=False
                )

                # Extract CLS token from last hidden state
                # ViT outputs.hidden_states[-1] has shape (batch_size, seq_len, hidden_size)
                cls_features = vit_outputs.hidden_states[-1][:, 0, :]  # CLS token

                # Apply projection if needed
                features = self.proj(cls_features)

            return features

    def get_pooled_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Alternative method to get pooled features (alias for get_image_feature).

        Args:
            images: Input images of shape (batch_size, 3, 224, 224)

        Returns:
            Pooled features of shape (batch_size, out_dim)
        """
        return self.get_image_feature(images)

    def get_feature_dim(self) -> int:
        """Get output feature dimension."""
        return self.out_dim

    def get_classifier_logits(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get classification logits from ViT.

        Args:
            images: Input images of shape (batch_size, 3, 224, 224)

        Returns:
            Classification logits of shape (batch_size, num_classes)
        """
        return self.forward(images)

    def get_classifier_probabilities(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get classification probabilities from vision encoder.

        Args:
            images: Input images of shape (batch_size, 3, 224, 224)

        Returns:
            Classification probabilities of shape (batch_size, num_classes)
        """
        if self.encoder_type == "clip":
            return self._encoder.get_classifier_probabilities(images)
        else:
            return self.vit_model.get_probabilities(images)

    def save_model(self, save_path: str):
        """Save the model state."""
        if self.encoder_type == "clip":
            self._encoder.save_model(save_path)
        else:
            # Create directory if it doesn't exist
            import os
            os.makedirs(save_path, exist_ok=True)

            # Save ViT model
            torch.save(self.vit_model.state_dict(), f"{save_path}/vit_model.pt")

            # Save projection layer if it's not identity
            if not isinstance(self.proj, nn.Identity):
                torch.save(self.proj.state_dict(), f"{save_path}/projection_state.pt")

            # Save metadata
            metadata = {
                "num_classes": self.num_classes,
                "out_dim": self.out_dim,
                "freeze": self.freeze,
                "model_path": self.model_path,
                "encoder_type": "vit"
            }

            import json
            with open(f"{save_path}/metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"ViT model saved to {save_path}")

    @classmethod
    def load_model(cls, load_path: str, **kwargs):
        """Load model from saved state."""
        import json
        import os

        # Load metadata
        with open(f"{load_path}/metadata.json", "r") as f:
            metadata = json.load(f)

        # Check encoder type
        encoder_type = metadata.get("encoder_type", "vit")

        if encoder_type == "clip":
            # Load CLIP encoder
            from .clip_vision_encoder import ClipVisionEncoder
            encoder = ClipVisionEncoder.load_model(load_path, **kwargs)
            # Wrap in VitVisionEncoder for compatibility
            wrapper = cls(num_classes=metadata["num_classes"], encoder_type="clip")
            wrapper._encoder = encoder
            wrapper.device = encoder.device
            wrapper.out_dim = encoder.get_feature_dim()
            wrapper.proj = nn.Identity()
            return wrapper
        else:
            # Create ViT encoder
            encoder = cls(
                num_classes=metadata["num_classes"],
                encoder_type="vit",
                model_path=metadata.get("model_path"),
                freeze=metadata["freeze"],
                **kwargs
            )

            # Load ViT model state
            encoder.vit_model.load_state_dict(
                torch.load(f"{load_path}/vit_model.pt", map_location=encoder.device)
            )

            # Load projection state if exists
            proj_path = f"{load_path}/projection_state.pt"
            if os.path.exists(proj_path):
                encoder.proj.load_state_dict(torch.load(proj_path, map_location=encoder.device))

            print(f"ViT model loaded from {load_path}")
            return encoder

    def print_model_info(self):
        """Print model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        encoder_name = "CLIP Vision" if self.encoder_type == "clip" else "ViT Vision"
        print(f"\n{encoder_name} Encoder Info:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen: {self.freeze}")
        print(f"  Feature dimension: {self.out_dim}")
        print(f"  Device: {self.device}")
        print(f"  Num classes: {self.num_classes}")


class PretrainedViTEncoder(VitVisionEncoder):
    """
    ViT encoder that loads from HuggingFace hub instead of local model.
    Alternative implementation for flexibility.
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        num_classes: int = 6,
        freeze: bool = False,
        proj_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Args:
            model_name: HuggingFace model name
            num_classes: Number of output classes
            freeze: Whether to freeze ViT backbone
            proj_dim: Optional projection dimension
            dropout: Dropout rate for projection layer
        """
        # This would require implementing a new ViT model using HuggingFace directly
        # For now, we'll use the existing local implementation
        super().__init__(
            num_classes=num_classes,
            model_path=None,  # Use default
            freeze=freeze,
            proj_dim=proj_dim,
            dropout=dropout
        )

        print(f"Using pretrained ViT: {model_name} (via local implementation)")


if __name__ == "__main__":
    # Test the ViT encoder
    print("Testing ViT Vision Encoder...")

    # Create encoder
    encoder = VitVisionEncoder(
        num_classes=6,
        freeze=True,
        proj_dim=512
    )

    encoder.print_model_info()

    # Test with dummy input
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224)

    print(f"\nTesting with input shape: {dummy_images.shape}")

    # Test forward pass (classification)
    with torch.no_grad():
        logits = encoder(dummy_images)
        print(f"Classification logits shape: {logits.shape}")

        # Test feature extraction
        features = encoder.get_image_feature(dummy_images)
        print(f"Image features shape: {features.shape}")

        # Test probabilities
        probs = encoder.get_classifier_probabilities(dummy_images)
        print(f"Probabilities shape: {probs.shape}")
        print(f"Sum of probabilities (should be 1.0): {probs.sum(dim=1)}")

    print(f"\nFeature dimension: {encoder.get_feature_dim()}")
    print("\nViT Vision Encoder test completed successfully!")