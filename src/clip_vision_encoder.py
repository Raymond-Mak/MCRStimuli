"""
OpenAI CLIP Vision Encoder for Multimodal Training

This module provides a CLIP visual encoder that uses the original OpenAI CLIP
implementation with forced FP32 precision for training compatibility.
Maintains full compatibility with VitVisionEncoder interface.
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
from typing import Optional, Dict

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import OpenAI CLIP from project's local copy
try:
    import clip
    from utils.model_utils import load_clip_to_cpu
    OPENAI_CLIP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: OpenAI CLIP not available: {e}")
    print("Please ensure the 'clip' directory exists in project root")
    OPENAI_CLIP_AVAILABLE = False


class ClipVisionEncoder(nn.Module):
    """
    CLIP vision encoder using original OpenAI implementation with FP32 precision.

    Extracts pooled visual features from CLIP for multimodal training.
    Maintains the same API as VitVisionEncoder for seamless integration.
    """

    def __init__(
        self,
        num_classes: int = 6,  # Default for Emotion6
        model_name: str = "ViT-B/32",  # CLIP model name
        model_path: Optional[str] = None,  # For compatibility, unused in OpenAI CLIP
        freeze: bool = False,
        proj_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Initialize OpenAI CLIP Vision Encoder with forced FP32 precision.

        Args:
            num_classes: Number of output classes for classification head
            model_name: CLIP model name (e.g., "ViT-B/32", "ViT-B/16", "ViT-L/14")
            model_path: Path to custom model (for compatibility, unused in OpenAI CLIP)
            freeze: Whether to freeze CLIP backbone parameters
            proj_dim: Optional projection dimension (if None, use CLIP feature dim)
            dropout: Dropout rate for projection layer
        """
        super().__init__()

        if not OPENAI_CLIP_AVAILABLE:
            raise ImportError("OpenAI CLIP is required but not available")

        self.num_classes = num_classes
        self.model_name = model_name
        self.model_path = model_path  # Keep for compatibility
        self.freeze = freeze

        # Validate model name
        available_models = clip.available_models()
        if model_name not in available_models:
            raise ValueError(f"Unsupported CLIP model: {model_name}. Available: {available_models}")

        # Load OpenAI CLIP model with forced FP32 precision
        self._load_openai_clip_model_fp32(model_name, proj_dim, dropout, num_classes, freeze)

    def _load_openai_clip_model_fp32(self, model_name: str, proj_dim: Optional[int], dropout: float, num_classes: int, freeze: bool):
        """Load OpenAI CLIP model with forced FP32 precision."""

        # Mock config for CLIP loading (compatible with existing utils)
        class MockConfig:
            def __init__(self, model_name):
                self.MODEL = type('MODEL', (), {})()
                self.MODEL.BACKBONE = type('BACKBONE', (), {})()
                self.MODEL.BACKBONE.NAME = model_name
                self.MODEL.WEIGHT_ROOT = "all_weights"

        cfg = MockConfig(model_name)

        try:
            # Load CLIP model using existing utility function
            print(f"🔄 Loading OpenAI CLIP model: {model_name}")
            self.clip_model = load_clip_to_cpu(cfg)

            # Extract the visual encoder from CLIP
            self.visual = self.clip_model.visual

            # 🔥 CRITICAL: Force FP32 precision for training compatibility
            print("📊 Converting CLIP model to FP32 precision...")
            self.clip_model = self.clip_model.float()
            self.visual = self.visual.float()

            # Verify FP32 conversion
            visual_dtype = next(self.visual.parameters()).dtype
            print(f"✅ CLIP visual encoder precision: {visual_dtype}")

            if visual_dtype != torch.float32:
                raise RuntimeError(f"Failed to convert to FP32. Current dtype: {visual_dtype}")

        except Exception as e:
            raise RuntimeError(f"Failed to load OpenAI CLIP model {model_name}: {e}")

        # Determine device - CLIP loads on CPU by default
        self.device = next(self.visual.parameters()).device

        # Get CLIP visual feature dimension
        # For ViT-B/32: 512, for ViT-B/16: 512, for ViT-L/14: 768
        self.clip_feature_dim = self.visual.output_dim

        # Set output dimension (ensure compatibility with existing fusion heads)
        if proj_dim is not None:
            self.out_dim = proj_dim
        else:
            # Project to 768 to match ViT-base and existing fusion heads
            self.out_dim = 768 if self.clip_feature_dim != 768 else self.clip_feature_dim

        # Setup projection layers if needed
        if self.clip_feature_dim != self.out_dim:
            self.proj = nn.Sequential(
                nn.Linear(self.clip_feature_dim, self.out_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            print(f"🔧 Added projection: {self.clip_feature_dim} -> {self.out_dim}")
        else:
            self.proj = nn.Identity()
            print(f"🔧 No projection needed (CLIP dim: {self.clip_feature_dim})")

        # Classification head (for compatibility with single-modality evaluation)
        self.classifier = nn.Linear(self.out_dim, num_classes)

        # Freeze parameters if requested
        if freeze:
            self._freeze_parameters()

        print(f"✅ OpenAI CLIP Vision Encoder initialized:")
        print(f"  Model name: {model_name}")
        print(f"  CLIP feature dim: {self.clip_feature_dim}")
        print(f"  Output dim: {self.out_dim}")
        print(f"  Num classes: {num_classes}")
        print(f"  Frozen: {freeze}")
        print(f"  Device: {self.device}")
        print(f"  Precision: {visual_dtype}")

    def _freeze_parameters(self):
        """Freeze CLIP visual encoder parameters."""
        for param in self.visual.parameters():
            param.requires_grad = False
        # Keep projection and classifier trainable
        for param in self.proj.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        print("🧊 CLIP visual encoder frozen, projection and classifier remain trainable")

    def unfreeze_all(self):
        """Unfreeze all CLIP parameters."""
        for param in self.visual.parameters():
            param.requires_grad = True
        self.freeze = False
        print("🔥 All CLIP visual encoder parameters unfrozen")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CLIP encoder for classification.

        Args:
            images: Input images of shape (batch_size, 3, 224, 224)

        Returns:
            Classification logits of shape (batch_size, num_classes)
        """
        # Extract features first, then classify
        features = self.get_image_feature(images)
        return self.classifier(features)

    def get_image_feature(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract pooled visual features from CLIP for multimodal fusion.

        Args:
            images: Input images of shape (batch_size, 3, 224, 224)

        Returns:
            Visual features of shape (batch_size, out_dim)
        """
        # Images are already on correct device from trainer

        # Extract features from CLIP visual encoder
        # Ensure precision alignment: use FP32 inputs for OpenAI CLIP visual encoder
        if images.dtype != torch.float32:
            images = images.float()

        with torch.set_grad_enabled(not self.freeze):
            # OpenAI CLIP visual encoder returns pooled features directly
            clip_features = self.visual(images)

            # clip_features shape: (batch_size, clip_feature_dim)

            # Apply projection if needed to match expected dimension
            features = self.proj(clip_features)
            # features shape: (batch_size, out_dim)

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
        Get classification logits from CLIP encoder.

        Args:
            images: Input images of shape (batch_size, 3, 224, 224)

        Returns:
            Classification logits of shape (batch_size, num_classes)
        """
        return self.forward(images)

    def get_classifier_probabilities(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get classification probabilities from CLIP encoder.

        Args:
            images: Input images of shape (batch_size, 3, 224, 224)

        Returns:
            Classification probabilities of shape (batch_size, num_classes)
        """
        logits = self.forward(images)
        return torch.softmax(logits, dim=-1)

    def save_model(self, save_path: str):
        """Save the CLIP model state."""
        import os
        os.makedirs(save_path, exist_ok=True)

        # Save CLIP visual encoder state (FP32)
        torch.save(self.visual.state_dict(), f"{save_path}/clip_visual_encoder_fp32.pt")

        # Save projection and classifier states
        if not isinstance(self.proj, nn.Identity):
            torch.save(self.proj.state_dict(), f"{save_path}/projection_state.pt")

        torch.save(self.classifier.state_dict(), f"{save_path}/classifier_state.pt")

        # Save metadata
        metadata = {
            "num_classes": self.num_classes,
            "out_dim": self.out_dim,
            "clip_feature_dim": self.clip_feature_dim,
            "model_name": self.model_name,
            "freeze": self.freeze,
            "model_path": self.model_path,
            "encoder_type": "clip_openai",
            "precision": "fp32"
        }

        import json
        with open(f"{save_path}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ OpenAI CLIP Vision Encoder (FP32) saved to {save_path}")

    @classmethod
    def load_model(cls, load_path: str, **kwargs):
        """Load CLIP model from saved state."""
        import json
        import os

        # Load metadata
        with open(f"{load_path}/metadata.json", "r") as f:
            metadata = json.load(f)

        # Create encoder
        encoder = cls(
            num_classes=metadata["num_classes"],
            model_name=metadata["model_name"],
            model_path=metadata.get("model_path"),
            freeze=metadata["freeze"],
            **kwargs
        )

        # Load CLIP visual encoder state
        encoder.visual.load_state_dict(
            torch.load(f"{load_path}/clip_visual_encoder_fp32.pt", map_location=encoder.device)
        )

        # Load projection state if exists
        proj_path = f"{load_path}/projection_state.pt"
        if os.path.exists(proj_path):
            encoder.proj.load_state_dict(torch.load(proj_path, map_location=encoder.device))

        # Load classifier state
        encoder.classifier.load_state_dict(
            torch.load(f"{load_path}/classifier_state.pt", map_location=encoder.device)
        )

        print(f"✅ OpenAI CLIP Vision Encoder (FP32) loaded from {load_path}")
        return encoder

    def to(self, device):
        """Override to() method to properly move CLIP model to device and maintain FP32."""
        super().to(device)
        self.device = torch.device(device)

        # Move CLIP model to device and maintain FP32 precision
        if hasattr(self, 'clip_model'):
            self.clip_model = self.clip_model.to(device)
            # Ensure FP32 precision after device move
            self.clip_model = self.clip_model.float()
            self.visual = self.clip_model.visual.float()

        return self

    def print_model_info(self):
        """Print model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        current_dtype = next(self.visual.parameters()).dtype

        print(f"\n🎯 OpenAI CLIP Vision Encoder Info:")
        print(f"  Model name: {self.model_name}")
        print(f"  CLIP feature dimension: {self.clip_feature_dim}")
        print(f"  Output dimension: {self.out_dim}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen: {self.freeze}")
        print(f"  Device: {self.device}")
        print(f"  Num classes: {self.num_classes}")
        print(f"  Precision: {current_dtype}")
        print(f"  🎮 Implementation: OpenAI CLIP (FP32 forced)")


if __name__ == "__main__":
    # Test the OpenAI CLIP encoder
    print("🧪 Testing OpenAI CLIP Vision Encoder (FP32)...")

    try:
        # Create encoder
        encoder = ClipVisionEncoder(
            num_classes=6,
            model_name="ViT-B/32",
            freeze=True,
            proj_dim=768  # Ensure compatibility with existing fusion heads
        )

        encoder.print_model_info()

        # Test with dummy input
        batch_size = 4
        dummy_images = torch.randn(batch_size, 3, 224, 224)

        print(f"\n🔍 Testing with input shape: {dummy_images.shape}")

        # Test forward pass (classification)
        with torch.no_grad():
            logits = encoder(dummy_images)
            print(f"✅ Classification logits shape: {logits.shape}")

            # Test feature extraction
            features = encoder.get_image_feature(dummy_images)
            print(f"✅ Image features shape: {features.shape}")

            # Test probabilities
            probs = encoder.get_classifier_probabilities(dummy_images)
            print(f"✅ Probabilities shape: {probs.shape}")
            print(f"📊 Sum of probabilities (should be 1.0): {probs.sum(dim=1)}")

        print(f"\n🎯 Feature dimension: {encoder.get_feature_dim()}")
        print("🎉 OpenAI CLIP Vision Encoder test completed successfully!")

        # Test device movement
        if torch.cuda.is_available():
            print("\n🔄 Testing GPU movement...")
            encoder = encoder.to('cuda')
            dummy_images_cuda = dummy_images.to('cuda')

            with torch.no_grad():
                features_cuda = encoder.get_image_feature(dummy_images_cuda)
                print(f"✅ GPU features shape: {features_cuda.shape}")
                print(f"📊 GPU precision: {features_cuda.dtype}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
