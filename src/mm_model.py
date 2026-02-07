"""
Multimodal Model Integration

This module provides the complete multimodal model that integrates
visual encoder, text encoder, and fusion head for end-to-end training.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.text_encoder import BertTextEncoder
from src.vision_encoder import VitVisionEncoder
from src.fusion_head import FusionHead, LightweightFusionHead


class MultiModalModel(nn.Module):
    """
    Complete multimodal model for image-text classification.

    Combines ViT vision encoder, BERT text encoder, and fusion MLP
    for end-to-end multimodal training.
    """

    def __init__(
        self,
        # Vision encoder config
        num_classes: int = 6,
        encoder_type: str = "vit",  # "vit" or "clip"
        vision_model_name: str = "ViT-B/32",  # Model name for both ViT and CLIP
        vision_model_path: Optional[str] = None,
        freeze_vision: bool = False,
        vision_proj_dim: Optional[int] = None,

        # Text encoder config
        bert_model_dir: Optional[str] = None,
        bert_model_name: Optional[str] = None,
        freeze_bert: bool = True,
        text_proj_dim: Optional[int] = None,
        use_adapter: bool = False,
        adapter_type: str = "pfeiffer",
        adapter_reduction_factor: int = 16,
        adapter_dropout: float = 0.0,
        adapter_trainable: Optional[bool] = None,

        # Fusion head config
        fusion_dim: int = 768,
        fusion_type: str = "standard",  # "standard" or "lightweight"
        dropout: float = 0.1,

        # General config
        device: Optional[str] = None,
        visual_only: bool = False  # Pure visual classification mode
    ):
        """
        Args:
            num_classes: Number of output classes
            encoder_type: Type of vision encoder - "vit" or "clip"
            vision_model_name: Model name for both ViT and CLIP encoders
            vision_model_path: Path to custom model (if any)
            freeze_vision: Whether to freeze vision backbone
            vision_proj_dim: Optional projection dimension for vision features
            bert_model_dir: Path to fine-tuned BERT model directory
            bert_model_name: HuggingFace BERT model name (if bert_model_dir not provided)
            freeze_bert: Whether to freeze BERT parameters
            text_proj_dim: Optional projection dimension for text features
            use_adapter: Whether to attach AdapterHub adapters to the text encoder
            adapter_type: Adapter architecture (e.g., pfeiffer/houlsby)
            adapter_reduction_factor: Bottleneck factor for adapters
            adapter_dropout: Dropout probability inside adapters
            adapter_trainable: Explicit flag controlling whether adapters require grad (defaults to enabling training when adapters are active or the encoder is unfrozen)
            fusion_dim: Hidden dimension for fusion layers
            fusion_type: Type of fusion head ("standard" or "lightweight")
            dropout: Dropout rate
            device: Device to use (auto-detected if None)
            visual_only: Whether to use pure visual classification mode (frozen CLIP + linear classifier)
        """
        super().__init__()

        self.num_classes = num_classes
        self.fusion_type = fusion_type
        self.visual_only = visual_only

        # Auto-detect device if not provided
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Initializing MultiModalModel on device: {self.device}")
        print(f"Visual-only mode: {self.visual_only}")

        if self.visual_only:
            # Pure visual classification mode
            print("=== Pure Visual Classification Mode ===")
            print("Using frozen CLIP visual encoder + trainable linear classifier")

            # Initialize frozen CLIP visual encoder
            self.vision_encoder = VitVisionEncoder(
                num_classes=num_classes,
                encoder_type="clip",  # Force use CLIP for visual-only mode
                model_name=vision_model_name,
                model_path=vision_model_path,
                freeze=True,  # Freeze the visual encoder
                proj_dim=None,  # Use raw CLIP features
                dropout=dropout
            )
            self.vision_encoder.to(self.device)

            # Store model name for info printing
            self.vision_model_name = vision_model_name

            # Get CLIP feature dimension (typically 512 for CLIP ViT-B/32)
            self.d_v = self.vision_encoder.get_feature_dim()
            print(f"CLIP feature dimension: {self.d_v}")

            # Add trainable two-layer classifier on top of frozen CLIP features
            self.visual_classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.d_v, self.d_v),  # First layer: CLIP features -> hidden
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(self.d_v, num_classes)  # Second layer: hidden -> classes
            ).to(self.device)

            # Initialize text encoder as None
            self.text_encoder = None
            self.fusion_head = None

            # Only train the linear classifier
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            for param in self.visual_classifier.parameters():
                param.requires_grad = True

            print(f"Visual-only model initialized:")
            print(f"  Frozen CLIP encoder parameters: {sum(p.numel() for p in self.vision_encoder.parameters()):,}")
            print(f"  Trainable classifier parameters: {sum(p.numel() for p in self.visual_classifier.parameters()):,}")

        else:
            # Original multimodal mode
            print("=== Multimodal Mode ===")

            # Initialize vision encoder
            self.vision_encoder = VitVisionEncoder(
                num_classes=num_classes,
                encoder_type=encoder_type,
                model_name=vision_model_name,
                model_path=vision_model_path,
                freeze=freeze_vision,
                proj_dim=vision_proj_dim,
                dropout=dropout
            )
            # Move to correct device and ensure CLIP encoder gets the right device
            self.vision_encoder.to(self.device)

            # Initialize text encoder
            bert_source = bert_model_dir or bert_model_name
            if bert_source is None:
                raise ValueError("A fine-tuned text encoder directory (or HF model id) must be provided via --bert_dir or --bert_model_name.")
            self.text_encoder = BertTextEncoder(
                model_dir=bert_source,
                freeze=freeze_bert,
                proj_dim=text_proj_dim,
                dropout=dropout,
                use_adapter=use_adapter,
                adapter_type=adapter_type,
                adapter_reduction_factor=adapter_reduction_factor,
                adapter_dropout=adapter_dropout,
                adapter_trainable=adapter_trainable
                if adapter_trainable is not None
                else (use_adapter or not freeze_bert)
            ).to(self.device)

            # Get feature dimensions
            self.d_v = self.vision_encoder.get_feature_dim()
            self.d_t = self.text_encoder.get_feature_dim()

            print(f"Feature dimensions - Vision: {self.d_v}, Text: {self.d_t}")

            # Initialize fusion head
            if fusion_type == "lightweight":
                self.fusion_head = LightweightFusionHead(
                    d_v=self.d_v,
                    d_t=self.d_t,
                    num_classes=num_classes,
                    dropout=dropout
                ).to(self.device)
            else:
                self.fusion_head = FusionHead(
                    d_v=self.d_v,
                    d_t=self.d_t,
                    d_fuse=fusion_dim,
                    num_classes=num_classes,
                    dropout=dropout
                ).to(self.device)

        # Move everything to device
        self.to(self.device)

        self.print_model_info()

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through multimodal model.

        Args:
            images: Input images of shape (batch_size, 3, 224, 224)
            input_ids: Token IDs of shape (batch_size, seq_len) (only for multimodal mode)
            attention_mask: Attention mask of shape (batch_size, seq_len) (only for multimodal mode)

        Returns:
            Classification logits of shape (batch_size, num_classes)
        """
        # Inputs are already on correct device from trainer

        if self.visual_only:
            # Pure visual classification mode
            # Extract CLIP features and pass through linear classifier
            v_features = self.vision_encoder.get_image_feature(images)
            # v_features shape: (batch_size, d_v) where d_v is CLIP feature dim
            logits = self.visual_classifier(v_features)
            return logits
        else:
            # Original multimodal mode
            # Extract visual features
            v_features = self.vision_encoder.get_image_feature(images)
            # v_features shape: (batch_size, d_v)

            # Extract text features
            t_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            # t_features shape: (batch_size, d_t)

            # Fuse and classify
            logits = self.fusion_head(v_features, t_features)
            # logits shape: (batch_size, num_classes)

            return logits

    def get_features(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract individual and fused features.

        Args:
            images: Input images
            input_ids: Token IDs
            attention_mask: Attention mask

        Returns:
            Tuple of (visual_features, text_features, fused_features)
        """
        images = images.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Extract features
        v_features = self.vision_encoder.get_image_feature(images)
        t_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        fused_features = self.fusion_head.get_fused_features(v_features, t_features)

        return v_features, t_features, fused_features

    def get_probabilities(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get classification probabilities.

        Args:
            images: Input images
            input_ids: Token IDs
            attention_mask: Attention mask

        Returns:
            Classification probabilities of shape (batch_size, num_classes)
        """
        logits = self.forward(images, input_ids, attention_mask)
        return torch.softmax(logits, dim=-1)

    def get_individual_predictions(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get predictions from individual modalities and fusion.

        Args:
            images: Input images
            input_ids: Token IDs
            attention_mask: Attention mask

        Returns:
            Dictionary with predictions from each modality
        """
        images = images.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Vision-only predictions
        vision_logits = self.vision_encoder(images)

        # Text doesn't have individual classifier, so we create a simple one
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Simple linear projection for text-only prediction
        if not hasattr(self, 'text_classifier'):
            self.text_classifier = nn.Linear(self.d_t, self.num_classes).to(self.device)
        text_logits = self.text_classifier(text_features)

        # Multimodal predictions
        multimodal_logits = self.forward(images, input_ids, attention_mask)

        return {
            "vision": torch.softmax(vision_logits, dim=-1),
            "text": torch.softmax(text_logits, dim=-1),
            "multimodal": torch.softmax(multimodal_logits, dim=-1)
        }

    def unfreeze_components(
        self,
        vision: bool = False,
        text: bool = False,
        fusion: bool = True
    ):
        """
        Unfreeze specific model components.

        Args:
            vision: Whether to unfreeze vision encoder
            text: Whether to unfreeze text encoder
            fusion: Whether to unfreeze fusion head (usually True)
        """
        if vision:
            self.vision_encoder.unfreeze_all()
            print("Unfrozen vision encoder")

        if text:
            # For BERT, we need to access the backbone directly
            for param in self.text_encoder.backbone.parameters():
                param.requires_grad = True
            self.text_encoder.freeze = False
            print("Unfrozen text encoder")

        # Fusion head is typically always trainable
        if not fusion:
            for param in self.fusion_head.parameters():
                param.requires_grad = False
            print("Frozen fusion head")

    def print_model_info(self):
        """Print comprehensive model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        if self.visual_only:
            # Visual-only mode information
            vision_params = sum(p.numel() for p in self.vision_encoder.parameters())
            classifier_params = sum(p.numel() for p in self.visual_classifier.parameters())

            print(f"\n" + "="*60)
            print(f"VISUAL-ONLY MODEL ARCHITECTURE")
            print(f"="*60)
            print(f"Device: {self.device}")
            print(f"Number of classes: {self.num_classes}")
            print(f"Vision model: {getattr(self, 'vision_model_name', 'Unknown')}")
            print(f"\nComponent parameters:")
            print(f"  Frozen CLIP encoder: {vision_params:,}")
            print(f"  Trainable classifier: {classifier_params:,}")
            print(f"\nFeature dimensions:")
            print(f"  CLIP features: {self.d_v}")
            print(f"\nTotal parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Trainable ratio: {trainable_params/total_params:.2%}")
            print(f"="*60)
        else:
            # Original multimodal information
            vision_params = sum(p.numel() for p in self.vision_encoder.parameters())
            text_params = sum(p.numel() for p in self.text_encoder.parameters())
            fusion_params = sum(p.numel() for p in self.fusion_head.parameters())

            print(f"\n" + "="*60)
            print(f"MULTIMODAL MODEL ARCHITECTURE")
            print(f"="*60)
            print(f"Device: {self.device}")
            print(f"Number of classes: {self.num_classes}")
            print(f"Fusion type: {self.fusion_type}")
            print(f"\nComponent parameters:")
            print(f"  Vision encoder: {vision_params:,}")
            print(f"  Text encoder: {text_params:,}")
            print(f"  Fusion head: {fusion_params:,}")
            print(f"\nFeature dimensions:")
            print(f"  Vision: {self.d_v}")
            print(f"  Text: {self.d_t}")
            print(f"\nTotal parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Trainable ratio: {trainable_params/total_params:.2%}")
            print(f"="*60)

    def save_model(self, save_path: str):
        """Save the complete multimodal model."""
        import os
        import json

        os.makedirs(save_path, exist_ok=True)

        # Save each component
        self.vision_encoder.save_model(f"{save_path}/vision_encoder")
        self.text_encoder.save_pretrained(f"{save_path}/text_encoder")
        self.fusion_head.save_model(f"{save_path}/fusion_head")

        # Save overall model metadata
        metadata = {
            "num_classes": self.num_classes,
            "d_v": self.d_v,
            "d_t": self.d_t,
            "fusion_type": self.fusion_type,
            "device": str(self.device),
            "model_class": "MultiModalModel"
        }

        with open(f"{save_path}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"MultiModalModel saved to {save_path}")

    @classmethod
    def load_model(cls, load_path: str, device: Optional[str] = None):
        """Load multimodal model from saved state."""
        import json

        # Load metadata
        with open(f"{load_path}/metadata.json", "r") as f:
            metadata = json.load(f)

        # Create model (will use saved component configs)
        model = cls(
            num_classes=metadata["num_classes"],
            fusion_type=metadata.get("fusion_type", "standard"),
            device=device
        )

        # Load components
        model.vision_encoder = VitVisionEncoder.load_model(f"{load_path}/vision_encoder")
        # Note: Text encoder loading would need to be implemented
        # model.text_encoder = BertTextEncoder.from_pretrained(f"{load_path}/text_encoder")
        model.fusion_head = FusionHead.load_model(f"{load_path}/fusion_head")

        print(f"MultiModalModel loaded from {load_path}")
        return model


if __name__ == "__main__":
    # Test the multimodal model
    print("Testing MultiModalModel...")

    # Create model
    model = MultiModalModel(
        num_classes=6,
        freeze_vision=False,
        freeze_bert=True,
        fusion_dim=512,
        fusion_type="standard"
    )

    # Test with dummy inputs
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    seq_len = 128
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    print(f"\nTesting with:")
    print(f"  Images shape: {images.shape}")
    print(f"  Input IDs shape: {input_ids.shape}")
    print(f"  Attention mask shape: {attention_mask.shape}")

    # Forward pass
    with torch.no_grad():
        logits = model(images, input_ids, attention_mask)
        print(f"Classification logits shape: {logits.shape}")

        # Test probabilities
        probs = model.get_probabilities(images, input_ids, attention_mask)
        print(f"Probabilities shape: {probs.shape}")
        print(f"Probability sums: {probs.sum(dim=1)}")

        # Test feature extraction
        v_feat, t_feat, fused_feat = model.get_features(images, input_ids, attention_mask)
        print(f"Visual features shape: {v_feat.shape}")
        print(f"Text features shape: {t_feat.shape}")
        print(f"Fused features shape: {fused_feat.shape}")

        # Test individual predictions
        predictions = model.get_individual_predictions(images, input_ids, attention_mask)
        print(f"Available predictions: {list(predictions.keys())}")

    print("\nMultiModalModel test completed successfully!")