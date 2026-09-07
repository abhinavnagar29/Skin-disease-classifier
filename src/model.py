"""
Model architecture for the Hybrid Skin Disease Classifier.

This is the EXISTING, ALREADY-TRAINED architecture. Nothing here is new —
it is extracted verbatim from the deployed app so that evaluation,
benchmarking, and interpretability scripts can all import a single
source of truth instead of redefining the model in five places.

Architecture summary:
    ResNet50 (FC stripped)      -> GAP -> 2048-d  \
                                                     concat -> 3072-d -> MLP head -> logits
    DenseNet121 (classifier stripped) -> ReLU -> GAP -> 1024-d /

Fusion is feature-level concatenation, not a learned gating/attention
mechanism. State it that way in writing/interviews — it's a legitimate,
defensible design choice (simplicity, no extra fusion params to overfit
on top of two large pretrained backbones), not a shortcoming to hide.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class HybridBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet50(weights=None)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2])
        self.resnet_target_layer = self.resnet_features[-1]  # used by Grad-CAM

        densenet = models.densenet121(weights=None)
        self.densenet_features = densenet.features
        self.densenet_target_layer = self.densenet_features.denseblock4

    def forward(self, x):
        r = self.resnet_features(x)
        r = F.adaptive_avg_pool2d(r, (1, 1))
        r = torch.flatten(r, 1)

        d = self.densenet_features(x)
        d = F.relu(d, inplace=False)
        d = F.adaptive_avg_pool2d(d, (1, 1))
        d = torch.flatten(d, 1)

        return torch.cat([r, d], dim=1)


class HybridSkinClassifier(nn.Module):
    def __init__(self, num_classes=22, dropout_rate=0.5):
        super().__init__()

        self.backbone = HybridBackbone()

        self.classifier = nn.ModuleDict({
            "net": nn.Sequential(
                nn.Linear(3072, 2048), nn.BatchNorm1d(2048), nn.ReLU(), nn.Dropout(dropout_rate),
                nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(dropout_rate),
                nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
        })

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier["net"](features)


def count_parameters(model: nn.Module):
    """Returns (total_params, trainable_params). Run this after loading
    the checkpoint to get real numbers for the README/resume — don't
    estimate them by hand."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model: nn.Module):
    """Approximate in-memory parameter size in MB (float32 params)."""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 ** 2)
