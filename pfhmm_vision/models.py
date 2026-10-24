"""Backbone providers and zero-initialized residual classification adapters."""

import torch
from torch import nn
from torchvision import models


def make_backbone(config, classes):
    """Build an explicit architecture; shared source pretraining supplies all initial weights."""
    providers = {"resnet18": models.resnet18, "resnet34": models.resnet34}
    if config.architecture not in providers:
        raise ValueError(f"Unknown architecture: {config.architecture}")
    model = providers[config.architecture](weights=None, num_classes=classes)
    if config.small_stem:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
        model.maxpool = nn.Identity()
    return model


def encode(model, images):
    """Extract pooled ResNet representations and unchanged reference classification logits."""
    hidden = model.relu(model.bn1(model.conv1(images)))
    hidden = model.maxpool(hidden)
    hidden = model.layer4(model.layer3(model.layer2(model.layer1(hidden))))
    features = torch.flatten(model.avgpool(hidden), 1)
    return features, model.fc(features)


class ResidualAdapter(nn.Module):
    """Change logits through a small trainable module while retaining a frozen visual encoder."""

    def __init__(self, feature_dim, hidden_dim, classes):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim),
                                     nn.GELU(), nn.Linear(hidden_dim, classes))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features, reference_logits):
        """Start exactly at the reference model, then add a learned logit residual."""
        return reference_logits + self.network(features)


def make_adapter(config, features, logits):
    """Infer interface dimensions from encoder outputs instead of a fixed class count."""
    if features.ndim != 2 or logits.ndim != 2 or len(features) != len(logits):
        raise ValueError("Expected aligned feature and reference-logit matrices")
    return ResidualAdapter(features.shape[1], config.adapter_hidden, logits.shape[1])
