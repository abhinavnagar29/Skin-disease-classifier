"""
Preprocessing built directly from the checkpoint's stored config.

The checkpoint stores its own `preprocessing` dict (image_size, mean, std).
This module never hardcodes those values elsewhere — every script imports
`build_transform()` so evaluation/benchmarking/robustness testing all use
byte-identical preprocessing to the deployed app. This is what "preprocessing
matches training" actually means in practice: one function, one place.
"""

from torchvision import transforms


def build_transform(image_size: int, mean: list, std: list):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def preprocess_image(image, transform):
    """image: PIL.Image (already RGB). Returns a batched tensor (1, C, H, W)."""
    return transform(image).unsqueeze(0)
