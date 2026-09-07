"""
Grad-CAM interpretability for the ResNet50 branch.

Scope, stated explicitly rather than glossed over:
    - Hooked on model.backbone.resnet_target_layer (ResNet50's final
      conv block, "layer4"). This is a valid Grad-CAM target: it's the
      last spatial feature map before global pooling.
    - The DenseNet121 branch is NOT visualized. Grad-CAM here explains
      what the ResNet branch attended to, not the fused decision as a
      whole. That's a real scope limitation to state in the README/
      interview notes, not a bug.
    - Grad-CAM highlights correlation between image regions and the
      target class's logit, via gradient-weighted activation maps. It
      does NOT prove the model is reasoning about clinically correct
      features (texture, asymmetry, etc.) — it shows where the ResNet
      branch's gradients were large, nothing about medical validity.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import cm


class GradCAM:
    def __init__(self, target_model, target_layer):
        self.model = target_model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx, logits=None):
        self.model.zero_grad(set_to_none=True)

        output = logits if logits is not None else self.model(input_tensor)
        score = output[:, class_idx].squeeze()
        score.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam_min, cam_max = cam.min(), cam.max()
        if (cam_max - cam_min) > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.detach().cpu().numpy()


def make_gradcam_visuals(original_image: Image.Image, cam: np.ndarray):
    """Returns (heatmap_image, overlay_image) resized to original_image's resolution."""
    width, height = original_image.size

    cam_img = Image.fromarray(np.uint8(cam * 255))
    cam_resized = cam_img.resize((width, height), resample=Image.BICUBIC)
    cam_resized_np = np.array(cam_resized).astype(np.float32) / 255.0

    colored = cm.jet(cam_resized_np)[:, :, :3]
    heatmap_image = Image.fromarray(np.uint8(colored * 255))

    base_image = original_image.convert("RGB")
    overlay_image = Image.blend(base_image, heatmap_image, alpha=0.45)

    return heatmap_image, overlay_image
