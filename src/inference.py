"""
Model loading + inference abstraction.

    saved weights -> architecture -> preprocessing -> inference

is implemented exactly once here (`load_model`) and reused by app.py,
evaluate.py, benchmark.py, calibration.py, and robustness.py. This is
what Phase 1 asked for concretely: no script redefines the model or
re-implements loading with slightly different (and driftable) logic.
"""

import torch
from huggingface_hub import hf_hub_download

from src.model import HybridSkinClassifier
from src.preprocessing import build_transform

MODEL_REPO = "abhinav-29/skin-disease-classifier"
MODEL_FILE = "model.pt"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device=None):
    """
    Downloads the checkpoint from the Hub, builds the architecture from
    checkpoint["architecture"], loads weights with strict=True (fails
    loudly on any mismatch rather than silently dropping keys), and
    returns everything needed for inference.

    Returns:
        model        - HybridSkinClassifier, on `device`, in eval mode
        classes      - list[str], from checkpoint["class_names"] (never hardcoded)
        transform    - torchvision transform built from checkpoint["preprocessing"]
        device       - torch.device actually used
        checkpoint   - the raw checkpoint dict, in case callers need
                        architecture/preprocessing metadata directly
    """
    if device is None:
        device = get_device()

    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = HybridSkinClassifier(
        num_classes=checkpoint["architecture"]["num_classes"],
        dropout_rate=checkpoint["architecture"]["dropout_rate"],
    )

    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=True)
    # strict=True raises on mismatch, so reaching here means an exact match.
    # (Kept the return values above unused intentionally — strict=True already
    # verified this; see scripts/verify_model.py for an explicit printed check.)

    model.to(device)
    model.eval()

    classes = checkpoint["class_names"]

    prep = checkpoint["preprocessing"]
    transform = build_transform(prep["image_size"], prep["mean"], prep["std"])

    return model, classes, transform, device, checkpoint


@torch.inference_mode()
def predict(model, tensor, classes, top_k=5):
    """
    Runs inference under torch.inference_mode() (no autograd, no
    Grad-CAM here — use src/explainability.py separately for that,
    since Grad-CAM needs gradients and this path deliberately doesn't
    allow them, to keep plain inference as fast/lean as possible).

    Returns: dict {class_name: probability} for the top_k classes,
    sorted descending, plus the full probability tensor for anyone
    who needs top-k accuracy or calibration over the full distribution.
    """
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)[0]

    k = min(top_k, len(classes))
    top_probs, top_indices = torch.topk(probabilities, k)

    results = {
        classes[idx.item()]: float(p)
        for p, idx in zip(top_probs, top_indices)
    }

    return results, probabilities
