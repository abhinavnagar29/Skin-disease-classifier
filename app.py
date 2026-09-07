"""
Hybrid Skin Disease Classifier — Streamlit application.

The trained model is unchanged from the deployed Gradio version. This
file only changes the UI framework and adds a results/performance
section fed by scripts/evaluate.py's output — it does not touch
architecture, weights, or preprocessing.
"""

import json
import os

import streamlit as st
import torch
from PIL import Image

from src.inference import load_model, predict
from src.preprocessing import preprocess_image
from src.explainability import GradCAM, make_gradcam_visuals

RESULTS_DIR = "results"


# ------------------------------------------------------------------
# Cached model load — runs once per server process, not per rerun.
# This is the actual mechanism Phase 10 asked for: st.cache_resource
# on the model itself (not on individual predictions), so Streamlit's
# rerun-on-every-interaction behavior doesn't reload the model or
# re-download the checkpoint each time a widget changes.
# ------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")
def get_model():
    model, classes, transform, device, checkpoint = load_model()
    gradcam = GradCAM(model, model.backbone.resnet_target_layer)
    return model, classes, transform, device, checkpoint, gradcam


def load_results_file(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def run_inference(model, classes, transform, device, gradcam, image: Image.Image):
    image = image.convert("RGB")
    tensor = preprocess_image(image, transform).to(device)

    with torch.enable_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    top5_probs, top5_indices = torch.topk(probabilities.detach(), min(5, len(classes)))
    top5 = {classes[i.item()]: float(p) for p, i in zip(top5_probs, top5_indices)}

    top_class_idx = top5_indices[0].item()
    top_class = classes[top_class_idx]
    top_confidence = float(top5_probs[0]) * 100

    heatmap_image, overlay_image = None, None
    try:
        cam = gradcam.generate(tensor, top_class_idx, logits=logits)
        heatmap_image, overlay_image = make_gradcam_visuals(image, cam)
    except Exception as e:
        st.warning(f"Grad-CAM could not be generated for this image: {e}")

    return top5, top_class, top_confidence, heatmap_image, overlay_image


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------

st.set_page_config(page_title="Hybrid Skin Disease Classifier", layout="wide")

st.title("Hybrid Skin Disease Classifier")
st.caption(
    "An ML research/educational demonstration of a hybrid ResNet50 + DenseNet121 "
    "classifier with Grad-CAM interpretability. Not a medical device."
)

st.warning(
    "**This is not a medical diagnostic tool.** Predictions are AI-generated "
    "estimates from a research project and have not been clinically validated. "
    "Do not use this to make medical decisions. Consult a licensed dermatologist "
    "or physician for any real skin concern."
)

try:
    model, classes, transform, device, checkpoint, gradcam = get_model()
    model_loaded = True
except Exception as e:
    st.error(f"Model failed to load: {e}")
    model_loaded = False

# ---- Sidebar ----
with st.sidebar:
    st.header("Model Info")
    if model_loaded:
        st.write(f"**Model type:** Hybrid ResNet50 + DenseNet121")
        st.write(f"**Classes:** {len(classes)}")
        st.write(f"**Input size:** {checkpoint['preprocessing']['image_size']}x{checkpoint['preprocessing']['image_size']}")
        st.write(f"**Inference device:** {device}")
        st.write(f"**Fusion:** feature concatenation (3072-d)")
    else:
        st.write("Model not loaded.")

    st.divider()
    st.header("Dataset")
    dataset_stats = load_results_file("dataset_stats.json")
    if dataset_stats and "splits" in dataset_stats:
        for split_name, s in dataset_stats["splits"].items():
            st.write(f"**{split_name.capitalize()}:** {s['total_images']} images")
    else:
        st.caption("Run `scripts/dataset_stats.py` to populate dataset info here.")

# ---- Main: upload + prediction ----
uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None and model_loaded:
    try:
        image = Image.open(uploaded_file)
    except Exception:
        st.error("Couldn't read this file as an image. Please upload a JPG, PNG, or WEBP.")
        image = None

    if image is not None:
        left, right = st.columns([1, 1])
        with left:
            st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Analyze image", type="primary"):
            with st.spinner("Running inference..."):
                top5, top_class, top_confidence, heatmap_image, overlay_image = run_inference(
                    model, classes, transform, device, gradcam, image
                )

            with right:
                st.subheader("Result")
                st.markdown(f"**Top-1 prediction:** {top_class}")
                st.markdown(f"**Confidence:** {top_confidence:.1f}%")

                st.write("**Top-3 predictions:**")
                for label, prob in list(top5.items())[:3]:
                    st.write(f"{label} — {prob * 100:.1f}%")

                st.write("**Top-5 predictions:**")
                st.bar_chart({label: prob for label, prob in top5.items()})

            if heatmap_image is not None and overlay_image is not None:
                st.subheader("Model Explanation — Grad-CAM")
                st.caption(
                    "Grad-CAM shows which image regions most influenced the ResNet50 "
                    "branch's prediction — it reflects gradient activity, not a claim "
                    "of clinical relevance."
                )
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(image, caption="Original", use_container_width=True)
                with col2:
                    st.image(heatmap_image, caption="Grad-CAM heatmap", use_container_width=True)
                with col3:
                    st.image(overlay_image, caption="Overlay", use_container_width=True)
else:
    st.info("Upload an image to get started.")

# ---- How the model works ----
with st.expander("How the model works"):
    st.markdown("""
The classifier combines two independently pretrained convolutional backbones:

- **ResNet50** (final FC layer removed) → global average pooling → 2048-d feature vector
- **DenseNet121** (classifier removed) → ReLU → global average pooling → 1024-d feature vector

These two vectors are **concatenated** into a single 3072-d feature vector — this is
feature-level fusion by concatenation, not a learned gating or attention mechanism.
The joint vector is passed through a 4-layer fully-connected head
(`3072 → 2048 → 1024 → 512 → num_classes`, each block using BatchNorm, ReLU, and
Dropout) to produce class logits.

Grad-CAM is computed against the ResNet50 branch's final convolutional block only —
it explains what that branch attended to, not the DenseNet branch or the fused
decision as a whole.
""")

# ---- Model performance ----
with st.expander("Model Performance (test set)"):
    metrics = load_results_file("metrics.json")
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
        col2.metric("Balanced Accuracy", f"{metrics['balanced_accuracy']*100:.1f}%")
        col3.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
        col4.metric("Weighted F1", f"{metrics['weighted_f1']:.3f}")

        cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion matrix (test set)")
    else:
        st.info(
            "No evaluation results found yet. Run `python scripts/evaluate.py "
            "--test-dir data/test` to generate real test-set metrics, which will "
            "then appear here automatically — nothing is hardcoded in this section."
        )

st.divider()
st.caption(
    "Research/educational ML demonstration only. Not a certified medical device. "
    "Not clinically validated. Do not use for medical decision-making."
)
