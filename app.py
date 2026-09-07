import spaces
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import gradio as gr
import numpy as np

from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
from matplotlib import cm

# NOTE for Hugging Face Space deployment:
# Grad-CAM visualization requires `matplotlib` in addition to the existing
# torch / torchvision / gradio / huggingface_hub / pillow dependencies.
# Make sure `matplotlib` is listed in requirements.txt for the Space.


# ============================================================
# CONFIG
# ============================================================

MODEL_REPO = "abhinav-29/skin-disease-classifier"
MODEL_FILE = "model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MODEL ARCHITECTURE  (UNCHANGED)
# ============================================================

class HybridBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet50(weights=None)
        self.resnet_features = nn.Sequential(
            *list(resnet.children())[:-2]
        )
        self.resnet_target_layer = self.resnet_features[-1]

        densenet = models.densenet121(weights=None)
        self.densenet_features = densenet.features
        self.densenet_target_layer = self.densenet_features.denseblock4

    def forward(self, x):
        resnet_out = self.resnet_features(x)
        resnet_out = F.adaptive_avg_pool2d(
            resnet_out,
            (1, 1)
        )
        resnet_out = torch.flatten(
            resnet_out,
            1
        )

        densenet_out = self.densenet_features(x)
        densenet_out = F.relu(
            densenet_out,
            inplace=False
        )
        densenet_out = F.adaptive_avg_pool2d(
            densenet_out,
            (1, 1)
        )
        densenet_out = torch.flatten(
            densenet_out,
            1
        )

        return torch.cat(
            [resnet_out, densenet_out],
            dim=1
        )


class HybridSkinClassifier(nn.Module):
    def __init__(
        self,
        num_classes=22,
        dropout_rate=0.5
    ):
        super().__init__()

        self.backbone = HybridBackbone()

        self.classifier = nn.ModuleDict({
            "net": nn.Sequential(
                nn.Linear(3072, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(dropout_rate),

                nn.Linear(2048, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(dropout_rate),

                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),

                nn.Linear(512, num_classes)
            )
        })

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier["net"](features)


# ============================================================
# LOAD MODEL  (UNCHANGED)
# ============================================================

print("Downloading/loading model...")

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE
)

checkpoint = torch.load(
    model_path,
    map_location=device,
    weights_only=False
)

model = HybridSkinClassifier(
    num_classes=checkpoint["architecture"]["num_classes"],
    dropout_rate=checkpoint["architecture"]["dropout_rate"]
)

model.load_state_dict(
    checkpoint["state_dict"]
)

model.to(device)
model.eval()

classes = checkpoint["class_names"]

image_size = checkpoint["preprocessing"]["image_size"]
mean = checkpoint["preprocessing"]["mean"]
std = checkpoint["preprocessing"]["std"]

transform = transforms.Compose([
    transforms.Resize(
        (image_size, image_size)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=mean,
        std=std
    )
])

print("Model loaded successfully.")
print(f"Device: {device}")
print(f"Classes: {len(classes)}")


# ============================================================
# GRAD-CAM
# ============================================================
# Uses the existing, already-loaded `model` and its
# `model.backbone.resnet_target_layer` (the ResNet50 layer4 block)
# as the target layer. No weights are modified — hooks only read
# activations/gradients that flow through the frozen, loaded model.

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
        """
        Computes a Grad-CAM heatmap for `class_idx`.
        If `logits` is provided (from a forward pass the caller already
        ran with gradients enabled), that forward pass is reused instead
        of running the model again — this avoids a redundant full
        forward through both backbones.
        Returns a 2D numpy array in [0, 1] at the target layer's
        spatial resolution.
        """
        self.model.zero_grad(set_to_none=True)

        output = logits if logits is not None else self.model(input_tensor)
        score = output[:, class_idx].squeeze()
        score.backward()

        gradients = self.gradients[0]      # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Global-average-pooled gradients act as per-channel importance weights
        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(
            activations.shape[1:],
            dtype=torch.float32,
            device=activations.device
        )

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)

        cam_min = cam.min()
        cam_max = cam.max()

        if (cam_max - cam_min) > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.detach().cpu().numpy()


gradcam = GradCAM(model, model.backbone.resnet_target_layer)


def generate_gradcam_visuals(original_image, cam):
    """
    original_image: the original (pre-resize) PIL RGB image the user uploaded.
    cam: 2D numpy array in [0, 1] returned by GradCAM.generate().

    Returns (heatmap_image, overlay_image) as PIL Images, both resized
    to the original image's resolution so they line up with the upload.
    """
    width, height = original_image.size

    cam_img = Image.fromarray(np.uint8(cam * 255))
    cam_resized = cam_img.resize((width, height), resample=Image.BICUBIC)
    cam_resized_np = np.array(cam_resized).astype(np.float32) / 255.0

    # Colorize the single-channel CAM with a "jet"-style colormap
    colored = cm.jet(cam_resized_np)[:, :, :3]
    heatmap_image = Image.fromarray(np.uint8(colored * 255))

    base_image = original_image.convert("RGB")
    overlay_image = Image.blend(base_image, heatmap_image, alpha=0.45)

    return heatmap_image, overlay_image


# ============================================================
# PREDICTION  (classification logic unchanged; Grad-CAM appended)
# ============================================================

@spaces.GPU
def predict(image):
    if image is None:
        return (
            {},
            "Upload an image to begin analysis.",
            "Waiting for image",
            None,
            None,
            None
        )

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGB")

    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)

    # Single forward pass (gradients enabled) reused for both the
    # classification result AND Grad-CAM — avoids running the model twice.
    with torch.enable_grad():
        logits = model(tensor)
        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    top_probs, top_indices = torch.topk(
        probabilities.detach(),
        min(5, len(classes))
    )

    results = {}

    for prob, idx in zip(
        top_probs,
        top_indices
    ):
        results[classes[idx.item()]] = float(prob)

    top_class_idx = top_indices[0].item()

    top_class = classes[top_class_idx]

    top_confidence = float(
        top_probs[0]
    ) * 100

    summary = (
        f"Top prediction: **{top_class}**  \n"
        f"Model confidence: **{top_confidence:.1f}%**"
    )

    status = "Analysis complete"

    # ---- Grad-CAM explanation (reuses the forward pass above) ----
    heatmap_image = None
    overlay_image = None

    try:
        cam = gradcam.generate(tensor, top_class_idx, logits=logits)

        heatmap_image, overlay_image = generate_gradcam_visuals(
            image,
            cam
        )
    except Exception as grad_cam_error:
        # Never let an explainability failure break the core prediction.
        print(f"Grad-CAM generation failed: {grad_cam_error}")

    return (
        results,
        summary,
        status,
        image,
        heatmap_image,
        overlay_image
    )


# ============================================================
# CUSTOM CSS — clean, product-grade interface
# ============================================================

CSS = """

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ---------------------------------------------------------
   GLOBAL
--------------------------------------------------------- */

body {
    background: #f7f8fa;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.gradio-container {
    max-width: 1040px !important;
    margin: auto !important;
    padding: 0 20px 40px 20px !important;
}


/* ---------------------------------------------------------
   TOP BAR
--------------------------------------------------------- */

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 22px 4px 20px 4px;
    border-bottom: 1px solid #e4e7ec;
    margin-bottom: 28px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 10px;
}

.brand-mark {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: #2563eb;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-weight: 700;
    font-size: 15px;
}

.brand-name {
    font-size: 16px;
    font-weight: 700;
    color: #111827;
}

.brand-tag {
    font-size: 12px;
    color: #6b7280;
    font-weight: 500;
    padding-left: 12px;
    border-left: 1px solid #e4e7ec;
    margin-left: 2px;
}

.beta-pill {
    font-size: 11px;
    font-weight: 600;
    color: #2563eb;
    background: #eff4ff;
    border: 1px solid #d6e4ff;
    padding: 3px 9px;
    border-radius: 999px;
}


/* ---------------------------------------------------------
   PAGE INTRO
--------------------------------------------------------- */

.page-intro {
    margin-bottom: 22px;
}

.page-title {
    font-size: 23px;
    font-weight: 700;
    color: #111827;
    margin: 0 0 6px 0;
}

.page-subtitle {
    font-size: 14px;
    color: #6b7280;
    line-height: 1.6;
    max-width: 640px;
    margin: 0;
}


/* ---------------------------------------------------------
   CARDS
--------------------------------------------------------- */

.card {
    background: #ffffff;
    border: 1px solid #e4e7ec;
    border-radius: 12px;
    padding: 22px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
}

.card-title {
    font-size: 14.5px;
    font-weight: 600;
    color: #111827;
    margin-bottom: 3px;
}

.card-description {
    color: #6b7280;
    font-size: 12.5px;
    margin-bottom: 16px;
    line-height: 1.5;
}


/* ---------------------------------------------------------
   UPLOAD
--------------------------------------------------------- */

.upload-box {
    min-height: 400px;
}

.upload-box .image-container {
    border-radius: 8px !important;
}

.upload-box label {
    font-weight: 600 !important;
    color: #374151 !important;
    font-size: 13px !important;
}


/* ---------------------------------------------------------
   RESULTS
--------------------------------------------------------- */

.result-card {
    min-height: 400px;
}

.result-title {
    font-size: 14.5px;
    font-weight: 600;
    color: #111827;
}

.result-subtitle {
    color: #6b7280;
    font-size: 12.5px;
    margin-top: 2px;
    margin-bottom: 16px;
    line-height: 1.5;
}

.empty-result {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    height: 260px;
    color: #9ca3af;
    font-size: 13px;
    border: 1px dashed #e4e7ec;
    border-radius: 8px;
    background: #fafbfc;
}


/* ---------------------------------------------------------
   BUTTONS
--------------------------------------------------------- */

.primary-btn {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    min-height: 44px !important;
    background: #2563eb !important;
    border: 1px solid #2563eb !important;
    box-shadow: none !important;
}

.primary-btn:hover {
    background: #1d4ed8 !important;
    border: 1px solid #1d4ed8 !important;
}


/* ---------------------------------------------------------
   NOTICE STRIP (compact disclaimer)
--------------------------------------------------------- */

.notice {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    background: #fffaf0;
    border: 1px solid #fde8c7;
    border-radius: 10px;
    padding: 13px 16px;
    margin-top: 22px;
}

.notice-icon {
    font-size: 15px;
    line-height: 1.4;
}

.notice-text {
    color: #92590a;
    font-size: 12.5px;
    line-height: 1.55;
}

.notice-text b {
    color: #6d4106;
}


/* ---------------------------------------------------------
   DETAILS ACCORDION CONTENT
--------------------------------------------------------- */

.details-block {
    color: #4b5563;
    font-size: 13px;
    line-height: 1.75;
    padding: 4px 2px 8px 2px;
}

.details-block b {
    color: #111827;
}

.spec-row {
    display: flex;
    justify-content: space-between;
    padding: 9px 0;
    border-bottom: 1px solid #f0f1f3;
    font-size: 13px;
}

.spec-row:last-child {
    border-bottom: none;
}

.spec-row span:first-child {
    color: #6b7280;
}

.spec-row span:last-child {
    color: #111827;
    font-weight: 600;
}

.category-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin-top: 10px;
}

.category-tag {
    font-size: 12px;
    color: #374151;
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
    padding: 4px 10px;
    border-radius: 999px;
    font-weight: 500;
}


/* ---------------------------------------------------------
   GRAD-CAM EXPLAINABILITY SECTION
--------------------------------------------------------- */

.gradcam-section {
    margin-top: 22px;
}

.gradcam-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 4px;
}

.gradcam-title {
    font-size: 14.5px;
    font-weight: 600;
    color: #111827;
}

.gradcam-badge {
    font-size: 11px;
    font-weight: 600;
    color: #6b7280;
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
    padding: 2px 9px;
    border-radius: 999px;
}

.gradcam-description {
    color: #6b7280;
    font-size: 12.5px;
    margin-bottom: 16px;
    line-height: 1.5;
}

.gradcam-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
}

.gradcam-caption {
    text-align: center;
    font-size: 12.5px;
    font-weight: 600;
    color: #374151;
    margin-top: 8px;
}

.gradcam-subcaption {
    text-align: center;
    font-size: 11.5px;
    color: #9ca3af;
    margin-top: 1px;
}

@media (max-width: 800px) {
    .gradcam-grid {
        grid-template-columns: 1fr;
    }
}


/* ---------------------------------------------------------
   FOOTER
--------------------------------------------------------- */

.footer {
    text-align: center;
    color: #9ca3af;
    font-size: 12px;
    padding: 26px 0 4px 0;
    margin-top: 8px;
    line-height: 1.7;
}


/* ---------------------------------------------------------
   MOBILE
--------------------------------------------------------- */

@media (max-width: 800px) {

    .page-title {
        font-size: 20px;
    }

    .gradio-container {
        padding: 0 12px 28px 12px !important;
    }

    .brand-tag {
        display: none;
    }
}

"""


# ============================================================
# UI
# ============================================================

with gr.Blocks(
    title="SkinAI — Skin Disease Classifier"
) as demo:

    # --------------------------------------------------------
    # TOP BAR
    # --------------------------------------------------------

    gr.HTML(
        """
        <div class="topbar">
            <div class="brand">
                <div class="brand-mark">S</div>
                <div class="brand-name">SkinAI</div>
                <div class="brand-tag">Skin Image Screening</div>
            </div>
            <div class="beta-pill">BETA</div>
        </div>
        """
    )


    # --------------------------------------------------------
    # PAGE INTRO
    # --------------------------------------------------------

    gr.HTML(
        """
        <div class="page-intro">
            <p class="page-title">Analyze a skin image</p>
            <p class="page-subtitle">
                Upload a clear photo of the affected area to get an
                instant, AI-assisted screening across 22 common skin
                conditions.
            </p>
        </div>
        """
    )


    # --------------------------------------------------------
    # MAIN ANALYSIS AREA
    # --------------------------------------------------------

    with gr.Row(equal_height=True):

        # IMAGE CARD
        with gr.Column(
            scale=1,
            elem_classes=["card", "upload-box"]
        ):

            gr.HTML(
                """
                <div class="card-title">
                    1. Upload
                </div>

                <div class="card-description">
                    Use a close, well-lit, in-focus photo for the
                    most reliable result.
                </div>
                """
            )

            image_input = gr.Image(
                type="pil",
                label="Skin image",
                height=280
            )

            status = gr.Markdown(
                "Waiting for image"
            )


        # RESULT CARD
        with gr.Column(
            scale=1,
            elem_classes=["card", "result-card"]
        ):

            gr.HTML(
                """
                <div class="result-title">
                    2. Result
                </div>

                <div class="result-subtitle">
                    Ranked by model confidence
                </div>
                """
            )

            output = gr.Label(
                num_top_classes=5,
                label="Likelihood"
            )

            summary = gr.Markdown(
                "Upload an image to see a result."
            )


    # --------------------------------------------------------
    # ACTION BUTTON
    # --------------------------------------------------------

    with gr.Row():

        predict_button = gr.Button(
            "Analyze Image",
            variant="primary",
            size="lg",
            elem_classes=["primary-btn"]
        )

        clear_button = gr.ClearButton(
            components=[
                image_input,
                output,
                summary,
                status
            ],
            value="Clear"
        )


    # --------------------------------------------------------
    # GRAD-CAM EXPLAINABILITY
    # --------------------------------------------------------

    with gr.Column(elem_classes=["card", "gradcam-section"]):

        gr.HTML(
            """
            <div class="gradcam-header">
                <div class="gradcam-title">Why this prediction?</div>
                <div class="gradcam-badge">Grad-CAM</div>
            </div>
            <div class="gradcam-description">
                Highlighted regions show where the model focused most
                when producing its top prediction.
            </div>
            """
        )

        with gr.Row(elem_classes=["gradcam-grid"]):

            with gr.Column():
                original_output = gr.Image(
                    show_label=False,
                    interactive=False,
                    height=220
                )
                gr.HTML(
                    """
                    <div class="gradcam-caption">Original Image</div>
                    <div class="gradcam-subcaption">As uploaded</div>
                    """
                )

            with gr.Column():
                heatmap_output = gr.Image(
                    show_label=False,
                    interactive=False,
                    height=220
                )
                gr.HTML(
                    """
                    <div class="gradcam-caption">Grad-CAM Heatmap</div>
                    <div class="gradcam-subcaption">Model attention</div>
                    """
                )

            with gr.Column():
                overlay_output = gr.Image(
                    show_label=False,
                    interactive=False,
                    height=220
                )
                gr.HTML(
                    """
                    <div class="gradcam-caption">Overlay</div>
                    <div class="gradcam-subcaption">Heatmap on image</div>
                    """
                )


    predict_button.click(
        fn=predict,
        inputs=image_input,
        outputs=[
            output,
            summary,
            status,
            original_output,
            heatmap_output,
            overlay_output
        ]
    )

    clear_button.add(
        [
            original_output,
            heatmap_output,
            overlay_output
        ]
    )


    # --------------------------------------------------------
    # COMPACT SAFETY NOTICE
    # --------------------------------------------------------

    gr.HTML(
        """
        <div class="notice">
            <div class="notice-icon">⚠️</div>
            <div class="notice-text">
                <b>Not a medical diagnosis.</b> This tool provides an
                AI-generated screening estimate only. For any concerning,
                changing, or persistent skin condition, please consult a
                licensed dermatologist or physician.
            </div>
        </div>
        """
    )


    # --------------------------------------------------------
    # DETAILS — tucked away so the main page stays uncluttered
    # --------------------------------------------------------

    with gr.Accordion("About this tool", open=False):

        gr.HTML(
            """
            <div class="details-block">
                SkinAI uses a hybrid deep-learning model that combines
                two convolutional neural networks to recognize visual
                patterns across common skin conditions. It's designed
                as a quick first-look screening aid, not a replacement
                for a clinical exam. Each result also includes a
                Grad-CAM visualization showing which regions of the
                image most influenced the prediction.

                <div class="spec-row"><span>Model type</span><span>Hybrid CNN ensemble</span></div>
                <div class="spec-row"><span>Categories detected</span><span>22</span></div>
                <div class="spec-row"><span>Recommended input</span><span>Clear, close-up photo</span></div>
                <div class="spec-row"><span>Explainability</span><span>Grad-CAM (ResNet50 branch)</span></div>

                <div style="margin-top: 14px; font-weight: 600; color: #111827;">
                    Conditions covered
                </div>

                <div class="category-tags">
                    <span class="category-tag">Acne</span>
                    <span class="category-tag">Actinic Keratosis</span>
                    <span class="category-tag">Benign Tumors</span>
                    <span class="category-tag">Bullous</span>
                    <span class="category-tag">Candidiasis</span>
                    <span class="category-tag">Drug Eruption</span>
                    <span class="category-tag">Eczema</span>
                    <span class="category-tag">Infestations &amp; Bites</span>
                    <span class="category-tag">Lichen</span>
                    <span class="category-tag">Lupus</span>
                    <span class="category-tag">Moles</span>
                    <span class="category-tag">Psoriasis</span>
                    <span class="category-tag">Rosacea</span>
                    <span class="category-tag">Seborrheic Keratoses</span>
                    <span class="category-tag">Skin Cancer</span>
                    <span class="category-tag">Sun/Sunlight Damage</span>
                    <span class="category-tag">Tinea</span>
                    <span class="category-tag">Unknown/Normal</span>
                    <span class="category-tag">Vascular Tumors</span>
                    <span class="category-tag">Vasculitis</span>
                    <span class="category-tag">Vitiligo</span>
                    <span class="category-tag">Warts</span>
                </div>
            </div>
            """
        )


    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    gr.HTML(
        """
        <div class="footer">
            SkinAI · For informational purposes only, not a medical device
        </div>
        """
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate"
        )
    )
