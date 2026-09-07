# Hybrid Skin Disease Classifier

A Gradio app that classifies an uploaded skin image into one of 22
categories using a hybrid ResNet50 + DenseNet121 model, with Grad-CAM
visual explanations for every prediction.

Live demo: https://huggingface.co/spaces/abhinav-29/hybrid-skin-disease-classifier

> **Medical disclaimer:** This project is a research / decision-support
> tool only. It is not a certified medical device, has not been
> clinically validated, and should never replace an evaluation by a
> licensed dermatologist or physician. Model confidence is not medical
> certainty.

## What it does

- Upload a skin image.
- The model returns its top-5 predicted categories with confidence
  scores.
- A Grad-CAM heatmap and overlay show which region of the image the
  model actually focused on to make that prediction.

## Architecture

Two ImageNet-pretrained backbones extract features from the same input
image; a four-layer classifier head turns the concatenated feature
vector into class logits.

```
Input (3x224x224)
      │
      ├────────────→ ResNet50 (no FC layer)     → 2048-d
      │
      └────────────→ DenseNet121 (no classifier) → 1024-d
                          │
                          ▼
                    concatenate → 3072-d
                          │
                          ▼
             Linear(3072→2048) BN ReLU Dropout
             Linear(2048→1024) BN ReLU Dropout
             Linear(1024→512)  BN ReLU Dropout
             Linear(512→num_classes)
                          │
                          ▼
                   disease logits → softmax
```

The trained weights (`model.pt`) are not stored in this repo — they're
pulled at runtime from the Hugging Face Hub
(`abhinav-29/skin-disease-classifier`), so the checkpoint is versioned
separately from the app code.

## Grad-CAM

Grad-CAM runs against `model.backbone.resnet_target_layer` (the final
ResNet50 conv block). It reuses the same forward pass as the
classification step — one forward pass, one backward pass per request,
not two separate forward passes — so the explanation doesn't cost much
more than the prediction alone.

## Running locally

```bash
git clone https://github.com/<your-username>/hybrid-skin-disease-classifier.git
cd hybrid-skin-disease-classifier

python -m venv .venv
source .venv/bin/activate      # .venv\Scripts\activate on Windows

pip install -r requirements.txt
python app.py
```

The first run downloads the model checkpoint from the Hub, so it needs
an internet connection the first time. `@spaces.GPU` is a no-op outside
of a Hugging Face ZeroGPU Space, so this runs fine locally on CPU (or a
local GPU if you have one) — just slower.

## Deploying on Hugging Face Spaces

This app targets Hugging Face's **ZeroGPU** — free, on-demand GPU
access that currently only supports the Gradio SDK.

```bash
hf auth login
hf repos create <your-username>/hybrid-skin-disease-classifier --type space --space-sdk gradio --flavor zero-a10g
hf upload <your-username>/hybrid-skin-disease-classifier . . --repo-type space
```

Notes:

- ZeroGPU is only available to accounts in good standing that are at
  least 30 days old (or that have been granted a community GPU grant
  from the target Space's Settings tab). New accounts will see a 402
  error until then.
- Static and CPU-basic/Docker Spaces cannot run this app — Static
  Spaces don't execute Python at all, and CPU-basic/Docker currently
  require a PRO subscription. ZeroGPU is the only free path for a
  live-inference Gradio app on Hugging Face at the time of writing.

## Project structure

```
app.py            - model definition, checkpoint loading, Grad-CAM, Gradio UI
requirements.txt  - Python dependencies
README.md         - this file
```

Everything lives in a single `app.py` by design — model class, Grad-CAM
hooks, inference function, and UI are all in one file so the whole app
can be understood top to bottom without jumping between modules.

## Model

- **Repo:** `abhinav-29/skin-disease-classifier` on the Hugging Face Hub
- **Input:** RGB image, resized per the checkpoint's stored
  preprocessing config (224x224, ImageNet-style normalization)
- **Output:** 22-class softmax distribution
- **Checkpoint format:** a single `torch.save`d dict containing
  `state_dict`, `architecture` (num_classes, dropout_rate),
  `class_names`, and `preprocessing` (image_size, mean, std) — so the
  app has no hardcoded assumptions about class count or normalization
  values, it reads them from the checkpoint itself.

## Limitations

- Evaluated only on the dataset it was trained on; performance may not
  generalize to images from different cameras, lighting, or skin tones
  not well represented in training data.
- Some class boundaries between visually similar conditions are
  inherently ambiguous, even for expert clinicians, from a single image.
- No fairness or subgroup performance analysis has been done. Treat any
  use beyond casual/research purposes with real caution.

## License

MIT — see [LICENSE](LICENSE).
