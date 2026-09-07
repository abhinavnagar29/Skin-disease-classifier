# Model Storage

The trained checkpoint (`model.pt`) is **not committed to this Git repository**.

## Why

A hybrid ResNet50 + DenseNet121 model is a large binary artifact — rough
back-of-envelope estimate is in the tens-of-millions of parameters range
(run `python scripts/verify_model.py` for the exact, real parameter count
and file size; don't trust an estimate here over the actual number).
That's well past GitHub's soft limits for plain Git, and Git LFS adds
storage-quota complexity that isn't needed when a better option already
exists for this project.

## What's actually used

The checkpoint is hosted on the **Hugging Face Hub**
(`abhinav-29/skin-disease-classifier`) and downloaded at runtime via
`huggingface_hub.hf_hub_download()` — see `src/inference.py:load_model()`.
This means:

- The Git repo stays small and fast to clone.
- The model is versioned separately from the app code (a model update
  doesn't require a code change, and vice versa).
- Both the Gradio Space and this Streamlit app pull from the exact same
  source of truth — no risk of the two deployments silently drifting
  onto different weights.
- No base64-embedding, no LFS setup, no custom artifact server — this is
  the standard, reproducible way to separate code and model weights for
  a Hub-hosted model.

## First run

The first time `app.py` or any `scripts/*.py` runs, it will download
`model.pt` from the Hub — this needs an internet connection on first run
(subsequent runs use the local `huggingface_hub` cache).
