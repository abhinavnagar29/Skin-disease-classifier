# Interview Notes

Answers below are grounded in the actual implementation. Anywhere a
specific number is needed (accuracy, ECE, latency), it's marked
`[RUN SCRIPT]` — fill it in from the real `results/` output after
running the corresponding script, don't improvise a number in the room.

## 1. What exactly makes this model "hybrid"?

Two structurally different, independently-pretrained CNN backbones
(ResNet50 — residual connections; DenseNet121 — dense connections)
run in parallel on the same input image. Their pooled feature vectors
(2048-d and 1024-d) are concatenated into a single 3072-d vector before
classification. "Hybrid" refers to combining two backbone families at
the feature level, not to any novel layer type.

## 2. Why did you choose this architecture?

ResNet and DenseNet learn different inductive biases from the same
data — residual skip connections vs. dense feature reuse — so their
learned features aren't fully redundant. Combining them can capture
complementary visual cues without designing a new backbone from
scratch. This is a legitimate ensembling-at-the-feature-level strategy,
not a claim that it's provably superior to a single well-tuned backbone
— that comparison would require an ablation (single ResNet50 vs. hybrid)
that hasn't been run. Be upfront about that if asked.

## 3. Why is the fusion mechanism appropriate?

It's simple concatenation, deliberately not a learned gating/attention
fusion module. Trade-off, stated honestly: concatenation adds zero
extra trainable parameters at the fusion point (lower overfitting risk
on top of two already-large backbones), at the cost of not letting the
model learn to weight one branch over the other per-input. A learned
gate is a reasonable next experiment, not something this version does.

## 4. What dataset did you use?

`[RUN scripts/dataset_stats.py]` — report the real class list, image
counts, and source here. Don't guess.

## 5. How did you split the data?

`[RUN scripts/dataset_stats.py]` — report actual train/val/test counts
and ratios once known.

## 6. How did you prevent leakage?

`scripts/dataset_stats.py` hashes every image file (SHA-256) and flags
any hash appearing in more than one split — that catches exact-duplicate
leakage across train/val/test. It does **not** catch near-duplicates
(e.g. the same lesion photographed twice from slightly different
angles) — that would need perceptual hashing or manual review, and is
a real limitation to state if asked, not something to imply is covered.

## 7. How did you handle class imbalance?

`scripts/dataset_stats.py` reports the actual imbalance ratio
(max class count / min class count). `[RUN SCRIPT]` for the real number.
If imbalance is significant, the honest options are: class-weighted
loss, oversampling the minority classes, or reporting macro-averaged
metrics (which this project already does) so imbalance doesn't hide
behind a high accuracy number. Whether class weighting was actually
used during training is a training-time detail outside this repo's
scope unless you add it from your training logs.

## 8. Why is macro F1 important here?

Macro F1 averages per-class F1 equally, regardless of class size — so
a model that's great on common classes and poor on rare ones gets
penalized, unlike plain accuracy which a majority class can dominate.
For a 22-class medical-adjacent problem, missing a rare-but-serious
condition matters more than the accuracy number implies, which is
exactly what macro F1 (vs. weighted F1 or raw accuracy) is designed to
surface.

## 9. Which classes are hardest?

`[RUN scripts/error_analysis.py]` — report the real worst-F1 classes
from `results/error_analysis/worst_classes.csv`.

## 10. What does the confusion matrix reveal?

`[RUN scripts/evaluate.py]` — describe the actual most-confused pairs
from `results/error_analysis/most_confused_pairs.csv`. Generally,
expect confusion to cluster among visually similar conditions
(different classes that look alike in a photo) rather than being
randomly distributed — confirm or refute that with the real matrix,
don't assume it.

## 11. What are the biggest failure modes?

`[RUN scripts/error_analysis.py]` — report real high-confidence-wrong
examples from `results/error_analysis/high_confidence_errors.csv`.
These are the most interesting to discuss: cases where the model was
confidently wrong reveal something about what visual cues it's actually
keying on (confirm with Grad-CAM on those specific examples).

## 12. How does Grad-CAM work?

Backprop the target class's logit back to a chosen convolutional
layer's activations (here, ResNet50's final conv block). Global-average
the gradients per channel to get per-channel importance weights, take a
weighted sum of the activation maps, ReLU it (only positive influence
matters), normalize to [0,1], and upsample to the image resolution to
overlay as a heatmap.

## 13. What does Grad-CAM NOT tell you?

It doesn't prove the model is using clinically valid features (border
irregularity, asymmetry, etc.) — it shows where gradients were large
for the predicted class, which can correlate with spurious features
(image borders, lighting artifacts, skin markers) just as easily as
with the actual lesion. It also only covers the ResNet50 branch here —
it says nothing about what the DenseNet branch attended to.

## 14. How is model confidence different from calibration?

Raw softmax confidence is just the model's own output — it can be
systematically over- or under-confident. Calibration measures whether
that confidence is *meaningful*: does "80% confident" actually mean
right 80% of the time across many predictions? ECE quantifies the gap.
A model can have high accuracy and still be poorly calibrated (e.g.
consistently overconfident), which matters a lot for a domain where a
user might trust a high confidence number.

## 15. What is ECE?

Expected Calibration Error: bin predictions by confidence (e.g. 15
bins from 0 to 1), and within each bin compare average confidence to
actual accuracy in that bin. ECE is the weighted average of
|confidence − accuracy| across bins, weighted by how many predictions
fall in each bin. Lower is better; 0 is perfect calibration.
`[RUN scripts/calibration.py]` for the real value.

## 16. How did you benchmark GPU inference?

Warmup iterations first (excludes CUDA kernel compilation overhead
from the measurement), then `torch.cuda.synchronize()` both immediately
before starting the timer and immediately before stopping it for each
timed run — CUDA calls are asynchronous, so without synchronizing you'd
measure "time to queue the operation," not "time for it to actually
finish." See `scripts/benchmark.py`.

## 17. Why is `torch.inference_mode()` useful?

It disables autograd's version counter and view tracking on top of what
`torch.no_grad()` already does, giving somewhat lower overhead for
pure-inference code paths with no backward pass planned. Note: this
codebase's Grad-CAM path deliberately does NOT use inference_mode,
since it needs gradients — `predict()` in `src/inference.py` uses it,
`GradCAM.generate()` does not. That distinction itself is worth being
able to explain.

## 18. How does Streamlit caching work here?

`@st.cache_resource` on `get_model()` means the model, checkpoint, and
Grad-CAM hook object are created once per server process and reused
across every rerun (Streamlit reruns the whole script top-to-bottom on
every widget interaction) — without it, every button click would
re-download and reload the model from scratch.

## 19. How does the model run without CUDA?

`src/inference.py`'s `get_device()` checks `torch.cuda.is_available()`
and falls back to CPU automatically — no separate code path, no
environment-specific branching in the app itself. Streamlit Community
Cloud has no GPU, so this fallback isn't optional, it's the only path
that actually runs there.

## 20. How would you scale this system?

Batch inference (the codebase already benchmarks batch throughput vs.
single-image latency separately — see `scripts/benchmark.py`), a proper
inference server (TorchServe / a FastAPI + queue setup) instead of
Streamlit's synchronous request handling, and moving off Streamlit
Community Cloud's single-CPU-instance model to something with
autoscaling if traffic actually demanded it. Whether ZeroGPU-style
on-demand GPU or a dedicated always-on GPU instance makes sense depends
on real traffic volume, which isn't something this project has data on.

## 21. What would you improve with more data?

`[Fill in after reviewing dataset_stats.json and error_analysis]` —
likely candidates: more examples for the worst-performing/rarest
classes specifically (targeted collection, not just "more data
generally"), and possibly more demographic/skin-tone diversity if the
current dataset is skewed (this project doesn't currently measure
that — it's a stated limitation, not a validated finding).

## 22. What would you change if deploying this in production?

Class-imbalance-aware training if imbalance turns out to be severe,
post-hoc calibration (temperature scaling) if raw ECE is poor, human-
in-the-loop review rather than fully automated decisions given the
medical-adjacent domain, monitoring for input distribution shift over
time, and almost certainly a proper regulatory/clinical validation
process before any real medical use — this project explicitly does not
attempt that and says so throughout.
