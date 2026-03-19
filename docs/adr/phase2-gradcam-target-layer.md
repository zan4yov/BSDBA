# ADR-0004 — Q4 Resolution: Grad-CAM Target Layer for EfficientNet-B4
**Document:** DSDBA-SRS-2026-002 v2.1 | Phase 2 Deliverable  
**Authors:** Ferel, Safa — ITS Informatics | KCVanguard ML Workshop  
**Date:** 2026-03-18  
**Status:** ✅ RESOLVED — blocks Sprint C (FR-CV-010, FR-CV-011, FR-CV-015, FR-CV-016)  
**SRS References:** FR-CV-010, FR-CV-011, FR-CV-012, FR-CV-015, FR-CV-016  
**MCP Tool Used:** context7-mcp (`/jacobgil/pytorch-grad-cam`)

---

## Decision

**Confirmed Grad-CAM target layer:** `model.features[7][-1]`  
**Named module path:** `features.7.1`  
**Module type:** `MBConv` (torchvision internal, final block of Stage 7)  
**config.yaml key to update:** `gradcam.target_layer: "model.features[7][-1]"`

---

## Architecture Analysis — torchvision 0.16.0 EfficientNet-B4

EfficientNet-B4 applies compound scaling: `depth_coeff=1.8`, `width_coeff=1.4`, `input_resolution=380` (SRS uses 224×224 per FR-AUD-008 — this is valid; Grad-CAM operates on feature maps, not raw resolution).

`model.features` is a `torch.nn.Sequential` with **9 children (indices 0–8)**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| `features[0]` | stem | `Conv2dNormActivation` | 3 → 48 ch, k=3, s=2 |
| `features[1]` | Stage 1 | `Sequential` (2× MBConv1) | 48 → 24 ch, k=3, s=1 |
| `features[2]` | Stage 2 | `Sequential` (4× MBConv6) | 24 → 32 ch, k=3, s=2 |
| `features[3]` | Stage 3 | `Sequential` (4× MBConv6) | 32 → 56 ch, k=5, s=2 |
| `features[4]` | Stage 4 | `Sequential` (6× MBConv6) | 56 → 112 ch, k=3, s=2 |
| `features[5]` | Stage 5 | `Sequential` (6× MBConv6) | 112 → 160 ch, k=5, s=1 |
| `features[6]` | Stage 6 | `Sequential` (8× MBConv6) | 160 → 272 ch, k=5, s=2 |
| **`features[7]`** | **Stage 7** | **`Sequential` (2× MBConv6)** | **272 → 448 ch, k=3, s=1** |
| `features[8]` | head | `Conv2dNormActivation` | 448 → 1792 ch, k=1 |

> **Block counts derived from B4 depth scaling:** B0 blocks × 1.8, ceil-rounded per stage.  
> Stage 7 base count (B0) = 1 block × 1.8 = 1.8 → `ceil(1.8) = 2` blocks.

### Why `features[7][-1]` and NOT `features[-1]`

- `model.features[-1]` = `model.features[8]` = the **head `Conv2dNormActivation`**. This is a 1×1 pointwise convolution that increases channels 448→1792 and is **not** an MBConv block.
- FR-CV-010 specifies: *"Grad-CAM targets the **final MBConv** convolutional block"* — this is unambiguously `features[7][-1]` (the last block in the last MBConv stage).
- `model.features[7][-1]` = `model.features[7][1]` = the second (final) MBConv block in Stage 7, named `features.7.1`.
- The MBConv block at `features.7.1` contains the deepest semantic spatial features before the head conv expands channels, making it the best Grad-CAM signal source per the pytorch-grad-cam convention: `target_layers = [model.layer4[-1]]` (ResNet) → `target_layers = [model.features[7][-1]]` (EfficientNet-B4).

### Previously Incorrect config.yaml Value

```yaml
# BEFORE (incorrect):
gradcam:
  target_layer: "model.features[-1]"  # → features[8]: head Conv2dNormActivation, NOT MBConv
```

```yaml
# AFTER (confirmed correct):
gradcam:
  target_layer: "model.features[7][-1]"  # → features.7.1: final MBConv block, Stage 7
```

---

## Introspection Command (Colab Sprint C Verification Step)

Run this at Sprint C setup in Colab before implementing `src/cv/gradcam.py`:

```python
import torchvision

model = torchvision.models.efficientnet_b4(weights=None)

# Step 1: Confirm top-level features structure
for i, (name, mod) in enumerate(model.features.named_children()):
    print(f"features[{name}]: {type(mod).__name__}")

# Step 2: Confirm features[7] sub-blocks
print("\n--- features[7] sub-blocks ---")
for name, mod in model.features[7].named_children():
    print(f"  features.7.{name}: {type(mod).__name__}")

# Step 3: Confirm target_layer resolves correctly
target_layer = model.features[7][-1]
print(f"\nTarget layer: {target_layer}")
print(f"Named path: features.7.{len(list(model.features[7].children())) - 1}")

# Step 4: Smoke-test — verify non-zero, non-NaN saliency
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

model.eval()
target_layers = [model.features[7][-1]]
input_tensor = torch.randn(1, 3, 224, 224)

with GradCAM(model=model, target_layers=target_layers) as cam:
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(1)])

assert not torch.isnan(torch.tensor(grayscale_cam)).any(), "FAIL: NaN in saliency"
assert grayscale_cam.max() > 0, "FAIL: all-zero saliency map"
print("Smoke test PASSED — saliency is non-zero, non-NaN")
```

**Expected output structure:**
```
features[0]: Conv2dNormActivation
features[1]: Sequential
...
features[7]: Sequential
features[8]: Conv2dNormActivation

--- features[7] sub-blocks ---
  features.7.0: MBConv
  features.7.1: MBConv   ← this is model.features[7][-1]

Smoke test PASSED — saliency is non-zero, non-NaN
```

---

## pytorch-grad-cam Usage Contract (Sprint C Reference)

```python
# src/cv/gradcam.py — interface sketch (NOT final code)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# target_layers: list containing exactly one module
# MUST be model.features[7][-1] per config.yaml:gradcam.target_layer
target_layers = [model.features[7][-1]]

with GradCAM(model=model, target_layers=target_layers) as cam:
    grayscale_cam = cam(
        input_tensor=input_tensor,       # [1, 3, 224, 224] float32
        targets=[ClassifierOutputTarget(1)]  # spoof class = 1
    )
    # grayscale_cam: numpy array [1, H, W], values in [0, 1]
    grayscale_cam = grayscale_cam[0]     # [H, W]
```

> **Note:** `use_cuda=False` is the default when CUDA is unavailable. On HF Spaces (CPU-only), no flag is needed — pytorch-grad-cam will use CPU automatically.

---

## Alternatives Rejected

| Alternative | Reason Rejected |
|------------|----------------|
| `model.features[-1]` (= `features[8]`, head conv) | Not an MBConv block — violates FR-CV-010 literal requirement |
| `model.features[7][-1].block[-1][0]` (projection Conv2d) | Sub-layer targeting adds complexity with no accuracy benefit for GradCAM outputs |
| `model.features[6][-1]` (Stage 6 last block) | Second-to-last MBConv stage — less semantic specificity |
| EigenGradCAM | Higher quality but ≥ 3× latency — violates FR-CV-015 ≤ 3,000 ms on CPU |

---

## config.yaml Impact

| Key | Old Value | New Value | Rationale |
|-----|-----------|-----------|-----------|
| `gradcam.target_layer` | `"model.features[-1]"` | `"model.features[7][-1]"` | Correct final MBConv block per FR-CV-010 |

---

## Sprint C Entry Gate Update

**Q4 Status:** ✅ RESOLVED — `model.features[7][-1]` confirmed as Grad-CAM target.  
**Empirical confirmation required:** Run introspection smoke-test at Sprint C setup (see above). If assertion fails, escalate immediately — do not implement downstream FRs.

---

*[DRAFT — Phase 2 — Pending V.E.R.I.F.Y.]*
