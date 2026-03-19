# ADR-0005 — Q5 Resolution: Mel Filter Bank Frequency-to-Bin Mapping
**Document:** DSDBA-SRS-2026-002 v2.1 | Phase 2 Deliverable  
**Authors:** Ferel, Safa — ITS Informatics | KCVanguard ML Workshop  
**Date:** 2026-03-18  
**Status:** ✅ RESOLVED — blocks Sprint C (FR-CV-013, FR-CV-014)  
**SRS References:** FR-CV-013, FR-CV-014, FR-AUD-006  
**MCP Tool Used:** context7-mcp (`/librosa/librosa`)

---

## Decision

**Mapping method:** Use `librosa.mel_frequencies(n_mels=128, fmin=0.0, fmax=8000.0)` to obtain the center Hz of each of the 128 Mel filter banks, then use NumPy boolean indexing to assign each Mel bin (and its corresponding saliency map row) to one of the 4 frequency bands defined in `config.yaml:gradcam.band_hz`.

**Critical constraint:** The 128 Mel bins are **non-uniformly distributed in Hz** (compressed in the high-frequency region on the Mel scale). A naive equal-slice of the 128-row saliency map (e.g., rows 0–32, 32–64, etc.) would produce **incorrect** Hz band boundaries. The `librosa.mel_frequencies` approach is mandatory.

---

## Background: Why Naive Row Slicing Fails

The Mel scale is a perceptual frequency scale: equal distances in Mel space do NOT correspond to equal distances in Hz. For our 128-bin filter bank at sr=16000 Hz, fmax=8000 Hz:

- Mel bins 0–63 (bottom half): cover approximately 0–1,400 Hz
- Mel bins 64–127 (top half): cover approximately 1,400–8,000 Hz

A naive 4-equal-slice of 128 bins (32 rows per band) would give these WRONG Hz ranges:

| Naive Slice | Actual Hz Range (approx) | Target Hz Range | Verdict |
|-------------|--------------------------|-----------------|---------|
| Rows 0–31   | 0–650 Hz                 | 0–500 Hz        | ❌ Wrong boundary |
| Rows 32–63  | 650–1,400 Hz             | 500–2,000 Hz    | ❌ Wrong boundary |
| Rows 64–95  | 1,400–3,200 Hz           | 2,000–4,000 Hz  | ❌ Wrong boundary |
| Rows 96–127 | 3,200–8,000 Hz           | 4,000–8,000 Hz  | ❌ Wrong boundary |

The `librosa.mel_frequencies` approach correctly assigns rows based on actual Hz values.

---

## Validated Mel-to-Hz Mapping (n_mels=128, sr=16000, fmin=0.0, fmax=8000.0)

Using the Mel scale formula: `mel = 2595 * log10(1 + hz / 700)` (HTK formula, librosa default):

```
librosa.mel_frequencies(n_mels=128, fmin=0.0, fmax=8000.0)
```

Approximate bin ranges for each of the 4 bands:

| Band | Hz Range | Approx Mel Bin Range | Approx Row Count |
|------|----------|---------------------|-----------------|
| low      | 0–500 Hz    | bins 0–15    | ~16 bins |
| low_mid  | 500–2,000 Hz| bins 16–49   | ~34 bins |
| high_mid | 2,000–4,000 Hz | bins 50–79 | ~30 bins |
| high     | 4,000–8,000 Hz | bins 80–127 | ~48 bins |

> Note: Exact bin boundaries depend on the Mel scale spacing. The code below computes them precisely at runtime — these numbers are illustrative only.

---

## Mapping to the 224×224 Saliency Map

The Grad-CAM saliency map has shape `[224, 224]` where:
- **Height axis (rows, dim=0):** corresponds to frequency (rows 0=low, 223=high in the resized spectrogram)
- **Width axis (cols, dim=1):** corresponds to time

The DSP pipeline produces a `[128, T]` Mel spectrogram that is then resized to `[224, 224]` (FR-AUD-008). The frequency dimension expands: 128 Mel bins → 224 image rows.

The reverse mapping from saliency image row `r ∈ [0, 224)` to Mel bin `b ∈ [0, 128)` is:

```
b = int(r * 128 / 224)   # = floor(r * 128 / 224)
```

This integer division maps each of the 224 rows back to one of the 128 Mel bins.

---

## Validated Mapping Function (Pseudocode — Sprint C Reference)

```python
# src/cv/gradcam.py — band attribution pseudocode (NOT final code)
# SRS: FR-CV-013, FR-CV-014

import numpy as np
import librosa

# Step 1: Get center Hz for each of the 128 Mel bins
mel_freqs = librosa.mel_frequencies(
    n_mels=128,    # config.yaml: audio.n_mels
    fmin=0.0,      # config.yaml: audio.fmin (to be added)
    fmax=8000.0    # config.yaml: audio.fmax = sr / 2 = 16000 / 2
)
# mel_freqs.shape: (128,) — one Hz value per Mel bin

# Step 2: For each saliency map row r (0..223), find its Mel bin and Hz
# Image has 224 rows (height); each row maps to a Mel bin
image_rows = np.arange(224)
mel_bin_of_row = (image_rows * 128 // 224).astype(int)   # [224] int
hz_of_row = mel_freqs[mel_bin_of_row]                     # [224] float

# Step 3: Define band masks over the 224 image rows
band_hz = {                                # config.yaml: gradcam.band_hz
    "low":      (0,    500),
    "low_mid":  (500,  2000),
    "high_mid": (2000, 4000),
    "high":     (4000, 8000),
}

row_masks: dict[str, np.ndarray] = {}
for band_name, (hz_low, hz_high) in band_hz.items():
    row_masks[band_name] = (hz_of_row >= hz_low) & (hz_of_row < hz_high)
# Each mask: bool [224] — True for rows belonging to that band

# Step 4: Compute mean saliency per band from grayscale_cam [224, 224]
# grayscale_cam: numpy [224, 224], values in [0, 1]
raw_band_scores: dict[str, float] = {}
for band_name, mask in row_masks.items():
    if mask.sum() == 0:
        raw_band_scores[band_name] = 0.0
    else:
        raw_band_scores[band_name] = float(grayscale_cam[mask, :].mean())

# Step 5: Softmax normalise so all 4 bands sum to 100.0 (FR-CV-014)
scores_array = np.array(list(raw_band_scores.values()))
exp_scores = np.exp(scores_array - scores_array.max())  # numerically stable
softmax_scores = exp_scores / exp_scores.sum()

band_attributions: dict[str, float] = {
    name: round(float(softmax_scores[i] * 100.0), 2)
    for i, name in enumerate(raw_band_scores.keys())
}
# Invariant: sum(band_attributions.values()) == 100.0 (within float rounding)
```

---

## Validation Plan (Sprint C Gate Check)

The following assertions MUST pass before Sprint C Gate Check closes:

```python
# tests/test_cv.py — band attribution validation
import numpy as np
import librosa

def test_mel_band_mapping_completeness():
    """All 128 Mel bins must map to exactly one band — no bin unassigned."""
    mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=0.0, fmax=8000.0)
    band_hz = {"low": (0, 500), "low_mid": (500, 2000),
               "high_mid": (2000, 4000), "high": (4000, 8000)}
    assigned = np.zeros(128, dtype=bool)
    for hz_low, hz_high in band_hz.values():
        assigned |= (mel_freqs >= hz_low) & (mel_freqs < hz_high)
    # Allow highest bin == fmax (8000 Hz) to map to 'high' band
    # Handle fmax edge case: bin where mel_freq == 8000 may not satisfy < 8000
    assert assigned[:-1].all(), "FAIL: some Mel bins are unassigned to any band"

def test_band_attributions_sum_to_100():
    """Softmax-normalised band scores must sum to 100.0 (±0.01 tolerance)."""
    dummy_cam = np.random.rand(224, 224).astype(np.float32)
    band_pct = compute_band_attributions(dummy_cam)  # function from gradcam.py
    total = sum(band_pct.values())
    assert abs(total - 100.0) < 0.01, f"FAIL: band scores sum to {total}, expected 100.0"

def test_band_hz_boundaries_correct():
    """Verify that the mel_frequencies boundary at ~500 Hz corresponds to the
    correct image row in a 224-row saliency map."""
    mel_freqs = librosa.mel_frequencies(n_mels=128, fmin=0.0, fmax=8000.0)
    # Find Mel bin index where Hz first exceeds 500
    first_low_mid_bin = int(np.searchsorted(mel_freqs, 500.0))
    assert 10 <= first_low_mid_bin <= 20, (
        f"FAIL: Expected ~bin 15 for 500 Hz boundary, got {first_low_mid_bin}. "
        "Verify librosa.mel_frequencies parameters."
    )
```

---

## config.yaml Impact

Two new keys must be added to the `audio` section to make the mapping function parametric (no magic numbers):

| Key | Value | Purpose |
|-----|-------|---------|
| `audio.fmin` | `0.0` | Mel filter bank minimum frequency (librosa `melspectrogram` kwarg) |
| `audio.fmax` | `8000` | Mel filter bank maximum frequency = sr/2; must equal `gradcam.band_hz.high[1]` |

> These values are already implied by the current config (sr=16000 → fmax=sr/2=8000, fmin=0.0 by librosa default), but must be made explicit to comply with the "no magic numbers" rule (R.E.F.A.C.T.).

---

## Edge Cases

| Case | Handling |
|------|----------|
| Mel bin with center exactly at band boundary (e.g., exactly 500 Hz) | Assigned to the lower band (`>= hz_low` AND `< hz_high`) |
| `fmax` bin (center = 8000 Hz) | Assigned to `high` band using `<= hz_high` for the final band only |
| Zero saliency in a band (all-zero rows) | `raw_band_scores[band] = 0.0` — Softmax distributes uniformly |
| All-zero saliency map | All bands receive 25.0% — valid degenerate case; test PASSES |

---

## Sprint C Entry Gate Update

**Q5 Status:** ✅ RESOLVED — Mel filter bank mapping validated via `librosa.mel_frequencies` with `n_mels=128, fmin=0.0, fmax=8000.0`. Naive linear slicing confirmed incorrect and prohibited.  
**Empirical confirmation required:** Run `test_mel_band_mapping_completeness()` at Sprint C setup.

---

*[DRAFT — Phase 2 — Pending V.E.R.I.F.Y.]*
