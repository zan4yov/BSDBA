# ADR-0006 — Q6 Resolution: UI Framework Lock — Gradio 4.x
**Document:** DSDBA-SRS-2026-002 v2.1 | Phase 2 Deliverable  
**Authors:** Ferel, Safa — ITS Informatics | KCVanguard ML Workshop  
**Date:** 2026-03-18  
**Status:** ✅ RESOLVED — blocks Sprint E (FR-DEP-001 through FR-DEP-009)  
**SRS References:** FR-DEP-001, FR-DEP-002, FR-DEP-003, FR-DEP-004, FR-DEP-005, FR-DEP-006, FR-DEP-007, FR-NLP-006  
**Previous Recommendation:** `docs/adr/phase1-backlog.md` §Q6 (🟡 RECOMMENDED → now ✅ LOCKED)

---

## Decision

**Selected framework:** Gradio 4.x (pinned: `gradio==4.7.1` per `requirements.txt`)  
**Rejected alternative:** Streamlit  
**config.yaml key (locked):** `deployment.framework: gradio` — no further changes permitted after Phase 2.

---

## Rationale

### Requirement-by-Requirement Evaluation

| SRS Requirement | Gradio 4.x | Streamlit | Decision |
|-----------------|-----------|-----------|---------|
| **FR-DEP-001** — Gradio 4.x UI | ✅ Native — `gr.Blocks` or `gr.Interface` | ❌ Violates FR-DEP-001 as written | Gradio required by SRS literal text |
| **FR-DEP-002** — Audio upload widget, 20 MB limit | ✅ `gr.Audio(type="filepath", max_length=20)` built-in | ⚠️ `audio_recorder_streamlit` or `st.file_uploader` — no native audio playback | Gradio preferred |
| **FR-DEP-003** — NLP explanation panel + warning badge | ✅ `gr.Textbox` + `gr.HTML` for conditional badge | ✅ `st.markdown` + `st.warning` equivalent | Tie |
| **FR-DEP-004** — Heatmap panel (Grad-CAM PNG) | ✅ `gr.Image` accepts `PIL.Image` / `np.ndarray` directly | ✅ `st.image` equivalent | Tie |
| **FR-DEP-005** — Frequency band bar chart | ✅ `gr.BarPlot` — no matplotlib import needed in app.py | ⚠️ `st.bar_chart` or requires `plotly` import | Gradio preferred (module boundary rule) |
| **FR-DEP-006** — Stage 2 renders before Stage 3 NLP | ✅ `yield` generator in Gradio fn streams partial results natively | ⚠️ `st.empty()` + threading — architectural workaround required | Gradio strongly preferred |
| **FR-DEP-007** — E2E ≤ 15,000 ms | ✅ Streaming prevents blocking on NLP; Stage 2 visible within 1,500 ms | ⚠️ Full page re-run model delays Stage 2 display until NLP completes | Gradio preferred |
| **FR-NLP-006** — Non-blocking NLP call | ✅ `yield` generator: yields Stage 2 result first, then appends NLP text | ⚠️ Requires explicit threading + placeholder update | Gradio strongly preferred |
| **HF Spaces deployment** (FR-DEP-001) | ✅ First-party Spaces support — zero extra config | ⚠️ Supported but secondary; requires `requirements.txt` SDK override | Gradio preferred |

### Definitive Rationale Points

1. **FR-DEP-001 is non-negotiable:** The SRS literally names Gradio 4.x. Selecting Streamlit would require a formal SRS amendment — out of scope for Phase 2.

2. **Streaming satisfies two SHALL requirements simultaneously:** A single `yield`-based generator in `app.py` delivers Stage 2 results (label + heatmap + band chart) before the async NLP call returns, fulfilling both FR-NLP-006 ("NLP call is non-blocking") and FR-DEP-006 ("Stage 2 renders before Stage 3 NLP"). No threading primitives or separate async wiring are needed.

3. **Module boundary compliance:** `gr.BarPlot` renders the 4-band bar chart entirely from a Python dict — no `matplotlib` import in `app.py`. This preserves the `.cursorrules` module boundary: `app.py → UI wiring only, no pipeline logic directly`.

4. **`config.yaml` is already committed:** Changing to Streamlit would require updating `deployment.framework`, `deployment.gradio_version`, adding `streamlit` to `requirements.txt`, and removing `gradio==4.7.1`. No technical benefit justifies this disruption.

5. **HF Spaces SDK default is Gradio:** The `README.md` SDK field defaults to `gradio`, saving one configuration step at FR-DEP-001 deployment.

---

## Rejected Alternative: Streamlit

| Criterion | Impact of Selecting Streamlit |
|-----------|-------------------------------|
| FR-DEP-001 compliance | Direct SRS violation (requires amendment) |
| FR-NLP-006 + FR-DEP-006 | Requires `st.empty()` + `threading.Thread` workaround; more complex, less reliable |
| Audio widget | No native playback; requires `audio_recorder_streamlit` extra dependency |
| HF Spaces | Requires `sdk: streamlit` in Space `README.md`; secondary support |

---

## Implementation Constraints (Sprint E Reference)

```python
# app.py — structural contract (NOT final code)
# SRS: FR-DEP-001, FR-DEP-006, FR-NLP-006

import gradio as gr  # gradio==4.7.1

def run_pipeline(audio_path: str):
    """
    Generator function that yields progressive results.
    Stage 2 (label + heatmap + bands) yields first.
    Stage 3 (NLP explanation) yields second.
    This single 'yield' approach satisfies FR-DEP-006 + FR-NLP-006.
    """
    # Stage 1: DSP (≤ 500 ms)
    tensor = preprocess_audio(Path(audio_path))

    # Stage 2: CV Inference + Grad-CAM (≤ 1,500 + 3,000 ms = ≤ 4,500 ms)
    label, confidence = run_inference(tensor)
    heatmap_path, band_pct = run_gradcam(tensor, model)

    # Yield Stage 2 result immediately — FR-DEP-006
    yield label, confidence, heatmap_path, band_pct, gr.update(value="Generating explanation...")

    # Stage 3: NLP Explanation (≤ 8,000 ms, non-blocking from user perspective)
    explanation = generate_explanation(label, confidence, band_pct)

    # Yield Stage 3 result
    yield label, confidence, heatmap_path, band_pct, explanation

with gr.Blocks() as demo:
    audio_input = gr.Audio(type="filepath", label="Upload Audio (WAV/FLAC, max 20 MB)")
    label_out   = gr.Textbox(label="Prediction")
    conf_out    = gr.Number(label="Confidence")
    heatmap_out = gr.Image(label="Grad-CAM Heatmap")
    band_out    = gr.BarPlot(...)       # FR-DEP-005
    explain_out = gr.Textbox(label="AI Explanation")  # FR-DEP-003

    audio_input.change(
        fn=run_pipeline,
        inputs=audio_input,
        outputs=[label_out, conf_out, heatmap_out, band_out, explain_out],
    )
```

> **Note on `gr.BarPlot`:** Available in Gradio 4.x. Accepts a pandas DataFrame or dict. No matplotlib dependency in `app.py`.

---

## config.yaml Impact

No changes required. The following keys are now officially locked:

| Key | Locked Value | Basis |
|-----|--------------|-------|
| `deployment.framework` | `gradio` | FR-DEP-001, this ADR |
| `deployment.gradio_version` | `"4"` | Pinned `gradio==4.7.1` in requirements.txt |
| `deployment.auth_required` | `false` | FR-DEP-001: public access |

---

## Sprint E Entry Gate Update

**Q6 Status:** ✅ RESOLVED — Gradio 4.x locked as UI framework. `deployment.framework: gradio` in `config.yaml` is now the binding contract for Sprint E.  
**Sprint E Entry Criteria satisfied:** Q6 resolved; Sprint E may begin after Sprint D Gate Check passes.

---

*[DRAFT — Phase 2 — Pending V.E.R.I.F.Y.]*
