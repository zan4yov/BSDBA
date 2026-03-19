# ════════════════════════════════════════════════════════════════════════
# DSDBA Chain 09 — Sprint E: Deployment UI
# SRS: FR-DEP-001–010 | Phase: 4-E | Mode: Composer
# V.E.R.I.F.Y. Level: 2 | BLOCKER: Q6 MUST be resolved (Gradio locked)
# ════════════════════════════════════════════════════════════════════════

@file .cursorrules
@file config.yaml
@file docs/context/session-cheatsheet.md
@file src/audio/dsp.py
@file src/cv/infer.py
@file src/cv/gradcam.py
@file src/nlp/explain.py

CONFIRM: Q6 = ✅ RESOLVED (Gradio 4.x) in session-cheatsheet.md. If not → STOP.

Use context7-mcp to verify Gradio 4.x component API before building UI.
Use huggingface-mcp to configure HF Spaces deployment settings.

IMPLEMENT: app.py  [FR-DEP-001–010]
  Structure (top-to-bottom, no logic outside of functions):

  1. Imports and config loading (at top, ONNX session loaded ONCE here)
     cfg = load_config("config.yaml")
     onnx_session = load_onnx_session(cfg)   # FR-DEP-010: once at startup

  2. run_pipeline(audio_file) -> tuple:  [MAIN PIPELINE FUNCTION]
     - Stage 1: preprocess_audio(audio_file, cfg) → tensor [FR-AUD-001–008]
     - Validate file size ≤ 20 MB before Stage 1 [FR-DEP-002]
     - Stage 2: run_onnx_inference(onnx_session, tensor, cfg) → (label, confidence) [FR-CV-004]
     - Stage 3: run_gradcam(tensor, model, cfg) → (heatmap_path, band_pct) [FR-CV-010–016]
     - Stage 4: asyncio.create_task(generate_explanation(label, conf, band_pct, cfg))
       [FR-NLP-006: async, does NOT block Stage 2 result]
     - Return (label, confidence, heatmap_path, band_pct, explanation_task)

  3. Gradio interface definition:
     - gr.Audio upload: .wav/.flac, max 20 MB [FR-DEP-002]
     - gr.Waveform display [FR-DEP-003]
     - gr.Image for spectrogram [FR-DEP-003]
     - Verdict panel: gr.Label (badge) + gr.Number (confidence %) + gr.HTML (bar) [FR-DEP-004]
     - gr.Image for Grad-CAM overlay [FR-DEP-005]
     - gr.BarPlot for 4-band attribution chart [FR-DEP-005]
     - gr.Textbox for NLP explanation with loading spinner [FR-DEP-006]
     - gr.Markdown "AI-generated explanation (English)" label [FR-DEP-006]

  4. Demo examples (data/samples/):
     - 2 bonafide sample files + 2 spoof files [FR-DEP-008]
     - gr.Examples pointing to data/samples/

  5. About section:
     - gr.Markdown with dataset citation, architecture summary, team credits [FR-DEP-009]
     - Dataset: Abdel-Dayem, M. (2023). Fake-or-Real (FoR) Dataset. Kaggle.

  6. Error handling:
     - File > 20 MB: gr.Warning with message (not crash) [FR-DEP-002]
     - DSDBAError(AUD-001): gr.Warning "Audio too short (< 0.5 s)"
     - DSDBAError(AUD-002): gr.Warning "Unsupported format"
     - Generic exception: gr.Error with generic message (not stack trace) [NFR-Security]

IMPLEMENT src/tests/test_e2e.py:  [V.E.R.I.F.Y. L3 — end-to-end]
  1. test_valid_bonafide_sample: full pipeline with bonafide sample → label + confidence + heatmap
  2. test_valid_spoof_sample: full pipeline with spoof sample → all outputs populated
  3. test_e2e_latency: wall clock for complete pipeline ≤ 15,000 ms [FR-DEP-007]
  4. test_file_too_large: 21 MB mock file → rejected before processing
  5. test_too_short_audio: 0.3 s file → DSDBAError(AUD-001) surfaced as gr.Warning
  6. test_nlp_fallback_does_not_block_cv: CV result available before NLP task resolves

Use huggingface-mcp to generate HF Spaces deployment config (README.md, requirements.txt
should already be correct for Spaces). Verify secrets (QWEN_API_KEY, HF_TOKEN) are
documented in a HF Spaces secrets setup checklist in README.md.