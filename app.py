"""
Module: app
SRS Reference: FR-DEP-001, FR-DEP-002, FR-DEP-003, FR-DEP-004, FR-DEP-005,
               FR-DEP-006, FR-DEP-007, FR-DEP-008, FR-DEP-009, FR-DEP-010
SDLC Phase: Phase 3 — Sprint E (implementation begins Chain 09)
Sprint: E
Pipeline Stage: Deployment
Interface Contract:
  Input:  None (build_demo() called once at startup)
  Output: gr.Blocks — Gradio 4.x application object
Latency Target: Stage 2 result visible within 4,500 ms; E2E ≤ 15,000 ms [FR-DEP-007]
Open Questions Resolved: Q6 — Gradio 4.x locked (ADR-0006). deployment.framework: gradio.
Open Questions Blocking: None
MCP Tools Used: None (UI wiring only — no pipeline logic in app.py)
AI Generated: true
Verified (V.E.R.I.F.Y.): false
Author: Ferel / Safa
Date: 2026-03-19
"""

# [DRAFT — Phase 3 — Sprint E — Pending V.E.R.I.F.Y.]
# Implementation: Chain 09 — Sprint E
# Module boundary: UI WIRING ONLY — no pipeline logic permitted directly. [.cursorrules]
# Streaming generator satisfies FR-DEP-006 (Stage 2 renders before NLP) + FR-NLP-006 (non-blocking)

from __future__ import annotations

import gradio as gr  # gradio==4.7.1 [FR-DEP-001, Q6 RESOLVED, ADR-0006]


def build_demo() -> gr.Blocks:
    """Build and return the Gradio 4.x Blocks application.

    UI Components (all required for SHALL FRs):
        - gr.Audio     → audio file upload widget, max 20 MB [FR-DEP-002]
        - gr.Textbox   → prediction label (bonafide/spoof) [FR-DEP-003]
        - gr.Number    → confidence score ∈ (0.0, 1.0)
        - gr.Image     → Grad-CAM heatmap PNG [FR-DEP-004]
        - gr.BarPlot   → 4-band frequency attribution chart [FR-DEP-005]
        - gr.Textbox   → AI explanation (NLP output) [FR-DEP-003]
        - gr.HTML      → AI-unavailable warning badge (conditional) [FR-DEP-003]
        - Demo samples → 2 bonafide + 2 spoof examples [FR-DEP-008]
        - About section → dataset citation + architecture overview [FR-DEP-009]

    Pipeline wiring (generator — yields twice per request):
        yield 1 → Stage 2 result (label, confidence, heatmap, band_pct,
                  "Generating explanation...") [FR-DEP-006 — Stage 2 before NLP]
        yield 2 → Stage 3 NLP explanation appended [FR-NLP-006 — non-blocking]

    Returns:
        gr.Blocks: Configured Gradio application object.
                   Call demo.launch(share=False) in __main__ block.

    Latency Target:
        Stage 2 visible: ≤ 500 + 1,500 + 3,000 = ≤ 5,000 ms [FR-DEP-006]
        Full E2E: ≤ 15,000 ms [FR-DEP-007]
    """
    raise NotImplementedError(
        "build_demo() not implemented — stub only. "
        "Implement in Chain 09 (Sprint E)."
    )


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=False)
