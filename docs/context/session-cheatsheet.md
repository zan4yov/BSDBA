# DSDBA — Session Context Cheatsheet
**Document:** DSDBA-SRS-2026-002 v2.1 | Context Carry v1.0
**Rule:** READ THIS FILE at the start of every Cursor session (@file docs/context/session-cheatsheet.md)
**Rule:** UPDATE THIS FILE at the end of every Cursor session before closing.

---

## CURRENT STATUS

| Field | Value |
|-------|-------|
| **Active SDLC Phase** | Phase 3 — Environment Setup & MCP Configuration COMPLETE |
| **Active Sprint** | Sprint A (entry criteria satisfied — all Qs resolved) |
| **Last Completed** | Phase 3 — Environment Setup & MCP Configuration (Chain 04) |
| **Next Action** | Run Chain 05: Sprint A — `src/audio/dsp.py` implementation |
| **Gate Status** | Phase 3 COMPLETE — Q3 RESOLVED — all stubs created — Sprint A ready |

---

## COMPLETED SRS REQUIREMENTS

| Phase | Deliverable | FRs Addressed | File |
|-------|-------------|---------------|------|
| Phase 0 | Architectural Risk Register | RISK-001/002/003 (FR-AUD-008, FR-CV-010, FR-NLP-006) | `docs/adr/phase0-risk-register.md` |
| Phase 0 | Pipeline Architecture Diagram | All FR-AUD/CV/NLP/DEP (scoped) | `docs/adr/phase0-pipeline-diagram.md` |
| Phase 0 | ADR-0001 MCP Tool Selection | All FR groups + NFR-Security, NFR-Maintainability | `docs/adr/phase0-mcp-selection.md` |
| Phase 0 | Master Implementation Plan | Phases 0–7, all FRs | `docs/adr/phase0-mip.md` |
| Phase 0 | Q3 and Q4 Initial Assessment | Sprint B / Sprint C gate tracking | session-cheatsheet.md (this file) |
| Phase 1 | Prioritised Sprint Backlog | All 46 FRs decomposed by sprint + SHOULD/MAY deferral notes | `docs/adr/phase1-backlog.md` |
| Phase 1 | Requirement Traceability Matrix | All 46 FRs → module file → sprint → blocking Q | `docs/adr/phase1-rtm.md` |
| Phase 1 | Q6 Framework Recommendation | FR-DEP-001, FR-NLP-006, FR-DEP-006 — Gradio 4.x recommended | `docs/adr/phase1-backlog.md` §Q6 |
| Phase 2 | Q4 Grad-CAM Target Layer | FR-CV-010/011 — `model.features[7][-1]` confirmed (Stage 7 final MBConv) | `docs/adr/phase2-gradcam-target-layer.md` |
| Phase 2 | Q5 Mel Filter Bank Mapping | FR-CV-013/014 — `librosa.mel_frequencies(n_mels=128, fmin=0.0, fmax=8000.0)` validated | `docs/adr/phase2-mel-band-mapping.md` |
| Phase 2 | Q6 UI Framework Lock | FR-DEP-001–009 — Gradio 4.x officially locked, ADR-0006 | `docs/adr/phase2-ui-framework.md` |
| Phase 2 | Module Interface Contracts | All 5 pipeline module signatures locked (dsp, infer, gradcam, explain, app) | `docs/adr/phase2-interface-contracts.md` |
| Phase 2 | config.yaml Finalised | `gradcam.target_layer` corrected; `audio.fmin/fmax` added; `deployment.framework` locked | `config.yaml` |
| Phase 3 | Q3 VRAM Resolution (ADR-0008) | FR-CV-003, NFR-Performance — batch_size=16 feasible on T4 | `docs/adr/phase3-colab-vram.md` |
| Phase 3 | Colab Training Notebook Scaffold | FR-CV-003–008, Q3 verification protocol (Cell 3) | `notebooks/dsdba_training.ipynb` |
| Phase 3 | README.md GitHub Setup | All FRs (project overview, pipeline diagram, installation, architecture) | `README.md` |
| Phase 3 | Source File Stubs (10 modules) | All FRs — module docstrings + locked interface signatures | `src/**/*.py`, `app.py` |
| Phase 3 | `src/utils/errors.py` Full Implementation | FR-AUD-001, FR-AUD-005, FR-CV-004, FR-CV-010 | `src/utils/errors.py` |

---

## OPEN QUESTIONS STATUS

| Q# | Question | Status | Blocks |
|----|----------|--------|--------|
| Q1 | Qwen 2.5 API adoption | RESOLVED — Qwen 2.5 (Alibaba Cloud) confirmed | — |
| Q2 | FoR dataset variant | RESOLVED — for-2sec (2.0 s clips, 32,000 samples) | — |
| Q3 | EfficientNet-B4 VRAM on Colab Free T4 | RESOLVED — batch_size=16 feasible; est. peak ~4–6 GB on 15 GB T4. `gradient_checkpointing=true` confirmed (ADR-0008). Empirical verification: notebook Cell 3 at Sprint B start. | — |
| Q4 | Grad-CAM target layer path | RESOLVED — `model.features[7][-1]` (Stage 7 final MBConv, named `features.7.1`). Smoke-test at Sprint C setup. | — |
| Q5 | Mel bin-to-Hz mapping validation | RESOLVED — `librosa.mel_frequencies(n_mels=128, fmin=0.0, fmax=8000.0)` + row index `int(r*128//224)`. Naive linear slice prohibited. | — |
| Q6 | Gradio vs Streamlit framework lock | RESOLVED — Gradio 4.x locked (ADR-0006). `deployment.framework: gradio` is the binding contract for Sprint E. | — |
| Q7 | EER scoring protocol (scikit vs ASVspoof) | OPEN | Phase 7 accuracy validation |

### Q3 Resolution (Phase 3 — Chain 04)

**Question:** Is EfficientNet-B4 with batch_size=16 feasible on Colab Free Tier T4?

**Resolution (ADR-0008 — `docs/adr/phase3-colab-vram.md`):**
- EfficientNet-B4 parameter count: ~19.3M parameters.
- Peak VRAM estimated at ~4–6 GB (forward + backward, batch=16, mixed precision fp16).
- With `gradient_checkpointing=true`: reduces peak by ~30% → ~3–4 GB.
- Colab Free Tier T4 VRAM: **15 GB**.
- Conclusion: batch_size=16 is **feasible** with wide safety margin (~9–11 GB headroom).
- `config.yaml:training.gradient_checkpointing: true` confirmed (safety precaution, no change needed).
- Fallback protocol: if empirical Cell 3 shows > 11 GB → reduce batch_size to 8.

**Status:** RESOLVED — empirical confirmation via `notebooks/dsdba_training.ipynb` Cell 3 at Sprint B start.

### Q4 Assessment (Phase 0 — Chain 01, RESOLVED Phase 2 — Chain 03)

**Status:** RESOLVED — `model.features[7][-1]` confirmed (ADR-0004). Smoke-test deferred to Sprint C.

---

## FILES CREATED THIS PROJECT

| File | Created | Phase | Purpose |
|------|---------|-------|---------|
| `.cursorrules` | Session 00 | 0 | Vibe Coding master configuration |
| `config.yaml` | Session 00 | 0 | Hyperparameter single source of truth |
| `requirements.txt` | Session 00 | 0 | Pinned dependency manifest |
| `docs/context/session-cheatsheet.md` | Session 00 | 0 | Session context carry document |
| `docs/adr/phase0-risk-register.md` | Session 01 | 0 | Top 3 architectural risks |
| `docs/adr/phase0-pipeline-diagram.md` | Session 01 | 0 | Mermaid pipeline architecture diagram |
| `docs/adr/phase0-mcp-selection.md` | Session 01 | 0 | ADR-0001 MCP tool selection rationale |
| `docs/adr/phase0-mip.md` | Session 01 | 0 | Master Implementation Plan (Phases 0–7) |
| `docs/adr/phase1-backlog.md` | Session 02 | 1 | Sprint backlog (all 46 FRs) + Q6 recommendation |
| `docs/adr/phase1-rtm.md` | Session 02 | 1 | Full Requirement Traceability Matrix |
| `docs/adr/phase2-gradcam-target-layer.md` | Session 03 | 2 | ADR-0004: Q4 resolved — Grad-CAM target layer `model.features[7][-1]` |
| `docs/adr/phase2-mel-band-mapping.md` | Session 03 | 2 | ADR-0005: Q5 resolved — Mel filter bank Hz mapping + validation plan |
| `docs/adr/phase2-ui-framework.md` | Session 03 | 2 | ADR-0006: Q6 resolved — Gradio 4.x locked as UI framework |
| `docs/adr/phase2-interface-contracts.md` | Session 03 | 2 | ADR-0007: Binding interface contracts for all 5 pipeline modules |
| `docs/adr/phase3-colab-vram.md` | Session 04 | 3 | ADR-0008: Q3 resolved — batch_size=16 feasible on T4 |
| `README.md` | Session 04 | 3 | GitHub repo setup — pipeline diagram, install, architecture, credits |
| `notebooks/dsdba_training.ipynb` | Session 04 | 3 | Colab scaffold (6 cells) — env setup, GPU verify, Q3 VRAM test, HF login, dataset, training placeholder |
| `src/audio/dsp.py` | Session 04 | 3 | Sprint A stub — `preprocess_audio()` + `preprocess_batch()` signatures |
| `src/cv/model.py` | Session 04 | 3 | Sprint B stub — `build_model()` signature |
| `src/cv/train.py` | Session 04 | 3 | Sprint B stub — `train()` signature |
| `src/cv/infer.py` | Session 04 | 3 | Sprint B stub — `run_inference()` signature |
| `src/cv/gradcam.py` | Session 04 | 3 | Sprint C stub — `run_gradcam()` + `get_raw_saliency()` signatures |
| `src/nlp/explain.py` | Session 04 | 3 | Sprint D stub — `generate_explanation()` signature + rule-based templates |
| `src/utils/config.py` | Session 04 | 3 | Pydantic config loader — `DSDBAConfig` + `load_config()` |
| `src/utils/logger.py` | Session 04 | 3 | Structured JSON logger — `get_logger()` factory |
| `src/utils/errors.py` | Session 04 | 3 | FULL IMPL — `ErrorCode` enum + `DSDBAError` dataclass |
| `app.py` | Session 04 | 3 | Sprint E stub — `build_demo()` signature (Gradio 4.x wiring) |

---

## PIPELINE CONTRACT (Reference — Do Not Change)

| Stage | Module | Input | Output | Latency |
|-------|--------|-------|--------|---------|
| Audio DSP | src/audio/dsp.py | WAV/FLAC file | [3,224,224] float32 tensor | <= 500 ms |
| CV Inference | src/cv/infer.py | [3,224,224] float32 tensor | (label, confidence float) | <= 1,500 ms (ONNX) |
| XAI Grad-CAM | src/cv/gradcam.py | [3,224,224] tensor + model | heatmap PNG + band_pct[4] | <= 3,000 ms |
| NLP Explanation | src/nlp/explain.py | label + confidence + band_pct[4] | English paragraph (3–5 sentences) | <= 8,000 ms |

---

## TECH DEBT LOG

| Session | Item | SRS Ref | Priority |
|---------|------|---------|---------|
| Session 03 | [TECH DEBT: Grad-CAM smoke-test (non-zero saliency assertion) deferred to Sprint C setup — Colab env required] | FR-CV-010 | HIGH — must pass before Sprint C Gate Check |
| Session 03 | [TECH DEBT: `resampy==0.4.2` version pin not verified against librosa 0.10.1 compatibility — verify in Sprint A Colab setup] | FR-AUD-002 | MEDIUM — fallback: change `audio.resampling_method` to `soxr_hq` if resampy fails |
| Session 04 | [TECH DEBT: `notebooks/dsdba_training.ipynb` Cell 4 + Cell 5 use placeholder HF dataset repo ID `your-username/fake-or-real` — fill in actual repo ID before Sprint B] | FR-CV-007 | MEDIUM — blocks dataset download in Colab |

---

## SESSION HISTORY

### Session 00 — Project Bootstrap
**Date:** [DATE]
**Status:** COMPLETE — Setup only
**Actions:** Created .cursorrules, config.yaml, requirements.txt, session-cheatsheet.md
**Next:** Run Chain 01 in Cursor Agent Mode

### Session 01 — Phase 0: Architecture Design (Chain 01)
**Date:** 2026-03-18
**Status:** COMPLETE
**Actions:**
- Created `docs/adr/phase0-risk-register.md` (RISK-001/002/003 with SRS coupling points)
- Created `docs/adr/phase0-pipeline-diagram.md` (Mermaid 5-stage pipeline with typed interface arrows)
- Created `docs/adr/phase0-mcp-selection.md` (ADR-0001: context7 / huggingface-skills / stitch)
- Created `docs/adr/phase0-mip.md` (Master Implementation Plan Phases 0–7 with FR checkbox lists)
- Updated Q3 status: LIKELY FEASIBLE (batch_size=16 on T4, 15 GB VRAM)
- Updated Q4 status: CANDIDATE CONFIRMED (model.features[-1] path needs Sprint C introspection)
**FRs Addressed (scoped):** All FR groups documented in MIP; no code FRs implemented yet
**Next:** Run Chain 02 — Phase 1: Backlog & Environment Setup

### Session 02 — Phase 1: Requirements & Backlog Definition (Chain 02)
**Date:** 2026-03-18
**Status:** COMPLETE
**Actions:**
- Created `docs/adr/phase1-backlog.md` — all 46 FRs decomposed into 5 sprints (A–E) with SHALL/SHOULD/MAY priority, deferral notes, and blockers
- Created `docs/adr/phase1-rtm.md` — full Requirement Traceability Matrix: FR-ID to module file, sprint, blocking Q, config key, status
- Added Q6 framework recommendation to `phase1-backlog.md` — Gradio 4.x recommended
- Confirmed `requirements.txt` is fully pinned (no version ranges) — NFR-Security compliant
- Confirmed `config.yaml` is the single source of truth — all 46 FRs have config keys where applicable
**FRs Addressed:** All 46 FR-IDs traced to implementation modules and sprints
**Tech Debt Logged:** None this session
**Next:** Run Chain 03 — Phase 2: System Design

### Session 03 — Phase 2: System Design & Technical Specification (Chain 03)
**Date:** 2026-03-18
**Status:** COMPLETE
**Actions:**
- Created `docs/adr/phase2-gradcam-target-layer.md` (ADR-0004) — Q4 RESOLVED: Grad-CAM target confirmed as `model.features[7][-1]` (Stage 7 final MBConv, named `features.7.1`). Smoke-test script provided.
- Created `docs/adr/phase2-mel-band-mapping.md` (ADR-0005) — Q5 RESOLVED: Mel filter bank mapping via `librosa.mel_frequencies(n_mels=128, fmin=0.0, fmax=8000.0)`. Naive linear row slicing prohibited.
- Created `docs/adr/phase2-ui-framework.md` (ADR-0006) — Q6 RESOLVED: Gradio 4.x officially locked.
- Created `docs/adr/phase2-interface-contracts.md` (ADR-0007) — Binding function signatures for all 5 pipeline modules.
- Updated `config.yaml`: `gradcam.target_layer` corrected; `audio.fmin/fmax` added; config marked LOCKED.
- Updated `requirements.txt`: Added `resampy==0.4.2`.
- Updated Q4, Q5, Q6 to RESOLVED.
**FRs Addressed:** FR-CV-010 (Q4), FR-CV-013/014 (Q5), FR-DEP-001 (Q6), all interface contracts locked
**Tech Debt Logged:**
- Grad-CAM smoke-test deferred to Sprint C setup (FR-CV-010)
- resampy==0.4.2 compatibility unverified (FR-AUD-002)
**Next:** Run Chain 04 — Phase 3: Environment Setup

### Session 04 — Phase 3: Environment Setup & MCP Configuration (Chain 04)
**Date:** 2026-03-19
**Status:** COMPLETE
**Actions:**
- Created `docs/adr/phase3-colab-vram.md` (ADR-0008) — Q3 RESOLVED: batch_size=16 feasible on T4 (est. peak ~4–6 GB, 15 GB T4). VRAM analysis + fallback protocol documented.
- Created `notebooks/dsdba_training.ipynb` — 6-cell Colab scaffold: env setup (Cell 1), GPU verify (Cell 2), Q3 VRAM test (Cell 3), HF login (Cell 4), FoR dataset download (Cell 5), training placeholder (Cell 6).
- Created `README.md` — GitHub repo with pipeline diagram, installation, training, demo, architecture table, team credits.
- Stubbed 10 source files (module docstrings + interface signatures from ADR-0007): `src/audio/dsp.py`, `src/cv/model.py`, `src/cv/train.py`, `src/cv/infer.py`, `src/cv/gradcam.py`, `src/nlp/explain.py`, `src/utils/config.py`, `src/utils/logger.py`, `src/utils/errors.py`, `app.py`.
- Fully implemented `src/utils/errors.py`: `ErrorCode` (str Enum) + `DSDBAError` (dataclass Exception).
- Fully implemented `src/utils/config.py`: Pydantic `DSDBAConfig` root model + `load_config()` with validators.
- Fully implemented `src/utils/logger.py`: `_StructuredJSONFormatter` + `get_logger()` factory.
- Updated Q3 to RESOLVED in this cheatsheet.
**FRs Addressed:** FR-CV-003 (Q3), all stub modules scaffolded for FR-AUD-001–011, FR-CV-001–016, FR-NLP-001–009, FR-DEP-001–010
**Tech Debt Logged:**
- Notebook Cell 4 + Cell 5 use placeholder HF dataset repo ID — fill before Sprint B (FR-CV-007)
**Next:** Run Chain 05 — Sprint A: `src/audio/dsp.py` implementation (FR-AUD-001–011)
