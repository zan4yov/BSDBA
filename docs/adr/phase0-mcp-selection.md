# ADR-0001 — MCP Tool Selection for DSDBA Pipeline
**Document:** DSDBA-SRS-2026-002 v2.1 | Architectural Decision Record  
**Authors:** Ferel, Safa — ITS Informatics | KCVanguard ML Workshop  
**Date:** 2026-03-18  
**Status:** ACCEPTED

---

## Context

The DSDBA project requires three distinct categories of external tooling during development and at runtime:

1. **Library documentation access** — PyTorch, librosa, pytorch-grad-cam, Gradio, ONNX Runtime APIs change frequently. Inline documentation lookup is error-prone and slow.
2. **Model Hub and Spaces access** — EfficientNet-B4 checkpoints, FoR dataset, and the final HF Space deployment require authenticated Hugging Face Hub operations.
3. **LLM orchestration at inference time** — The NLP explanation stage requires async calls to Qwen 2.5 with a two-level fallback chain (Gemma 3 → rule-based). This orchestration must be isolated from the CV pipeline.

The Cursor IDE supports MCP (Model Context Protocol) servers that can be invoked as tools during both development (agent-mode code generation) and runtime (pipeline execution). Three MCP servers have been evaluated for assignment to these categories.

---

## Decision

**Three MCP servers are adopted, each with an exclusive domain assignment. No cross-domain usage is permitted.**

| MCP Server | `mcp.json` Name | Domain | Transport |
|------------|-----------------|--------|-----------|
| Context7 | `context7` | Library documentation (PyTorch, librosa, pytorch-grad-cam, Gradio, ONNX Runtime) | stdio / npx |
| Hugging Face Skills | `huggingface-skills` | HF Model Hub checkpoints, Spaces deployment, FoR dataset retrieval | url / browser login |
| Stitch | `stitch` | Qwen 2.5 async API, pipeline orchestration, NLP fallback chains | url / Google API key header |

---

## Alternatives Considered

### Alternative A — Manual Documentation Lookup (rejected for context7-mcp role)

**Description:** Engineers look up PyTorch/librosa/Gradio documentation manually via browser search or inline IDE tooltips.

**Why rejected:**
- Cursor agent-mode generation cannot access live browser documentation; it falls back to training-data cutoff knowledge, which may be outdated for librosa 0.10.x and Gradio 4.x APIs.
- context7-mcp provides version-pinned, structured documentation retrieval directly into the agent context, ensuring generated code matches the pinned versions in `requirements.txt`.
- Manual lookup breaks the chain-based workflow — every sprint prompt would need to re-describe API contracts.

**SRS Impact:** Without context7-mcp, FR-AUD-006 (librosa Mel spectrogram), FR-CV-011 (pytorch-grad-cam), and FR-DEP-001 (Gradio 4.x) implementations risk API mismatch regressions.

---

### Alternative B — Direct HF Hub Python API Calls Without MCP (rejected for huggingface-mcp role)

**Description:** Use `huggingface_hub` Python client directly in pipeline scripts for checkpoint download and Spaces deployment.

**Why rejected:**
- Authenticated Hub operations (model push, Space secrets management) during development require interactive browser login flows that are not reproducible in agent-mode generation.
- `huggingface-mcp` provides a structured, authenticated interface for Hub operations that is callable from within Cursor's agent context.
- The FoR dataset (`for-2sec` variant) requires Kaggle download by agreement — `huggingface-mcp` can reference the HF mirror and dataset cards without re-implementing the download protocol.
- Runtime checkpoint loading (FR-CV-007) remains via `huggingface_hub` Python client in `src/cv/model.py` — but the MCP handles the development-time operations (upload, verify, introspect).

**SRS Impact:** FR-CV-007 (HF Hub checkpoint storage), FR-DEP-001 (HF Spaces deployment) would lack structured tooling support.

---

### Alternative C — Synchronous Qwen 2.5 Direct API Calls Without Orchestration (rejected for stitch-mcp role)

**Description:** Call Qwen 2.5 REST API directly from `src/nlp/explain.py` using `httpx` or `requests`, with no orchestration layer.

**Why rejected:**
- A direct synchronous call blocks the Gradio UI event loop, violating FR-NLP-006 (Stage 2 must display before Stage 3 completes).
- The two-level fallback chain (Qwen 2.5 → Gemma 3 → rule-based, per FR-NLP-003 and FR-NLP-007) requires conditional retry logic with independent timeout handling. Implementing this without an orchestration layer produces complex, untestable branching in `explain.py`, violating the McCabe complexity ≤ 10 rule from `.cursorrules`.
- `stitch-mcp` provides async task dispatch and fallback chain definition as first-class primitives, keeping `explain.py` below the 300-line module limit.
- API key management: `stitch-mcp` reads `QWEN_API_KEY` from the HF Spaces secrets environment, never from code, satisfying FR-NLP-005 and S.H.I.E.L.D. rules.

**SRS Impact:** Without stitch-mcp, FR-NLP-002, FR-NLP-006, and FR-NLP-007 are at high risk (see RISK-003 in phase0-risk-register.md).

---

## Rationale

### Pipeline-Stage-to-MCP Mapping

Each MCP server maps to a specific set of pipeline stages and SRS FR groups:

| Pipeline Stage | Primary SRS FRs | MCP Tool Used | Justification |
|---------------|-----------------|---------------|---------------|
| Audio DSP (`dsp.py`) | FR-AUD-001–011 | `context7` (librosa docs) | librosa 0.10.x API — needs version-accurate documentation |
| CV Inference (`infer.py`) | FR-CV-001–009 | `context7` (PyTorch, ONNX Runtime) | EfficientNet-B4 torchvision API + ONNX Runtime C++ binding |
| XAI Grad-CAM (`gradcam.py`) | FR-CV-010–016 | `context7` (pytorch-grad-cam) | jacobgil library API for `GradCAM`, `EigenCAM`, layer targeting |
| Model Training (`train.py`) | FR-CV-003–008 | `huggingface-mcp` (FoR dataset, checkpoint push) | HF Hub authenticated upload for FR-CV-007 |
| NLP Explanation (`explain.py`) | FR-NLP-001–009 | `stitch-mcp` (Qwen 2.5 async) | Async orchestration + fallback chain + API key isolation |
| Deployment (`app.py`, HF Space) | FR-DEP-001–010 | `huggingface-mcp` (Spaces deployment) | Space creation, secrets management, demo file upload |

### Exclusivity Rule

No MCP server is shared across domain boundaries. This prevents:
- Context7 from making live API calls (documentation only — read-only)
- Stitch from receiving audio data (NLP stage only — text in/text out)
- HF Skills from executing pipeline inference (Hub operations only — artifact management)

---

## Consequences

**Positive:**
- Each sprint prompt can explicitly state which MCP tool to invoke for its dependency lookups, producing deterministic agent behaviour.
- S.H.I.E.L.D. rules are structurally enforced: API keys never appear in Python source because stitch-mcp mediates all LLM calls.
- Library version accuracy is guaranteed for chain-generated code.

**Negative / Trade-offs:**
- Three separate MCP server configurations must be maintained in `mcp.json`. If any server is unavailable, the corresponding sprint is blocked (not degraded gracefully).
- `stitch-mcp` requires a Google API key header for authentication — this must be provisioned before Sprint D begins.

---

## SRS References

All Audio/CV/NLP/Deployment FRs in DSDBA-SRS-2026-002 v2.1:
- FR-AUD-001–011 (Audio DSP)
- FR-CV-001–016 (Computer Vision + XAI)
- FR-NLP-001–009 (NLP Explanation)
- FR-DEP-001–010 (Deployment)
- NFR-Security (API key isolation)
- NFR-Maintainability (config-driven, modular tooling)

---

*[DRAFT — Phase 0 — Pending V.E.R.I.F.Y.]*
