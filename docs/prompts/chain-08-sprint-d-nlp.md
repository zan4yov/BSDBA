# ════════════════════════════════════════════════════════════════════════
# DSDBA Chain 08 — Sprint D: NLP / Qwen 2.5 Integration
# SRS: FR-NLP-001–009 | Phase: 4-D | Mode: Agent
# V.E.R.I.F.Y. Level: 3 | S.H.I.E.L.D. MANDATORY
# ════════════════════════════════════════════════════════════════════════

@file .cursorrules
@file config.yaml
@file docs/context/session-cheatsheet.md
@file docs/adr/phase2-interface-contracts.md
@file src/cv/gradcam.py

⚠️ SECURITY RULE: At no point in this chain may any API key, token, or
credential appear in code, test files, or docstrings. All credential access
must use os.environ.get(cfg.nlp.api_key_env_var) ONLY. [FR-NLP-005]

Use stitch-mcp (mcp.json: "stitch", url: https://stitch.googleapis.com/mcp) to configure the Qwen 2.5 API async call and fallback chain.

IMPLEMENT: src/nlp/explain.py  [FR-NLP-001–009]

  1. build_prompt(label: str, confidence: float, band_pct: dict[str, float], cfg) -> str:
     - Construct structured prompt: label + confidence + band percentages [FR-NLP-001]
     - Instruction: produce 3–5 sentence English explanation citing highest-attribution band
     - Return formatted prompt string (no API call here)

  2. call_qwen_api(prompt: str, cfg) -> str:  [async function] [FR-NLP-002]
     - Use OpenAI-compatible API client with Qwen 2.5 endpoint via stitch-mcp
     - API key from os.environ.get(cfg.nlp.api_key_env_var) — NEVER hardcoded
     - Apply asyncio.wait_for() with cfg.nlp.timeout_sec timeout [FR-NLP-002]
     - On timeout or exception → raise NLPTimeoutError

  3. call_gemma_fallback(prompt: str, cfg) -> str:  [async function] [FR-NLP-007 SHOULD]
     - Secondary fallback LLM using stitch-mcp
     - Same timeout enforcement

  4. build_rule_based_explanation(label: str, confidence: float, band_pct: dict) -> str:
     - ALWAYS available — no API dependency [FR-NLP-003]
     - Template: "Analysis indicates [label] speech with [confidence]% confidence.
       The [highest_band] frequency band (X%) showed the highest activation, suggesting
       [rule-based interpretation]. [1-2 more sentences based on label and top band.]"
     - Must produce grammatically correct English [FR-NLP-004]

  5. generate_explanation(label: str, confidence: float, band_pct: dict, cfg) -> tuple[str, bool]:
     - Async orchestration via stitch-mcp [FR-NLP-006]
     - Try: Qwen 2.5 → fallback Gemma-3 → fallback rule-based [FR-NLP-003, FR-NLP-007]
     - Return (explanation_text, api_was_used: bool)
     - api_was_used=False triggers warning badge in UI [FR-NLP-003]

  6. get_cached_explanation(label, confidence, band_pct, cfg) -> str | None: [FR-NLP-008 SHOULD]
     - Cache key: (label, confidence_bucket, top_band_name)
     - confidence_bucket = round to nearest in cfg.nlp.caching.confidence_buckets

IMPLEMENT: src/tests/test_nlp.py  [V.E.R.I.F.Y. L3]
  All tests must work WITHOUT a real API key (use mock/patch):
  1. test_build_prompt_contains_all_fields: prompt has label, confidence, all 4 bands
  2. test_rule_based_fallback_always_returns: works with any valid input, no API needed
  3. test_rule_based_grammar: output is non-empty string ≥ 3 sentences
  4. test_qwen_timeout_triggers_fallback: mock API to timeout → rule-based used [FR-NLP-003]
  5. test_warning_flag_on_fallback: api_was_used=False when fallback triggered
  6. test_cv_result_independent_of_nlp: generate_explanation is called as asyncio.Task
     and does NOT block a mock CV result display [FR-NLP-006]
  7. test_no_api_key_in_source: grep src/nlp/explain.py for any string matching
     known API key patterns (hf_, sk-, DASH, Bearer) — must find zero matches [FR-NLP-005]
  8. test_cache_hit_skips_api: second call with same (label, bucket, band) uses cache

All 8 tests must pass before Gate Check.