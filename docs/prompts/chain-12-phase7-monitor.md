# ════════════════════════════════════════════════════════════════════════
# DSDBA Chain 12 — Phase 7: Monitoring, Maintenance & Retrospective
# SRS: NFR-Reliability, FR-CV-008 final | Phase: 7 | Mode: Inline (Cmd+K)
# ════════════════════════════════════════════════════════════════════════

@file .cursorrules
@file docs/context/session-cheatsheet.md
@file docs/adr/phase5-accuracy-results.md
@file docs/adr/phase5-latency-benchmark.md
@file docs/adr/phase6-release-notes.md

TASK 1: Cold-Start Verification
  Access the live HF Spaces URL from a fresh browser (no cache).
  Time the cold-start from page load to first interactive state.
  Target: < 30 s [NFR-Reliability].
  If > 30 s → add warm-up ping script recommendation to README.md.
  Record result in docs/adr/phase7-monitoring.md.

TASK 2: Q7 Final Resolution
  If Q7 was resolved in Chain 10 using scikit-learn EER:
  Confirm the final EER matches the result logged in phase5-accuracy-results.md.
  Update session-cheatsheet.md: Q7 → ✅ FULLY RESOLVED — [final EER value, scoring method].

TASK 3: Post-Deployment Retrospective
  Create docs/adr/phase7-retrospective.md with sections:
  1. Accuracy Achievement: actual EER vs. ≤ 10% target [FR-CV-008]
  2. Latency Achievement: actual ms vs. all NFR-Performance targets
  3. XAI Quality: Grad-CAM saliency faithfulness (deletion AUC if computed)
  4. Security: any vulnerabilities found and resolved
  5. Open Questions: all Q1–Q7 resolved? Any outstanding?
  6. Technical Debt: list all [TECH DEBT] items flagged across all sprints
  7. Lessons Learned: top 3 engineering decisions that had the most impact
  8. Future Improvements: deferred SHOULD/MAY requirements for v2.2

TASK 4: Final Documentation Freeze
  Update README.md with final:
  - Accuracy results (EER, AUC-ROC, F1, accuracy)
  - Live demo URL
  - All citations and credits
  Tag final commit: v2.1.0-final
  Close all remaining checklist items in docs/context/session-cheatsheet.md.
  Final status: ALL 6 Gate Checks passed across all chains ✅