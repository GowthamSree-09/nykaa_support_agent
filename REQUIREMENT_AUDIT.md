# Requirement audit — Final_Capstone.pdf (16 tasks)

Status values: PASS = implemented, tested, and demonstrated with a transcript. PARTIAL = works but with a stated gap.
Run everything with `.\.venv\Scripts\python.exe` from the project folder. Tests: `-m pytest -q` (189 pass).

| # | Requirement (short) | Implementation | Test | Demo command → evidence | Status |
|---|---|---|---|---|---|
| 1 | Seeded dataset ≥40 orders; all categories ≥3, all statuses ≥1, delayed 10–30%, fields, printed counts, design stated | `dataset.py` (`generate_orders`, `validate`) | `tests/test_dataset.py` | `python dataset.py` → `transcripts/task1_dataset.txt` (60 orders, 20.0% delayed) | PASS |
| 2 | ≥12 KB docs, 2–5 sentences, every required topic | `kb/documents.py` | `tests/test_kb.py` | — (12 docs, one per topic) | PASS |
| 3 | Fixed+overlap and sentence chunking, local SentenceTransformers, two separate Chroma collections via `upsert()` | `rag/chunking.py`, `rag/store.py` | `tests/test_rag.py` | `-m rag.demo_task4` (retrieval from both; 23 / 48 chunks) | PASS |
| 4 | Grounded generation; threshold calibrated on ≥3 in-scope + ≥2 out-of-scope top-1 values; ≥5 in-scope demos + fallback | `rag/generate.py`, `rag/calibrate.py` | `test_threshold_between_measured_clusters`, `test_unsupported_questions_rejected` | `-m rag.calibrate`, `-m rag.demo_task4` → `transcripts/task4_*` (7 in-scope, 4+4 out-of-scope; threshold 0.44, floor 0.32) | PASS |
| 5 | Doc-level precision/recall for both collections, per-query arithmetic, 2–3 sentence recommendation | `rag/evaluate.py` | `test_multi_topic_retrieval_recall_improves`, `test_metrics_both_collections` | `-m rag.evaluate` → `transcripts/task5_chunking_eval.txt`; README table | PASS |
| 6 | `check_order_status` with designed escalation score, formula + threshold justified from dataset | `agents/order_tool.py` | `tests/test_order_tool.py`, `test_order_statuses_match_dataset` | `-m demos.run_demos` (Task 6 section) | PASS |
| 7 | CrewAI crew ≥3 agents (Retrieval+RAG tool, Lookup+tool, Composer), `.kickoff()`, both tools shown | `agents/crew.py`, `agents/tools.py`, `llm/crew_mock.py` | `test_policy_question_uses_only_rag_tool`, `test_order_question_uses_only_lookup_tool…`, `test_mixed_question_uses_both_tools` | `-m demos.run_demos` (Task 7: policy / order / mixed) | PASS |
| 8 | LangChain session memory; multi-turn transcript + separate fresh session | `agents/memory.py` | `test_memory_isolation_and_reset`, `test_memory_stores_masked_text_only` | `-m demos.run_demos` (Task 8: conversation A vs B) | PASS |
| 9 | Pydantic schema for every crew response, validated in code | `agents/schemas.py`, `run_crew`, `core_pipeline` | `test_schema_rejects_bad_response` | `-m demos.run_demos` (Task 9) | PASS |
| 10 | PII masking (phone, card last-4), injection detection, output groundedness refusal — each firing | `agents/guardrails.py`, `agents/service.py` | `test_pii_masking…`, `test_injection_attempts_blocked`, `test_groundedness_contradictions` (10 cases) | `-m demos.run_demos` (Task 10); documents added through `/add-document` are masked too (`tests/test_request_logging_and_doc_pii.py`) | PASS (guard is text matching; limits documented) |
| 11 | ≥2 HTTP endpoints + WebSocket surviving disconnect, Pydantic models | `api/main.py` | `tests/test_api_and_eval.py`, `test_api_validation_errors`, `test_ws_bad_message…` | `-m demos.run_demos` (Task 11); `uvicorn api.main:app` | PASS |
| 12 | JSON-Lines log, trace id, timing, no raw PII on disk | `api/logging_utils.py` | `test_jsonl_log_has_trace_ids_and_no_raw_pii`, `test_api_internal_error_is_generic…`, and one-entry-per-request tests for every operation in `tests/test_request_logging_and_doc_pii.py` | `-m demos.run_demos` (log lines + "raw phone on disk? False") | PASS (round 5: `/health`, `DELETE /sessions`, validation errors now logged) |
| 13 | 15 queries (every topic + ≥2 OOS/edge), 4 scores per query + 4 averages, mock judge | `eval/judge_eval.py` | `test_evaluation_15_queries_four_scores` | `-m eval.judge_eval` → `transcripts/task13_evaluation.txt`, `eval/results.json`; before: `eval/results_before_improvements.json` | PASS (15 required: 5.00 x4; stress set 5.00 x4 after the round-2 fixes, before 4.42/4.67/4.50/5.00; held-out set first 4.20/5.00/5.00/5.00, now 5.00 x4 after its failures were fixed, so it is no longer untuned) |
| 14 | AutoGen `RoundRobinGroupChat`, `max_turns=2`, `custom_message_types`, `output_content_type`, approve + revise demos | `review/autogen_review.py`, `llm/autogen_mock.py` | `test_review_approves…`, `test_review_removes_irrelevant…`, `…ungrounded_and_unsafe…`, `…all_removed_falls_back` | `-m demos.run_demos` (Task 14: 3 cases) | PASS (relevance limited to category/intent) |
| 15 | Least autonomy (only Lookup agent), Medium risk + justification, per-request budget cap rejecting oversized request | `governance/access.py`, `budget.py`, `audit.py` | `test_least_autonomy…`, `test_budget_rejection_is_audited…`, `test_access_denial_is_audited` | `-m demos.run_demos` (Task 15) | PASS |
| 16 | In-memory cache keyed by normalized query, hit avoids LLM call, before/after evidence | `governance/cache.py` | `test_cache_hit_avoids_llm_call`, `test_cache_holds_only_policy_answers…`, `test_cache_cleared_after_document_added` | `-m demos.run_demos` (Task 16: LLM calls 1 → 1, 32.55 ms → 0.01 ms) | PASS |

## Cross-cutting requirements
| Requirement | Evidence | Status |
|---|---|---|
| README states Nykaa track + seed/weights/amount range | top of `README.md` | PASS |
| MOCK_LLM, zero keys, zero network | `llm/*mock*.py`; network guard line "blocked: 0" in `transcripts/full_demo.txt` | PASS (the one-time embedding-model download is setup, documented) |
| `BaseLLM` subclass; no `Observation:` template pitfall; args from `args_schema` | `llm/crew_mock.py`; `test_mock_ignores_template_observation_text` | PASS |
| Telemetry disabled | `llm/__init__.py`, README | PASS |
| Code/text only (no images/PDF/video) | repo contents | PASS |

## Not verified / honest gaps
* A grader's own machine has not been tested; a fresh-clone run (needs the one-time model download) is untested here.
* The 10-question held-out evaluation (first 4.20, now 5.00) is no longer an untuned check. The remote-area, COD-timing and hair-dryer gaps listed in earlier rounds are fixed in round 4 (see below).
* The product glossary and the contact/escalation topic hints are hand-written lists; unknown products can still be rejected or sent to clarification.

## Round 4 (final audit) — executed evidence
* Full suite at the end of round 4: 177 passed (the suite is now 189 tests, see Round 5 below). Dataset, calibration, task-4 demo, task-5 P/R, demos, all three judge sets and the unseen probe were re-run on the final code; transcripts refreshed.
* 14 unseen probe questions (written before the fixes): 12/14 before, 14/14 after (`transcripts/unseen_probe_initial.txt`, `transcripts/unseen_probe_after_fixes.txt`).
* Closed gaps (earlier listed as not fixed): remote-area delivery, "When will … refund reach …?", hair-dryer/appliance ambiguity, unrelated sentence in multi-topic answers. Tests: `tests/test_round4_defects.py`.
* Network: `demos.run_demos` network guard reports 0 blocked outbound connection attempts; telemetry env vars set in `llm/__init__.py`.
* NOT VERIFIED: a clean-machine fresh clone (dependency install + one-time model download) was not run; only the existing venv was used. `requirements.txt` pins exact versions.

## Round 5 (final PDF verification) — executed evidence
* Checked every operation against Task 12 ("log every request"). **Two genuine gaps were reproduced and fixed** (the other four operations already complied):
  * `GET /health`, `DELETE /sessions/{id}` and Pydantic validation errors (e.g. an empty `/ask` question) wrote **no** log entry. A middleware now logs whatever an endpoint has not already logged (method, status, trace id, duration, empty request text); a request-scoped marker prevents duplicates. `/ask`, `/add-document` (200 and 422), the 413 and 500 paths and the WebSocket messages / disconnects keep their single, richer entries.
  * Task 10 (input-side PII masking): `POST /add-document` stored a phone number and a card last-4 verbatim in both ChromaDB collections, from where they could be returned in answers. The text is now masked with the same `mask_pii` before indexing, and the response reports `pii_masked`.
* Files changed: `api/main.py`, `api/logging_utils.py`, `agents/service.py`; new `tests/test_request_logging_and_doc_pii.py` (12 tests; 6 of them fail on the pre-fix code, the other 6 pin behavior that must not change).
* Full suite: 189 passed, 0 failed (`transcripts/pytest_results.txt`). Dataset, calibration, task-4 demo, task-5 P/R, `demos.run_demos` (network guard: 0 blocked connections), the three judge sets (all 5.00 on the four metrics) and the 14 probe questions (14/14) were re-run; all 16 tasks still meet their acceptance criteria. Only `full_demo.txt` and `pytest_results.txt` differ from the previous transcripts (new `pii_masked` field, a `/health` log line, fresh timings).
* Remaining limitations (not fixed, documented): `/add-document` does not run the injection patterns over document text and has no authentication; free-text names and addresses are still not masked.
