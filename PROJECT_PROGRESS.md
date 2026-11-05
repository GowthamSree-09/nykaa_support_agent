# PROJECT_PROGRESS — Nykaa Domain Support Agent (CrewAI)

Source of truth: `Final_Capstone.pdf`. Everything runs under `MOCK_LLM`, zero API keys, zero network.
Env: Windows, PowerShell, Python 3.12 venv at `.venv`. Repo: https://github.com/GowthamSree-09/nykaa_support_agent
Commands: `.\.venv\Scripts\python.exe -m pytest -q` (189 pass) · `-m demos.run_demos` · `-m eval.judge_eval [--stress|--heldout]` · `-m uvicorn api.main:app`

## Status
All 16 tasks implemented and tested (see `REQUIREMENT_AUDIT.md`). Two improvement rounds are DONE locally and NOT committed/pushed (user decides when).

## Round 1 (done)
Relevance (embedding-scored sentences, category gate, same-document rule, true sources), question routing, multi-query retrieval, threshold 0.44 (calibrated with near-domain junk), stronger groundedness (numbers/negation/antonyms/absolutes, re-check after review), richer AutoGen reviewer, API error handling, governance audit log.

## Round 2 — the two stress defects (done)
1. "How do I return my earphones?" falsely rejected -> root cause was vocabulary (product nouns score ~0.33, category words ~0.56), not the threshold. Fix: product glossary replaces product noun with category word for retrieval (`rag/relevance.py`); threshold stays 0.44. Evidence in `python -m rag.calibrate`; out-of-scope questions unchanged (max 0.3682).
2. "Who do I contact if it fails?" -> dangling "it" + fragment retrieval matched "fail" + no answer-type check. Fixes: `clarify` intent (`agents/routing.py`), session topic carry-over (`agents/memory.py`), same-message resolution + topic hint + who-gate (`rag/generate.py`), reviewer answer-type check (`llm/autogen_mock.py`).
3. Found and fixed on the way: relative "that" wrongly treated as a reference (caught by the 15-query eval: query 12 fell to 1.00); junk sub-question rode along (sub-queries now need 0.44 unless hinted); order-only part of a mixed question searched the KB (found by held-out set).

## Round 3 — COD refund answer and hair-dryer check (done)
- COD: "How long will a refund take if I paid cash on delivery?" returned the generic COD sentence. Root cause: the picker ranks by similarity + shared words and had no answer-type notion for "how long". Fix: duration questions (how long / how soon / how quickly / how many days...) give +0.12 to sentences that state a duration (`rag/relevance.py` `is_duration_question`/`has_duration`, `rag/generate.py` `DURATION_BOOST`). Bonus, not filter. "when will ..." is deliberately NOT a duration question.
- Collateral check: 32 snapshot questions (`tests/snapshot_questions.py`) before/after -> exactly 2 changed, both better (COD; "How long are loyalty points valid?"). Snapshot frozen in `tests/answer_snapshot.json`; pre-fix in `tests/answer_snapshot_before_duration_fix.json`.
- Hair dryer: KB never mentions hair dryers (only Electronics 12 month, Home appliances 6 month). The old "12 month" expectation came from a comment in dataset.py, so the EXPECTATION was corrected (`"12 month|6 month"`, judge supports "|" alternatives); code unchanged.
- Still wrong (documented, not fixed): "How long does delivery take to remote areas?"; "When will ... refund reach ...?" returns the general sentence.

## Round 4 — final audit (done)
Fixed: remote-area wording, "when will ... reach", COD abbreviation, ambiguous appliances (both rules + confirm-category sentence), vague second part merged into previous part. 20 new tests (tests/test_round4_defects.py), unseen probe 12/14 -> 14/14 (eval/unseen_probe.py), 177 tests pass. Judge accepts the fixed category sentence as on-topic. Snapshot re-recorded (exactly the 2 known-bad answers changed).

## Results (actual)
- Tests: 189 passed (36 original + 57 round 1 + 25 round 2 + 39 round 3 + 20 round 4 + 12 round 5).
- Round 5: Task 12 gaps fixed (/health, DELETE /sessions, validation errors now logged once via middleware) and /add-document masks phone/card before indexing; see REQUIREMENT_AUDIT.md.
- 15 required queries: 5.00 / 5.00 / 5.00 / 5.00 (before all rounds 4.78 / 5.00 / 4.91 / 5.00).
- Stress set (12): 5.00 x4 (before round 2: 4.42 / 4.67 / 4.50 / 5.00). Tuned on, so not proof of generality.
- Held-out set (10): first 4.20 / 5.00 / 5.00 / 5.00; after COD fix 4.60; after correcting the hair-dryer expectation 5.00 x4. No longer untuned (its failures were fixed).
- Precision/recall unchanged: sentence+multi-query P 1.000 R 1.000; sentence single P 0.857 R 0.929; fixed single P 0.429 R 0.929.
- Network guard: 0 outbound connection attempts.

## Gotchas learned
- The tool layer collapses double backslashes in shell/heredoc input (regexes got backspace characters). Write regex-containing files with the Write/Edit tools, never via heredoc; check with a `\x08` scan if a regex misbehaves.
- autogen-agentchat must be >=0.7 (needs googleapis-common-protos==1.70.0 for protobuf 5).
- CrewAI ReAct: the observation arrives inside the latest ASSISTANT message followed by a user nudge.

## Known issues
- Appliance ambiguity list, glossary and phrase rewrites are hand-written.
- Glossary and topic hints are hand-written; unknown products may still be rejected / clarified.
- Names/addresses not masked (out of scope per brief). Memory is in-process only.

## Next step
Round 4 committed and pushed to main.
