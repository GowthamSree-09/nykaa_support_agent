# PROJECT_PROGRESS — Nykaa Domain Support Agent (CrewAI)

Source of truth: `Final_Capstone.pdf`. Everything runs under `MOCK_LLM`, zero API keys.
Env: Windows, PowerShell, Python 3.12 venv at `.venv`.
Commands: `.\.venv\Scripts\python.exe -m pytest -q` (36 pass) · `-m demos.run_demos` · `-m eval.judge_eval` · `-m uvicorn api.main:app`

## Status: Tasks 1-16 implemented, tested, documented (README.md has all evidence)
| # | Task | Status | Evidence |
|---|------|--------|----------|
| 1 | Dataset seed 12, 60 orders, 20.0% delayed | DONE | transcripts/task1_dataset.txt |
| 2 | KB 12 docs | DONE | tests/test_kb.py |
| 3 | 2 chunkings, 2 Chroma collections (23 / 48 chunks) | DONE | tests/test_rag.py |
| 4 | Grounded gen, threshold 0.32 from measured clusters | DONE | transcripts/task4_* |
| 5 | P/R both collections; sentence wins (P .857 vs .429) | DONE | transcripts/task5_chunking_eval.txt |
| 6 | check_order_status, score 0.6*delayed+0.4*days/30, threshold 0.83 (median 17d) | DONE | tests/test_order_tool.py |
| 7 | CrewAI crew (3 agents) via kickoff, MockCrewLLM(BaseLLM) | DONE | transcripts/full_demo.txt |
| 8 | LangChain session memory | DONE | demo + tests |
| 9 | SupportResponse Pydantic schema | DONE | |
| 10 | Guardrails (PII, injection, groundedness) | DONE | |
| 11 | FastAPI: /ask, /add-document, /health, WS /ws/chat | DONE | tests/test_api_and_eval.py |
| 12 | JSONL log, trace id, masked | DONE | logs, tests |
| 13 | 15-query eval, avgs acc 4.78 / grd 5.0 / cmp 4.91 / saf 5.0 | DONE | transcripts/task13_evaluation.txt |
| 14 | AutoGen RoundRobinGroupChat review (approve + revise) | DONE | |
| 15 | Least autonomy, Medium risk, budget cap 1000 tokens | DONE | |
| 16 | Response cache with hit evidence | DONE | |
Network guard demo: 0 outbound connection attempts.

## Key implementation notes
- autogen-agentchat must be >=0.7 (pip default gave 0.2.x). It pins protobuf 5.29 -> googleapis-common-protos==1.70.0 needed.
- AutoGen agent names must be identifiers (underscores).
- CrewAI ReAct: observation arrives inside the latest ASSISTANT message, followed by a user nudge; mock reads assistant messages only.
- Seed 42 was marginal; seed 12 used (no record edits).
- Judge was first too lenient (all 5.0); made stricter with on-topic-sentence penalty; generator got crude stemming + relative-overlap cutoff.

## Known issues
- Q7 retrieval weak (late parcel -> damaged_item_claim); extra sentences in eval Q3, Q13; grounding is 5.0 by construction.
- test_add_document_then_retrievable failed once, not reproducible in later clean runs.
- Names/addresses not masked (out of scope per brief).

## Remaining before submission (needs user)
- Create a PUBLIC GitHub repo and push (NOT done; needs user approval). Folder is not a git repo yet.
- Optionally re-run the demos on the user's machine and commit transcripts/.
