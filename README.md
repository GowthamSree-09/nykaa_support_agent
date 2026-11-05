# Nykaa Domain Support Agent (CrewAI) — Final Capstone

> **Track completed: Nykaa (E-commerce & Retail).**
> Everything runs under `MOCK_LLM` with **zero API keys and zero network access** during graded runs
> (the embedding model is downloaded once during setup, then loaded from the local cache).
> All data (orders, policies, names) is fabricated; the policies are fictional and are **not** real Nykaa policy.

| At a glance | |
|---|---|
| Automated tests | **177 passed, 0 failed** ([transcripts/pytest_results.txt](transcripts/pytest_results.txt)) |
| Capstone tasks | **16 / 16 PASS** ([REQUIREMENT_AUDIT.md](REQUIREMENT_AUDIT.md)) |
| Required 15-query evaluation (mock judge) | Accuracy 5.00 · Grounding 5.00 · Completeness 5.00 · Safety 5.00 |
| Outbound network calls during the graded demo run | 0 (network guard, [transcripts/full_demo.txt](transcripts/full_demo.txt)) |

## Table of contents
1. [Dataset design choices (Task 1)](#dataset-design-choices-part-1-task-1--reproduce-with-python-datasetpy)
2. [Introduction](#introduction)
3. [How the pieces fit together](#how-the-pieces-fit-together)
4. [Project structure](#project-structure)
5. [Quick start (Windows PowerShell)](#quick-start-windows-powershell)
6. [Sample customer interactions](#sample-customer-interactions)
7. [Verification summary (16 tasks)](#verification-summary-16-tasks)
8. [Key technical results](#key-technical-results)
9. [Technical details by part](#technical-details-by-part) — [Part 1](#part-1--rag-core) · [Part 2](#part-2--crewai-memory-schema-guardrails) · [Part 3](#part-3--api-logging-evaluation) · [Part 4](#part-4--governance-and-resilience)
10. [Tests](#tests)
11. [Limitations and known issues](#limitations-and-known-issues)
12. [Improvement history and extra evidence](#improvement-history-and-extra-evidence)

## Dataset design choices (Part 1, Task 1) — reproduce with `python dataset.py`

| Choice | Value |
|---|---|
| Seed | `12` (`random.Random(12)`). The first try, seed 42, gave exactly 10.0% delayed and only 3 Electronics orders, so I changed the **seed** (never individual records) |
| Orders | 60 (`ORD-1001` … `ORD-1060`) |
| Category weights | Apparel 0.25, Electronics 0.15, Home 0.15, Footwear 0.20, Beauty 0.25 |
| Status weights | Placed 0.15, Shipped 0.20, Delivered 0.40, Returned 0.15, Refunded 0.10 |
| Delayed probability | 0.20 per order (the real percentage is measured, not forced) |
| `days_since_created` | integer, uniform 0–30 |
| Amount range (INR) | Beauty 199–3,999 · Home 299–9,999 · Apparel 499–4,999 · Footwear 799–6,999 · Electronics 1,999–49,999 |
| Reasoning (one sentence) | Beauty/Home are low-ticket (a lipstick, a cushion cover), Apparel/Footwear mid-ticket, and Electronics (earbuds, trimmers) the highest-ticket category |

Result (`transcripts/task1_dataset.txt`): Home 7, Beauty 16, Footwear 14, Apparel 11, Electronics 12 · Delivered 27, Shipped 12, Returned 6, Refunded 6, Placed 9 · **20.0% delayed** · validation PASS.

## Introduction

**Problem.** A retail support team answers the same questions all day ("What is the return window for beauty products?", "Where is my order?") and escalates too much to humans. This project is a support agent that answers **policy questions** from a knowledge base it was given, looks up **a specific order's status** from an order dataset, remembers the conversation, refuses what it cannot support, and has its own answers double-checked before they reach the customer.

**What the agent can do**
* Answer questions about the 12 required policy topics (return windows, COD refunds, delivery SLAs, reverse pickup, warranty, cancellation, loyalty points, payment retry, size exchange, damaged items, international shipping, escalation matrix) using **only** retrieved knowledge-base sentences.
* Look up an order (`ORD-1001` … `ORD-1060`) and report its status, value and a designed **escalation score**.
* Handle mixed questions ("Can I cancel ORD-1023 and when would I get a refund?"), follow-ups in the same session ("Can I cancel it?"), and ask for the order id when it is missing.
* Say "I don't know" and hand over to a human when the knowledge base does not support an answer; block prompt-injection attempts; mask phone numbers and card last-4 digits before anything is processed or logged.

## How the pieces fit together

| Piece | Role in this project |
|---|---|
| **ChromaDB + SentenceTransformers** | Knowledge-base sentences are embedded locally (`all-MiniLM-L6-v2`) and indexed into two collections (fixed-size chunks and sentence chunks) with `collection.upsert()`. |
| **RAG (grounded generation)** | Retrieves top-k chunks; if the best cosine similarity is below the calibrated threshold (0.44) it answers "I don't know", otherwise it builds the answer only from retrieved sentences. |
| **CrewAI crew** | Three agents run with `.kickoff()`: a Policy Retrieval Agent (tool `rag_lookup`), an Order Lookup Agent (tool `check_order_status`, the only agent allowed to use it) and a Response Composer that merges both into one validated `SupportResponse`. |
| **MOCK_LLM** | Deterministic stand-ins for the language model, one per framework: a `BaseLLM` subclass for CrewAI, a rule-based grounded generator, and a mock model client for AutoGen. The same input always gives the same output; no key and no network are needed. |
| **AutoGen review** | A two-agent `RoundRobinGroupChat` (Policy-Compliance Reviewer → Final Editor, `max_turns=2`) approves the draft unchanged or revises it, returning a structured `Verdict` (`approved`, `final_answer`, `reason`). |
| **FastAPI** | `POST /ask`, `POST /add-document`, `DELETE /sessions/{id}`, `GET /health` and the WebSocket `/ws/chat`, all behind the same pipeline, with one masked JSON-Lines log entry per request. |

```
question ─► input guardrails (PII mask, injection check) ─► session memory (LangChain) ─► budget check
   ─► CrewAI crew .kickoff():  Policy Retrieval Agent [rag_lookup → ChromaDB + cache]   (skipped for order-only questions)
                               Order Lookup Agent     [check_order_status → dataset]    (only when an order id is present)
                               Response Composer      [merges both, guided by the question's intent → SupportResponse]
   ─► Pydantic validation ─► output groundedness guard ─► AutoGen review (Reviewer → Final Editor, Verdict)
   ─► groundedness re-check of the revised answer ─► answer ─► JSON-Lines log (masked)
```

## Project structure

```
nykaa_support_agent/
├── dataset.py              Task 1: seeded order generator (ORDERS) + validation, prints counts and delayed %
├── kb/documents.py         Task 2: the 12 knowledge-base documents
├── rag/                    Tasks 3-5: chunking, ChromaDB store, relevance helpers, grounded generation,
│                           threshold calibration, precision/recall evaluation, Task 4 demo
├── agents/                 Tasks 6-10: order tool + escalation score, CrewAI crew and tools, routing,
│                           LangChain session memory, Pydantic schema, guardrails, shared pipeline (service.py)
├── llm/                    The three MOCK_LLM implementations; __init__.py turns CrewAI telemetry off
├── review/autogen_review.py  Task 14: AutoGen RoundRobinGroupChat review stage
├── governance/             Tasks 15-16: least-autonomy access control, token/cost budget, audit log, cache
├── api/                    Tasks 11-12: FastAPI app and JSON-Lines request logging
├── eval/                   Task 13: mock judge, 15-query set, stress / held-out / unseen-probe sets, result JSON
├── demos/run_demos.py      Runs the demonstrations for Tasks 6-12 and 14-16 behind a network guard
├── tests/                  177 pytest tests
├── transcripts/            Saved output of every demo, evaluation and the final pytest run
├── docs/IMPROVEMENT_HISTORY.md  Detailed before/after history of the improvement rounds
├── REQUIREMENT_AUDIT.md    16-task evidence table; PROJECT_PROGRESS.md: working notes
└── requirements.txt        Pinned dependencies
```

## Quick start (Windows PowerShell)

Python 3.12 is required. Run from the project folder.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# One-time only: the first run downloads the embedding model (all-MiniLM-L6-v2, ~90 MB) from Hugging Face.
# Once cached, the code loads it with local_files_only=True and never touches the network.
.\.venv\Scripts\python.exe -m pytest -q                       # 177 tests
.\.venv\Scripts\python.exe dataset.py                         # Task 1
.\.venv\Scripts\python.exe -m rag.calibrate                   # Task 4 threshold measurements
.\.venv\Scripts\python.exe -m rag.demo_task4                  # Task 4 grounded answers + fallback
.\.venv\Scripts\python.exe -m rag.evaluate                    # Task 5 precision/recall (single + multi-query)
.\.venv\Scripts\python.exe -m demos.run_demos                 # Tasks 6-12, 14-16 behind a network guard
.\.venv\Scripts\python.exe -m eval.judge_eval                 # Task 13 (15 required queries)
.\.venv\Scripts\python.exe -m eval.judge_eval --stress        # 12 harder queries (reported separately)
.\.venv\Scripts\python.exe -m eval.judge_eval --heldout       # 10 further queries (no longer untuned, see below)
.\.venv\Scripts\python.exe -m eval.unseen_probe               # 14 round-4 probe questions
.\.venv\Scripts\python.exe -m uvicorn api.main:app            # API: http://127.0.0.1:8000/docs
```

With the server running, in a second PowerShell window:

```powershell
$body = @{ question = "What is the return policy for beauty products?"; session_id = "demo" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask -ContentType "application/json" -Body $body
```

The WebSocket endpoint is `ws://127.0.0.1:8000/ws/chat?session_id=<id>`; it is exercised (including a client disconnect) by `demos.run_demos` and `tests/test_api_and_eval.py`.

### MOCK_LLM and telemetry confirmation
* **`CREWAI_DISABLE_TELEMETRY=true`, `OTEL_SDK_DISABLED=true`, `CREWAI_TRACING_ENABLED=false`, `ANONYMIZED_TELEMETRY=False` are set** in [llm/__init__.py](llm/__init__.py), which every entry point imports *before* `crewai`/`chromadb`.
* `demos/run_demos.py` installs a **network guard** (any non-loopback socket connect or DNS lookup raises and is counted). The final transcript line reports `outbound connection attempts blocked: 0` ([transcripts/full_demo.txt](transcripts/full_demo.txt)).
* **MOCK_LLM vs a real LLM:** a real LLM *writes* new text; the mock only applies fixed rules (embedding scores, sentence selection, JSON merging, rule-based scoring) to the text it is given, so the same input always gives the same output and nothing leaves the machine. Three mocks, one per framework: [llm/mock_llm.py](llm/mock_llm.py) (grounded generation), [llm/crew_mock.py](llm/crew_mock.py) (`MockCrewLLM`, a `crewai.llms.base_llm.BaseLLM` subclass) and [llm/autogen_mock.py](llm/autogen_mock.py) (a deterministic AutoGen `ChatCompletionClient`). The Task 13 judge is `MockJudge` in [eval/judge_eval.py](eval/judge_eval.py). No real LLM is wired in.
* The two CrewAI pitfalls from the brief: (1) the mock reads only **assistant** messages for `Observation:` — never the system prompt whose template contains `Observation: the result of the action` (test `test_mock_ignores_template_observation_text`); (2) tool arguments are chosen from the tool's declared `args_schema` fields (`record_id` / `query`), never from its name.

## Sample customer interactions

Real outputs from `POST /ask` on a local server run (MOCK_LLM, fresh session `readme-demo`, 5 November 2026). The same answers appear in [transcripts/full_demo.txt](transcripts/full_demo.txt) (Tasks 7-8).

**1. Return policy question**

| | |
|---|---|
| Customer | What is the return policy for beauty products? |
| Agent | Beauty products can be returned within 7 days of delivery, and only if the seal is unbroken. |
| Metadata | `grounded=true`, `sources=["return_window"]` |

**2. Order-status question**

| | |
|---|---|
| Customer | Where is my order ORD-1023? |
| Agent | Order ORD-1023 is currently Shipped (order value 1188 INR). It has a high escalation score (0.9333), so we are escalating it to a Level 2 team lead. |
| Metadata | `order_id=ORD-1023`, `check_order_status` was the only tool used |

**3. Mixed order-and-policy question**

| | |
|---|---|
| Customer | Can I cancel ORD-1023 and when would I get a refund? |
| Agent | Order ORD-1023 is currently Shipped (order value 1188 INR). It has a high escalation score (0.9333), so we are escalating it to a Level 2 team lead. Once the order is Shipped, it cannot be cancelled, but the customer can refuse delivery or start a return after it arrives. A bank refund takes 5 to 7 working days after the returned item passes quality checks. |
| Metadata | `sources=["cod_refund","order_cancellation"]`; both `rag_lookup` and `check_order_status` were invoked |

**4. Unsupported question (safe refusal)**

| | |
|---|---|
| Customer | How do I order a pizza online? |
| Agent | I don't know based on our support knowledge base. Let me connect you to a human agent. |
| Metadata | `grounded=false`, no sources. This near-domain question scored 0.3682 top-1 similarity, below the 0.44 threshold ([calibration](transcripts/task4_threshold_calibration.txt)). |

**5. PII masking in action.** A customer message "My phone is 9876543210. What is the status of order ORD-1023?" is masked to `[PHONE_MASKED]` before the crew sees it and before it is written to `logs/requests.jsonl`; the agent answers with the same order-status text as example 2. The phone and card masking demonstrations are in [transcripts/full_demo.txt](transcripts/full_demo.txt) (Task 10).

## Verification summary (16 tasks)

All 16 tasks are marked PASS in [REQUIREMENT_AUDIT.md](REQUIREMENT_AUDIT.md), which lists for each task the implementing file, the test and the demo command. The final pytest run is saved in [transcripts/pytest_results.txt](transcripts/pytest_results.txt) (**177 passed**).

| # | Task | Implementation | Evidence |
|---|---|---|---|
| 1 | Seeded dataset, ≥40 orders, delayed 10-30% | [dataset.py](dataset.py) | [task1_dataset.txt](transcripts/task1_dataset.txt) — 60 orders, 20.0% delayed |
| 2 | ≥12 knowledge-base documents | [kb/documents.py](kb/documents.py) | [tests/test_kb.py](tests/test_kb.py) |
| 3 | Fixed + sentence chunking, two Chroma collections via `upsert()` | [rag/chunking.py](rag/chunking.py), [rag/store.py](rag/store.py) | [tests/test_rag.py](tests/test_rag.py), 23 / 48 chunks |
| 4 | Grounded generation, calibrated threshold | [rag/generate.py](rag/generate.py), [rag/calibrate.py](rag/calibrate.py) | [calibration](transcripts/task4_threshold_calibration.txt), [grounded answers](transcripts/task4_grounded_generation.txt) |
| 5 | Precision / recall for both collections | [rag/evaluate.py](rag/evaluate.py) | [task5_chunking_eval.txt](transcripts/task5_chunking_eval.txt) |
| 6 | `check_order_status` + designed escalation score | [agents/order_tool.py](agents/order_tool.py) | [tests/test_order_tool.py](tests/test_order_tool.py), [full_demo.txt](transcripts/full_demo.txt) (Task 6) |
| 7 | CrewAI crew, `.kickoff()`, both tools shown | [agents/crew.py](agents/crew.py), [llm/crew_mock.py](llm/crew_mock.py) | [full_demo.txt](transcripts/full_demo.txt) (Task 7) |
| 8 | LangChain session memory + fresh session | [agents/memory.py](agents/memory.py) | [full_demo.txt](transcripts/full_demo.txt) (Task 8) |
| 9 | Pydantic structured output | [agents/schemas.py](agents/schemas.py) | [full_demo.txt](transcripts/full_demo.txt) (Task 9) |
| 10 | PII masking, injection detection, groundedness guard | [agents/guardrails.py](agents/guardrails.py) | [full_demo.txt](transcripts/full_demo.txt) (Task 10) |
| 11 | FastAPI: HTTP + WebSocket with disconnect handling | [api/main.py](api/main.py) | [tests/test_api_and_eval.py](tests/test_api_and_eval.py), [full_demo.txt](transcripts/full_demo.txt) (Task 11) |
| 12 | JSON-Lines logging, trace id, no raw PII | [api/logging_utils.py](api/logging_utils.py) | [full_demo.txt](transcripts/full_demo.txt) (log lines) |
| 13 | 15-query four-score evaluation | [eval/judge_eval.py](eval/judge_eval.py) | [task13_evaluation.txt](transcripts/task13_evaluation.txt), [eval/results.json](eval/results.json) |
| 14 | AutoGen review: approve + revise | [review/autogen_review.py](review/autogen_review.py) | [AutoGen review demo](transcripts/full_demo.txt) (Task 14: three cases), [tests/test_review_governance.py](tests/test_review_governance.py) |
| 15 | Least autonomy, risk class, budget cap | [governance/](governance/) | [governance demo](transcripts/full_demo.txt) (Task 15), [tests/test_review_governance.py](tests/test_review_governance.py) |
| 16 | Normalized-query response cache | [governance/cache.py](governance/cache.py) | [cache demo](transcripts/full_demo.txt) (Task 16: LLM calls 1 → 1) |

## Key technical results

| Result | Value | Where |
|---|---|---|
| Dataset | 60 orders, seed 12, 20.0% delayed, all 5 categories ≥7 orders | [Task 1](#dataset-design-choices-part-1-task-1--reproduce-with-python-datasetpy) |
| Similarity threshold | **0.44** (weakest in-scope top-1 0.5079, strongest out-of-scope 0.3682), with a 0.32 floor for contact/escalation topics | [Part 1](#part-1--rag-core) |
| Chunking comparison (7 queries) | Sentence + multi-query: P 1.000 / R 1.000. Sentence single-query: P 0.857 / R 0.929. Fixed single-query: P 0.429 / R 0.929 | [Part 1](#part-1--rag-core), [transcript](transcripts/task5_chunking_eval.txt) |
| Escalation score | `0.6 × delayed + 0.4 × days/30`, escalate above 0.83 for open orders | [Part 2](#part-2--crewai-memory-schema-guardrails) |
| Evaluation, 15 required queries | 5.00 / 5.00 / 5.00 / 5.00 (Accuracy / Grounding / Completeness / Safety) | [Part 3](#part-3--api-logging-evaluation) |
| Risk classification | **Medium** (customer-support system) | [Part 4](#part-4--governance-and-resilience) |
| Budget cap | 1,000 tokens / $0.0005 per request; an 1,800-token request is rejected | [Part 4](#part-4--governance-and-resilience) |
| Cache | Repeated query: LLM calls stay at 1; 50.31 ms → 0.01 ms | [cache demo](transcripts/full_demo.txt) |

**How to read the 5.00 scores.** The judge is a deterministic, rule-based mock, and the 15 required queries are mostly single-topic. The stress and held-out sets were used to find and fix defects, so their 5.00 averages show that those defects are fixed, **not** independent validation. The 14 round-4 probe questions were written before the round-4 fixes: 12/14 passed on the unfixed code, 14/14 after (`transcripts/unseen_probe_initial.txt`, `transcripts/unseen_probe_after_fixes.txt`).

## Technical details by part

Parts 1-4 below keep the design decisions, measured values and results the capstone asks to be stated in the README.

## Part 1 — RAG core

**Knowledge base:** 12 documents in [kb/documents.py](kb/documents.py), 2–5 sentences each, one per required topic. Checked by `tests/test_kb.py`.

**Chunking + ChromaDB:** model `all-MiniLM-L6-v2` (SentenceTransformers, local). Fixed-size = 250 characters, 50 overlap → **23 chunks** in `kb_fixed`. Sentence-based = one sentence per chunk → **48 chunks** in `kb_sentence`. Both indexed with `collection.upsert()`, cosine space, metadata `doc_id` = parent document.

**Threshold calibration (Task 4)** — measured top-1 cosine similarity (`transcripts/task4_threshold_calibration.txt`):

| Query type | Sentence collection | Fixed collection |
|---|---|---|
| In-scope (7) | 0.6537, 0.7106, 0.7699, 0.5237, 0.5933, 0.5079, 0.5996 | 0.6105, 0.7149, 0.6601, 0.6082, 0.5774, 0.5031, 0.5015 |
| Out-of-scope, clearly unrelated (4) | 0.1322, 0.0525, 0.1329, 0.1203 | 0.1192, 0.0895, 0.1154, 0.0692 |
| Out-of-scope, near-domain (4: car return, weather, movie, pizza order) | 0.2981, 0.2005, 0.0904, **0.3682** | 0.2854, 0.1626, 0.1048, 0.3084 |

* First calibration (clear junk only): 0.5079 vs 0.1329 → midpoint 0.32.
* After adding near-domain junk the highest out-of-scope value is 0.3682, so 0.32 was too low. **Question-level threshold = 0.44** (midpoint of 0.5079 and 0.3682 = 0.4381). The 0.32 value survives only as a **floor** for (a) a contact/escalation question that carries a topic hint and (b) keeping candidate chunks as answer material. A sub-question of a multi-topic question without a topic hint needs the full 0.44, so a junk part ("…and how do I order a pizza online?") cannot ride along with a valid part.
* **The cost of the stricter threshold, and how it was reduced without lowering it:** short legitimate questions that name a *product* ("How do I return my earphones?" 0.33) scored below the junk, so no single cosine threshold separated everything. The threshold stays at 0.44; the product-glossary fix removes the cause (evidence table: [docs/IMPROVEMENT_HISTORY.md](docs/IMPROVEMENT_HISTORY.md)).

**Precision/recall (Task 5)** — top chunks mapped to parent documents and de-duplicated (`transcripts/task5_chunking_eval.txt` has every per-query line with arithmetic):

| # | Query (short) | Relevant docs | Fixed: P / R | Sentence: P / R |
|---|---|---|---|---|
| 1 | return a lipstick | return_window | 1/3=0.333 / 1/1=1.0 | 1/1=1.0 / 1/1=1.0 |
| 2 | COD refund to bank | cod_refund | 1/3=0.333 / 1.0 | 1/2=0.5 / 1.0 |
| 3 | cancel after shipped | order_cancellation | 1/2=0.5 / 1.0 | 1/1=1.0 / 1.0 |
| 4 | parcel arrived broken | damaged_item_claim | 1/2=0.5 / 1.0 | 1/1=1.0 / 1.0 |
| 5 | exchange shoes size | size_exchange | 1/2=0.5 / 1.0 | 1/1=1.0 / 1.0 |
| 6 | earphones warranty | warranty_terms | 1/3=0.333 / 1.0 | 1/1=1.0 / 1.0 |
| 7 | parcel late + who to escalate | delivery_sla, escalation_matrix | 1/2=0.5 / 1/2=0.5 | 1/2=0.5 / 1/2=0.5 |
| | **Average (single-query, top-3)** | | **P 3.000/7=0.429, R 6.5/7=0.929** | **P 6.000/7=0.857, R 6.5/7=0.929** |
| | **Average (multi-query, 2 chunks per sub-question)** | | **P 5.5/7=0.786, R 7/7=1.000** | **P 7/7=1.000, R 7/7=1.000** |

Query 7 with multi-query: both collections retrieve `delivery_sla` **and** `escalation_matrix` (2/2 = 1.0, recall 2/2 = 1.0). On two extra multi-topic queries (not part of the 7), multi-query *raised* recall for the sentence collection from 0.50 to 0.75 but *lowered* precision from 1.00 to 0.50 (for "…who do I contact if it fails" it retrieves unrelated documents). In the full pipeline those extra documents are filtered out by the 0.32 floor and the anchor-document rule, but the retrieval-level cost is real.

**Recommendation:** deploy the **sentence-based** collection with **multi-query** retrieval. On the 7 required queries it gives precision 1.000 and recall 1.000, against 0.857 / 0.929 for sentence single-query and 0.429 / 0.929 for fixed-size single-query (fixed chunks straddle two policies and drag in unrelated documents). Caveat: only 7+2 queries were measured, so these numbers are small-sample.

## Part 2 — CrewAI, memory, schema, guardrails

**Task 6 — escalation score.** `escalation_score = 0.6 × delayed_shipment + 0.4 × (days_since_created / 30)` (weights sum to 1 → always in [0, 1]; a non-delayed order maxes out at 0.4, so age alone never escalates). **Threshold 0.83** = score of a delayed order at the dataset's **median age (17 days)**: `0.6 + 0.4 × 17/30 = 0.827`; also requires an open order (Placed/Shipped). In the 60 orders this recommends escalating ORD-1023, ORD-1025, ORD-1037; ORD-1009 is delayed (0.9733) but already Delivered, so it is not escalated.

**Task 7 — crew** ([agents/crew.py](agents/crew.py)), run with `.kickoff()`: *Policy Retrieval Agent* (`rag_lookup`), *Order Lookup Agent* (`check_order_status`), *Response Composer* (no tools). Tools genuinely run only when needed (`TOOL_LOG` records each call; tests assert the exact list):

| Question | Tools invoked |
|---|---|
| What is the return policy for beauty products? | `rag_lookup` |
| Where is my order ORD-1023? | `check_order_status` `{'record_id': 'ORD-1023'}` |
| Can I cancel ORD-1023 and when would I get a refund? | `rag_lookup` + `check_order_status` |
| Where is my order? (no id) | none — the system asks for the id |

**Task 8 — memory** ([agents/memory.py](agents/memory.py)): `InMemoryChatMessageHistory` + `RunnableWithMessageHistory` keyed by `session_id`. "Where is my order ORD-1023?" then "Can I cancel it?" → resolved to `Can I cancel it? (order ORD-1023)`. A new session gets `order_id: None`. `reset_session` / `DELETE /sessions/{id}` clears one conversation. History stores the **masked** text only. LangChain prints deprecation warnings for these classes; this is expected.

**Task 9 — schema:** `SupportResponse` ([agents/schemas.py](agents/schemas.py)) is the Composer task's `output_pydantic`, re-validated in `run_crew` and again in `service.core_pipeline`.

**Task 10 — guardrails** ([agents/guardrails.py](agents/guardrails.py)): phone and card last-4 / full-card masking; regex prompt-injection detection; output groundedness check. A sentence is accepted only if **one** context sentence covers ≥ 60% of its content words **and** agrees on numbers (also "seven"→7), negation, contradictory word pairs (broken/unbroken, worn/unworn, …) and risky absolutes ("guaranteed", "tomorrow", "free", "lifetime", "bonus" must appear in the context). Tested cases (all in `tests/test_regressions.py`): negation, 30 vs 7 days, wrong refund time, "even if the seal is broken", "need not be unbroken", "delivered tomorrow, guaranteed", "within 1 working day". **Limitations:** it is text matching, not understanding; contradictions using words outside the lists can slip through, and an answer that merges two context sentences into one can be wrongly rejected. Names/addresses are free text and deliberately not masked.

## Part 3 — API, logging, evaluation

**Task 11** ([api/main.py](api/main.py)): `POST /ask`, `POST /add-document` (2–5 sentences enforced), `DELETE /sessions/{id}`, `GET /health`, WebSocket `/ws/chat?session_id=…`, all with Pydantic models. Validation errors → 422; oversized request → 413; unexpected exception → 500 with a generic body and trace id only (no customer text). The WebSocket catches `WebSocketDisconnect` (client leaves) and also survives a failing message (sends an error, keeps the connection and server alive).

**Task 12** ([api/logging_utils.py](api/logging_utils.py)): one JSON line per request in `logs/requests.jsonl` with `trace_id`, `timestamp`, `duration_ms`, endpoint, status; request text goes through the same `mask_pii`. Tests confirm raw phone/card digits are absent from the log, including on error paths. Governance decisions go to `logs/governance.jsonl` (metadata only).

**Task 13 — evaluation** (`transcripts/task13_evaluation.txt`, `eval/results.json`). 15 queries: 12 required KB topics, 1 order + PII query, 1 out-of-scope, 1 injection. Scores 1–5 from `MockJudge`, which parses a judge prompt containing question, context, answer and expected facts. Accuracy = 1 + 4 × (expected facts found ÷ expected) × (on-topic sentences ÷ sentences); Grounding = 1 + 4 × (supported sentences ÷ sentences); Completeness = 1 + 4 × (expected docs cited ÷ expected docs); Safety = 5 unless a raw phone number or prompt text leaks.

| Metric (average of 15) | Before improvements (`eval/results_before_improvements.json`) | After |
|---|---|---|
| Accuracy | 4.78 | **5.00** |
| Grounding | 5.00 | **5.00** |
| Completeness | 4.91 | **5.00** |
| Safety | 5.00 | **5.00** |

Before: query 3 (delivery SLA) scored accuracy 3.00 because an unrelated size-exchange sentence was appended; query 13 (order status) scored 3.67/3.67 because a policy sentence was appended. Both are fixed by the relevance and routing changes. All 15 per-query rows (all four scores each) are in the transcript and `eval/results.json`.

**Read the 5.00s with care:** the judge is rule-based and the 15 required queries are mostly single-topic, so they are easy. Grounding is high largely *by construction* (the mock copies sentences from the retrieved context). Three sets, all with the same judge:

| Set | Accuracy | Grounding | Completeness | Safety |
|---|---|---|---|---|
| 15 required queries — before the improvement rounds | 4.78 | 5.00 | 4.91 | 5.00 |
| 15 required queries — after round 1 | 5.00 | 5.00 | 5.00 | 5.00 |
| 15 required queries — after round 2 (this round; one regression found and fixed on the way) | **5.00** | **5.00** | **5.00** | **5.00** |
| 12-query **stress set** — before round 2 (`eval/stress_results_before_defect_fix.json`) | 4.42 | 4.67 | 4.50 | 5.00 |
| 12-query stress set — after round 2 (`transcripts/task13_stress_evaluation.txt`) | **5.00** | **5.00** | **5.00** | **5.00** |
| 10-query held-out set, first run after round 2 (`eval/heldout_results_before_duration_fix.json`) | 4.20 | 5.00 | 5.00 | 5.00 |
| 10-query held-out set, after the COD fix only (hair-dryer expectation still "12 month") | 4.60 | 5.00 | 5.00 | 5.00 |
| 10-query held-out set, after the COD fix **and** the corrected hair-dryer expectation (`transcripts/task13_heldout_evaluation.txt`) | **5.00** | **5.00** | **5.00** | **5.00** |

The stress set is the set the two defects were diagnosed on, so its 5.00s mean "those defects are fixed", **not** "the system is perfect". The held-out set was written after the fixes and first scored 4.20; it found problems, and the last two rows are the result of fixing them, so **it is no longer an untuned check** (a fresh set would be needed to measure generalisation again). The two failures it found:
* **COD refund time — fixed (code).** "How long will a refund take if I paid cash on delivery?" returned the right document but the wrong sentence. Root cause: the sentence picker ranks by embedding similarity plus shared words, so the generic sentence "Refunds for cash on delivery orders are paid to the customer's bank account or as store credit." (0.793, 4 shared words) beat "A bank refund takes 5 to 7 working days after the returned item passes quality checks." (0.687, 2 shared words), and nothing told it that "how long" asks for a duration. Fix: duration questions (`how long`, `how soon`, `how quickly`, `how many days/hours/weeks/months`) give a +0.12 bonus to sentences that state a duration (`rag/relevance.py`, `rag/generate.py`). It is a *bonus*, not a filter. The answer is now the 5-to-7-working-days sentence (plus the generic one), copied from the policy.
* **Hair dryer warranty — expectation corrected, code untouched.** I checked the written knowledge base: it never mentions hair dryers. It only says "Electronics carry a 12 month brand warranty…" and "Home appliances carry a 6 month warranty." My original expectation ("12 month") came from a comment in `dataset.py`, not from the policy, so it was an unsupported assumption. The system's answer ("Physical damage and water damage are not covered by any warranty. Home appliances carry a 6 month warranty.") is copied from the policy and is a defensible reading, so the code was not changed. The expectation now accepts either supported statement (`"12 month|6 month"`), and a test checks the alternatives are not a free pass. Honest caveat: the answer commits to one reading and does not tell the customer the product category is unclear.

**Collateral check for the COD fix** (`tests/test_duration_answers.py`): 32 refund / return / delivery / warranty / exchange / escalation questions (`tests/snapshot_questions.py`) were answered before and after. **Exactly two answers changed, both improvements:** the COD question, and "How long are loyalty points valid?" (before it led with the irrelevant "Each 1 loyalty point is worth 1 INR at checkout."; now just "Points expire 12 months after they are earned."). The other 30 are identical, and the reviewed answers are frozen as a test snapshot (`tests/answer_snapshot.json`; pre-fix answers in `tests/answer_snapshot_before_duration_fix.json`). The snapshot also records one answer that is still **wrong**, not fixed: "How long does delivery take to remote areas?" returns the other-towns and metro sentences and misses "Remote pin codes may need up to 10 working days."; the snapshot guards against *unintended* changes, it does not certify every answer.

## Part 4 — Governance and resilience

**Task 14 — AutoGen review** ([review/autogen_review.py](review/autogen_review.py)): `RoundRobinGroupChat([Policy_Compliance_Reviewer, Final_Editor], max_turns=2, custom_message_types=[StructuredMessage[Verdict]])`; the editor has `output_content_type=Verdict` (`approved: bool, final_answer: str, reason: str`). Input = QUESTION + draft + retrieved context. Demo (`transcripts/full_demo.txt`): clean draft → `approved=True`, unchanged; injected claim ("free lifetime insurance and a 500 INR bonus") → removed; a *grounded but irrelevant* home-products sentence → removed (`irrelevant`). The service re-checks groundedness of the revised answer and refuses if it is not grounded. The reviewer also checks the answer type: a single-part "who …" question whose draft names no role is flagged and removed. Limitation: relevance is judged by product category, order-vs-policy intent and answer type only, so other off-topic sentences inside the same category can still pass.

**Task 15 — four-layer governance (layers implemented):**
* *Application / least autonomy:* [governance/access.py](governance/access.py). Every agent is created through `make_agent`, which calls `assert_tool_allowed(role, tool)`. `check_order_status` is allowed only for "Order Lookup Agent"; giving it to any other agent raises `PermissionError` before the agent exists (and records an `access_denied` audit event). `build_crew` also never wires it elsewhere.
* *Risk classification:* **Medium.** It answers customer-support questions and reads order records (the brief's "customer support tickets" class), and wrong answers can affect refunds or deliveries. It is not High: no medical, hiring or payment data (card numbers are masked, only fabricated orders are used) and it cannot move money or change an order. More than Low because answers have real customer impact, hence guardrails, a review stage and human escalation.
* *Runtime:* [governance/budget.py](governance/budget.py) estimates tokens per request (≈4 characters per token + 600-token agent-prompt overhead) against a **1,000-token / $0.0005 cap**; oversized requests are rejected before the crew runs (estimated 1,800 tokens → rejected; API returns 413) and audited. The PDF does not require human-approval steps, so none were added.
* *Audit trail:* [governance/audit.py](governance/audit.py) logs access denials, budget rejections, injection blocks, groundedness refusals and cache clears — metadata only, never customer text.

**Task 16 — cache** ([governance/cache.py](governance/cache.py)): in-memory, key = normalized query + strategy. First call → miss, 1 LLM call; the repeated query (different case/spacing) → hit, **LLM calls still 1**, 0.02 ms vs 48.7 ms. The cache stores **policy answers only** (the order id and PII are stripped from the cached query), so no customer data is cached; it is cleared when a document is added (audited).

## Tests
`.\.venv\Scripts\python.exe -m pytest -q` → **177 passed** (`transcripts/pytest_results.txt`): 36 original + 57 round-1 regression tests ([tests/test_regressions.py](tests/test_regressions.py)) + 25 round-2 tests + 39 duration-answer tests ([tests/test_duration_answers.py](tests/test_duration_answers.py): COD answer from the written policy, duration detection, bonus-not-filter, hair-dryer expectation checked against the KB, exactly-two-answers-changed guard, 32-question answer snapshot). The 25 round-2 tests cover the two stress defects ([tests/test_stress_defects.py](tests/test_stress_defects.py): product-noun questions, glossary evidence and KB-disjointness, junk still rejected, clarification, same-message / session / order resolution, session isolation and reset, who-gate, reviewer, relative-"that" regression, mixed-question regression). Round-1 tests cover per-category return policies, intent routing, genuine tool invocation (exact `TOOL_LOG`), missing/returned/refunded orders, multi-topic retrieval, unsupported-question rejection, groundedness contradictions, AutoGen approve/revise/unsafe/all-removed, memory isolation/reset, PII in memory/cache/logs, injection, budget and access audit, API error handling and WebSocket recovery. Round 4 added 20 tests in [tests/test_round4_defects.py](tests/test_round4_defects.py) (157 + 20 = 177).

## Limitations and known issues
* The mock generator is simple. Grounding is high by design; a real LLM could hallucinate, and the guardrails/review stage were exercised mostly with injected drafts.
* A single cosine threshold cannot separate every legitimate question from near-domain junk. The product glossary removes the product-noun cause, but it is a hand-written list: a product it does not know (or a vague question with no product, category or topic word) can still be rejected or sent to clarification. Topic hints (contact/escalate/complain) are also a hand-written list.
* Within the right document the mock sentence picker can still choose a less useful sentence. "When will …" is treated as a duration question only when followed by an arrival verb (reach / arrive / credited / received / refunded / delivered / processed); other "when will …" wording may still return a general sentence.
* Ambiguous appliances (hair dryer, kettle, microwave, mixer grinder …) are a hand-written list ([rag/relevance.py](rag/relevance.py) `AMBIGUOUS_APPLIANCE_RE`); an appliance not on it is answered with whatever the embeddings pick. Only warranty and return/exchange questions trigger the "confirm the category" answer. The remote-area, village and COD rewrites are likewise small hand-written phrase lists.
* The held-out set is no longer untuned (its failures were fixed), so its 5.00 says little about unseen questions.
* The "who" answer-type check and the role word list are narrow (agent, lead, manager, team, support, level, human, supervisor).
* When "it" is resolved from the previous question, only the last product/category word is used; it can pick the wrong item if the customer changed topic.
* Multi-query retrieval lowers retrieval-level precision on vague sub-questions (final answers filter it).
* Names and addresses are not masked. Memory is in-process only.
* Dependency note: `autogen-agentchat` 0.7 pins `protobuf` 5.x, so `googleapis-common-protos==1.70.0` is pinned for CrewAI/ChromaDB to import.
* One API test failed once during development with a wrong sentence returned; it could not be reproduced in later clean runs (the answer-selection code was rewritten afterwards).
* The first run needs internet once to download the embedding model.

## Improvement history and extra evidence

The detailed before/after history is kept in **[docs/IMPROVEMENT_HISTORY.md](docs/IMPROVEMENT_HISTORY.md)**: improvement round 1 (relevance, routing, multi-query retrieval, stronger groundedness), round 2 (the two stress-evaluation defects), round 3 (COD duration answers) and round 4 (final audit). Each entry lists the reproduced problem, root cause, fix, regression tests and the before/after evaluation numbers. The raw evidence is in [transcripts/](transcripts/) and the earlier result files in [eval/](eval/) (`results_before_improvements.json`, `stress_results_before_defect_fix.json`, `heldout_results_before_duration_fix.json`).

**Round 4 in one paragraph.** A final audit reproduced four known weaknesses and fixed them with 20 new regression tests: remote-area delivery wording, "when will my refund reach…" timing questions (plus the "COD" abbreviation), appliances whose category the policy does not state (hair dryer, kettle … now answered with both written warranty rules and a request to confirm the category), and an unrelated sentence added to a multi-topic answer. One regression introduced by the first attempt was caught by the existing suite and fixed. Four older tests that had pinned the previously documented wrong answers were updated. No knowledge-base text, dataset, seed, threshold or dependency was changed.
