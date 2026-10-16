# Nykaa Domain Support Agent (CrewAI) — Final Capstone

**Track completed: Nykaa (E-commerce & Retail).**
Everything runs under `MOCK_LLM` with **zero API keys and zero network access** during graded runs.
All data (orders, policies, names) is fabricated; the policies are fictional and are **not** real Nykaa policy.

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

## How to run

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# One-time only: the first run downloads the embedding model (all-MiniLM-L6-v2, ~90 MB) from Hugging Face.
# After it is cached, the code loads it with local_files_only=True and never touches the network.
.\.venv\Scripts\python.exe -m pytest -q                 # 36 tests
.\.venv\Scripts\python.exe -m demos.run_demos           # all demos, behind a network guard
.\.venv\Scripts\python.exe -m eval.judge_eval           # Task 13 evaluation
.\.venv\Scripts\python.exe -m uvicorn api.main:app      # API on http://127.0.0.1:8000/docs
```

### MOCK_LLM and telemetry confirmation
* **`CREWAI_DISABLE_TELEMETRY=true`, `OTEL_SDK_DISABLED=true`, `CREWAI_TRACING_ENABLED=false`, `ANONYMIZED_TELEMETRY=False` are set** in [llm/__init__.py](llm/__init__.py), which every entry point imports *before* `crewai`/`chromadb`. Set them in your shell too if you like.
* `demos/run_demos.py` installs a **network guard** (any non-loopback socket connect or DNS lookup raises and is counted). The final transcript line reports `outbound connection attempts blocked: 0` ([transcripts/full_demo.txt](transcripts/full_demo.txt)).
* **MOCK_LLM vs a real LLM:** a real LLM *writes* new text; the mock only applies fixed rules (word overlap, JSON merging, rule-based scoring) to the text it is given, so the same input always gives the same output and nothing leaves the machine. Three mocks exist, one per framework: [llm/mock_llm.py](llm/mock_llm.py) (grounded generation), [llm/crew_mock.py](llm/crew_mock.py) (`MockCrewLLM`, a `crewai.llms.base_llm.BaseLLM` subclass) and [llm/autogen_mock.py](llm/autogen_mock.py) (a deterministic AutoGen `ChatCompletionClient`). The judge for Task 13 is `MockJudge` in [eval/judge_eval.py](eval/judge_eval.py). No real LLM is wired in.
* The two CrewAI pitfalls from the brief: (1) the mock reads only **assistant** messages for `Observation:` — never the system prompt whose template contains `Observation: the result of the action` (test `test_mock_ignores_template_observation_text`); (2) tool arguments are chosen from the tool's declared `args_schema` fields (`record_id` / `query`), never from its name.

## Architecture (text)

```
question ─► input guardrails (PII mask, injection check) ─► session memory (LangChain) ─► budget check
   ─► CrewAI crew .kickoff():  Policy Retrieval Agent [rag_lookup → ChromaDB + cache]
                               Order Lookup Agent     [check_order_status → dataset]
                               Response Composer      [merges both → SupportResponse]
   ─► Pydantic validation ─► output groundedness guard ─► AutoGen review (Reviewer → Final Editor, Verdict)
   ─► answer ─► JSON-Lines log (masked)
```
Folders: `dataset.py` · `kb/` (12 policy docs) · `rag/` (chunking, ChromaDB store, grounded generation, calibration, P/R) · `llm/` (3 mocks) · `agents/` (order tool, CrewAI tools/crew, memory, schemas, guardrails, `service.py` pipeline) · `review/` (AutoGen) · `governance/` (access, budget, cache) · `api/` (FastAPI, logging) · `eval/` · `demos/` · `tests/` · `transcripts/`.

## Part 1 — RAG core

**Knowledge base:** 12 documents in [kb/documents.py](kb/documents.py), 2–5 sentences each, one per required topic (return window, COD refund, delivery SLA, reverse pickup, warranty, cancellation, loyalty points, payment failure, size exchange, damaged-item claim, international shipping, escalation matrix). Checked by `tests/test_kb.py`.

**Chunking + ChromaDB:** model `all-MiniLM-L6-v2` (SentenceTransformers, local). Fixed-size = 250 characters, 50 overlap → **23 chunks** in collection `kb_fixed`. Sentence-based = one sentence per chunk → **48 chunks** in collection `kb_sentence`. Both indexed with `collection.upsert()`, cosine space, metadata `doc_id` keeps each chunk's parent document.

**Threshold calibration (Task 4)** — measured top-1 cosine similarity (`transcripts/task4_threshold_calibration.txt`):

| Query type | Sentence collection | Fixed collection |
|---|---|---|
| In-scope (7 queries) | 0.6537, 0.7106, 0.7699, 0.5237, 0.5933, 0.5079, 0.5996 | 0.6105, 0.7149, 0.6601, 0.6082, 0.5774, 0.5031, 0.5015 |
| Out-of-scope (4 queries) | 0.1322, 0.0525, 0.1329, 0.1203 | 0.1192, 0.0895, 0.1154, 0.0692 |

Lowest in-scope 0.5079 vs highest out-of-scope 0.1329 (sentence) → midpoint 0.3204. **Chosen threshold: 0.32.** No tutorial preset (0.5/0.6/0.7) was used — 0.5 would sit right on top of the weakest in-scope query. Demo: `transcripts/task4_grounded_generation.txt` (7 in-scope grounded answers, 2 out-of-scope fallbacks).

**Precision/recall (Task 5)** — top-3 chunks, mapped to parent documents and de-duplicated (`transcripts/task5_chunking_eval.txt` has the full per-query lines):

| # | Query (short) | Relevant docs | Fixed: P / R | Sentence: P / R |
|---|---|---|---|---|
| 1 | return a lipstick | return_window | 1/3=0.333 / 1/1=1.0 | 1/1=1.0 / 1/1=1.0 |
| 2 | COD refund to bank | cod_refund | 1/3=0.333 / 1.0 | 1/2=0.5 / 1.0 |
| 3 | cancel after shipped | order_cancellation | 1/2=0.5 / 1.0 | 1/1=1.0 / 1.0 |
| 4 | parcel arrived broken | damaged_item_claim | 1/2=0.5 / 1.0 | 1/1=1.0 / 1.0 |
| 5 | exchange shoes size | size_exchange | 1/2=0.5 / 1.0 | 1/1=1.0 / 1.0 |
| 6 | earphones warranty | warranty_terms | 1/3=0.333 / 1.0 | 1/1=1.0 / 1.0 |
| 7 | parcel late + who to escalate | delivery_sla, escalation_matrix | 1/2=0.5 / 1/2=0.5 | 1/2=0.5 / 1/2=0.5 |
| | **Average** | | **P 3.000/7=0.429, R 6.5/7=0.929** | **P 6.000/7=0.857, R 6.5/7=0.929** |

**Recommendation:** deploy the **sentence-based** collection. Recall is identical (0.929 for both), but precision is 0.857 versus 0.429 for fixed-size chunks, because 250-character windows straddle two policies and drag in unrelated documents. Both strategies miss `escalation_matrix` on query 7, so recall there is a retrieval limit, not a chunking effect.

## Part 2 — CrewAI, memory, schema, guardrails

**Task 6 — escalation score.** `escalation_score = 0.6 × delayed_shipment + 0.4 × (days_since_created / 30)` (weights sum to 1 → always in [0, 1]; a not-delayed order maxes out at 0.4, so age alone never escalates). **Threshold 0.83** = score of a delayed order at the dataset's **median age (17 days)**: `0.6 + 0.4 × 17/30 = 0.827`; it also requires an open order (Placed/Shipped). In the 60 generated orders this recommends escalating 3 orders: ORD-1023, ORD-1025, ORD-1037. ORD-1009 is delayed (0.9733) but already Delivered, so it is not escalated.

**Task 7 — crew** ([agents/crew.py](agents/crew.py)), run with `.kickoff()`: *Policy Retrieval Agent* (tool `rag_lookup`), *Order Lookup Agent* (tool `check_order_status`), *Response Composer* (no tools; merges the two JSON results into a `SupportResponse`). Demo (`transcripts/full_demo.txt`): the policy query invokes only `rag_lookup`; the order query invokes `rag_lookup` and `check_order_status` with `{'record_id': 'ORD-1023'}`. The Retrieval agent always runs RAG, even for order questions.

**Task 8 — memory** ([agents/memory.py](agents/memory.py)): `InMemoryChatMessageHistory` + `RunnableWithMessageHistory` keyed by `session_id`. Conversation A: "Where is my order ORD-1023?" then "Can I cancel it?" → resolved to `Can I cancel it? (order ORD-1023)`. Conversation B (new session, same follow-up) → `order_id: None`. LangChain prints deprecation warnings for these classes; this is expected.

**Task 9 — schema:** `SupportResponse` ([agents/schemas.py](agents/schemas.py)) is the Composer task's `output_pydantic` and is re-validated in code in `run_crew` and again in `service.core_pipeline`.

**Task 10 — guardrails** ([agents/guardrails.py](agents/guardrails.py)): phone (Indian mobile formats) and card last-4 / full-card masking → `[PHONE_MASKED]`, `[CARD_LAST4_MASKED]`, `[CARD_MASKED]`; regex prompt-injection detection; output-side groundedness check (each answer sentence needs ≥ 60% of its content words in the retrieved context or the order record) and refusal. All four are shown firing in the Task 10 section of the demo transcript. Names and addresses are free text and deliberately **not** masked (out of scope per the brief).

## Part 3 — API, logging, evaluation

**Task 11** ([api/main.py](api/main.py)): `POST /ask`, `POST /add-document` (2–5 sentences enforced), `GET /health`, and WebSocket `/ws/chat?session_id=…`, all with Pydantic models. The WebSocket catches `WebSocketDisconnect`, logs it and ends only that handler; the demo and `test_websocket_multi_turn_and_survives_disconnect` show a second client served afterwards. An oversized `/ask` returns HTTP 413.

**Task 12** ([api/logging_utils.py](api/logging_utils.py)): one JSON line per request in `logs/requests.jsonl` with `trace_id`, `timestamp`, `duration_ms`, endpoint, status. The request text goes through the same `mask_pii`; the demo and tests confirm the raw phone number and card digits are not on disk.

**Task 13 — evaluation** (`transcripts/task13_evaluation.txt`, `eval/results.json`). Scores are 1–5 from `MockJudge`, which parses a judge prompt containing question, context, answer and expected facts. Accuracy = 1 + 4 × (expected facts found ÷ expected) × (on-topic sentences ÷ sentences); Grounding = 1 + 4 × (supported sentences ÷ sentences); Completeness = 1 + 4 × (expected docs cited ÷ expected docs); Safety = 5 unless a raw phone number or prompt text leaks.

| # | Query | Acc | Grd | Cmp | Saf |
|---|---|---|---|---|---|
| 1 | return window (lipstick) | 5.00 | 5.00 | 5.00 | 5.00 |
| 2 | COD refund timeline | 5.00 | 5.00 | 5.00 | 5.00 |
| 3 | delivery SLA metro | **3.00** | 5.00 | 5.00 | 5.00 |
| 4 | reverse pickup | 5.00 | 5.00 | 5.00 | 5.00 |
| 5 | warranty electronics | 5.00 | 5.00 | 5.00 | 5.00 |
| 6 | cancel after shipping | 5.00 | 5.00 | 5.00 | 5.00 |
| 7 | loyalty points | 5.00 | 5.00 | 5.00 | 5.00 |
| 8 | payment failure | 5.00 | 5.00 | 5.00 | 5.00 |
| 9 | size exchange | 5.00 | 5.00 | 5.00 | 5.00 |
| 10 | damaged item | 5.00 | 5.00 | 5.00 | 5.00 |
| 11 | international shipping | 5.00 | 5.00 | 5.00 | 5.00 |
| 12 | escalation matrix | 5.00 | 5.00 | 5.00 | 5.00 |
| 13 | order status + phone number | **3.67** | 5.00 | **3.67** | 5.00 |
| 14 | out-of-scope (cricket) | 5.00 | 5.00 | 5.00 | 5.00 |
| 15 | injection attempt | 5.00 | 5.00 | 5.00 | 5.00 |
| | **Average of 15** | **4.78** | **5.00** | **4.91** | **5.00** |

Honest reading: Grounding is 5.0 largely **by construction** — the mock generator copies sentences from the retrieved context, so it cannot invent facts. The lower scores are real defects: query 3 got an extra unrelated sentence appended, and query 13's order answer had an irrelevant policy sentence added by the Composer.

## Part 4 — Governance and resilience

**Task 14 — AutoGen review** ([review/autogen_review.py](review/autogen_review.py)): `RoundRobinGroupChat([Policy_Compliance_Reviewer, Final_Editor], max_turns=2, custom_message_types=[StructuredMessage[Verdict]])`; the editor has `output_content_type=Verdict` (`approved: bool, final_answer: str, reason: str`). Input = draft + retrieved context. Demo: a clean draft → `approved=True`, unchanged; a draft with an injected claim ("free lifetime insurance and a 500 INR bonus") → `approved=False`, claim removed, reason names it. (AutoGen agent names must be Python identifiers, hence underscores.)

**Task 15 — four-layer governance (the layers I implemented):**
* *Application / least autonomy:* [governance/access.py](governance/access.py). Every agent is created through `make_agent`, which calls `assert_tool_allowed(role, tool)`. `check_order_status` is allowed only for "Order Lookup Agent"; giving it to the Retrieval Agent or the Composer raises `PermissionError` before the agent exists, and `build_crew` also never wires it elsewhere. The demo shows both blocked attempts and that only the Lookup agent holds it.
* *Risk classification:* **Medium.** The system answers customer-support questions and reads order records (the brief's "customer support tickets" class), which can affect a customer's refund or delivery decisions. It is not High: it holds no medical, hiring or payment data (card numbers are masked, only fabricated orders are used) and it cannot move money or change an order. It is more than Low because wrong answers have real customer impact, which is why it has guardrails, a review stage and escalation to humans.
* *Runtime:* [governance/budget.py](governance/budget.py) estimates tokens per request (≈4 characters per token plus a 600-token agent-prompt overhead) against a **1,000-token / $0.0005 cap**. An oversized request is rejected before the crew runs (demo: estimated 1,800 tokens → rejected, no crew run; API returns 413).

**Task 16 — cache** ([governance/cache.py](governance/cache.py)): in-memory, key = normalized query (lower-case, punctuation stripped, spaces collapsed) + strategy. Evidence: first call → miss, 1 LLM call, 29.8 ms; the repeated query (different case/spacing) → hit, **LLM calls still 1**, 0.03 ms. The cache is cleared when a document is added.

## Tests
`.\.venv\Scripts\python.exe -m pytest -q` → **36 passed** (`transcripts/pytest_results.txt`). Covers dataset, KB, both collections, threshold separation, metrics, order tool, crew tool invocation, template-`Observation:` pitfall, least autonomy, memory isolation, schema validation, PII/injection/groundedness guardrails, FastAPI + WebSocket disconnect, JSONL log privacy, the 15-query evaluation, AutoGen approve/revise, budget, cache, and an end-to-end customer flow.

## Limitations and known issues
* Grounding scores are high by design (extractive mock). A real LLM could hallucinate; the guardrail and AutoGen review exist for that case, but were only exercised with injected drafts.
* The mock generator is simple: query 7 ("parcel late … who do I escalate to") retrieves `damaged_item_claim` first and answers poorly; query 3 and 13 have extra sentences (see evaluation).
* The Retrieval agent always runs RAG, even for pure order questions, so the Composer can append an irrelevant policy sentence.
* Names and delivery addresses are not masked. Memory is in-process only.
* Dependency note: `autogen-agentchat` 0.7 pins `protobuf` 5.x, so `googleapis-common-protos==1.70.0` is pinned for CrewAI/ChromaDB to import.
* One API test (`test_add_document_then_retrievable`) failed once during development with a wrong sentence returned; it could not be reproduced in several later clean runs, and the test now cleans up its document.
* The first run needs internet once to download the embedding model.
