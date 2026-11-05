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

## How to run (Windows, PowerShell, Python 3.12)

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
.\.venv\Scripts\python.exe -m eval.judge_eval --heldout       # 10 untuned queries written after the fixes
.\.venv\Scripts\python.exe -m uvicorn api.main:app            # API: http://127.0.0.1:8000/docs
```

### MOCK_LLM and telemetry confirmation
* **`CREWAI_DISABLE_TELEMETRY=true`, `OTEL_SDK_DISABLED=true`, `CREWAI_TRACING_ENABLED=false`, `ANONYMIZED_TELEMETRY=False` are set** in [llm/__init__.py](llm/__init__.py), which every entry point imports *before* `crewai`/`chromadb`.
* `demos/run_demos.py` installs a **network guard** (any non-loopback socket connect or DNS lookup raises and is counted). The final transcript line reports `outbound connection attempts blocked: 0` ([transcripts/full_demo.txt](transcripts/full_demo.txt)).
* **MOCK_LLM vs a real LLM:** a real LLM *writes* new text; the mock only applies fixed rules (embedding scores, sentence selection, JSON merging, rule-based scoring) to the text it is given, so the same input always gives the same output and nothing leaves the machine. Three mocks, one per framework: [llm/mock_llm.py](llm/mock_llm.py) (grounded generation), [llm/crew_mock.py](llm/crew_mock.py) (`MockCrewLLM`, a `crewai.llms.base_llm.BaseLLM` subclass) and [llm/autogen_mock.py](llm/autogen_mock.py) (a deterministic AutoGen `ChatCompletionClient`). The Task 13 judge is `MockJudge` in [eval/judge_eval.py](eval/judge_eval.py). No real LLM is wired in.
* The two CrewAI pitfalls from the brief: (1) the mock reads only **assistant** messages for `Observation:` — never the system prompt whose template contains `Observation: the result of the action` (test `test_mock_ignores_template_observation_text`); (2) tool arguments are chosen from the tool's declared `args_schema` fields (`record_id` / `query`), never from its name.

## Architecture (text)

```
question ─► input guardrails (PII mask, injection check) ─► session memory (LangChain) ─► budget check
   ─► CrewAI crew .kickoff():  Policy Retrieval Agent [rag_lookup → ChromaDB + cache]   (skipped for order-only questions)
                               Order Lookup Agent     [check_order_status → dataset]    (only when an order id is present)
                               Response Composer      [merges both, guided by the question's intent → SupportResponse]
   ─► Pydantic validation ─► output groundedness guard ─► AutoGen review (Reviewer → Final Editor, Verdict)
   ─► groundedness re-check of the revised answer ─► answer ─► JSON-Lines log (masked)
```
Folders: `dataset.py` · `kb/` · `rag/` (chunking, ChromaDB store, relevance, grounded generation, calibration, P/R) · `llm/` (3 mocks) · `agents/` (order tool, routing, CrewAI tools/crew, memory, schemas, guardrails, `service.py`) · `review/` (AutoGen) · `governance/` (access, budget, cache, audit) · `api/` · `eval/` · `demos/` · `tests/` · `transcripts/`.

## Improvement round — what changed and why (before → after)

| Problem found | Root cause | Fix | Where |
|---|---|---|---|
| Beauty question also returned the **home** return sentence | Answer picker scored sentences by shared words ("return", "products" match every category) and appended any sentence above 60% of the best | Sentences are scored by embedding similarity; sentences about *other categories only* are dropped; extra sentences must come from the best sentence's own document **and** score within 0.10 | [rag/relevance.py](rag/relevance.py), [llm/mock_llm.py](llm/mock_llm.py), [rag/generate.py](rag/generate.py) |
| `sources` listed unused docs (e.g. `warranty_terms`) | Sources were taken from all retrieved chunks | Sources = documents of the sentences actually used | [rag/generate.py](rag/generate.py) |
| Missing order `ORD-9999` returned a cancellation-policy sentence; "Where is my order?" returned two random policy sentences | Retrieval ran for every question and the Composer always appended policy text | Question routing (`order` / `policy` / `mixed` / `order_id_missing`); Retrieval agent skips RAG for order-only questions; Composer writes only what the intent needs; clear "not found" / "please share your order id" sentences | [agents/routing.py](agents/routing.py), [llm/crew_mock.py](llm/crew_mock.py) |
| Late parcel + escalation question: recall 0.5 | One vector for a two-topic question | Multi-query retrieval: split into sub-questions, retrieve each | [rag/relevance.py](rag/relevance.py), [rag/evaluate.py](rag/evaluate.py) |
| "order a pizza online" would pass the 0.32 threshold (top-1 = 0.368) | Threshold was calibrated only on clearly unrelated questions | Re-calibrated with near-domain junk; question-level threshold 0.44 (sub-questions without a topic hint also need 0.44; see below) | [rag/generate.py](rag/generate.py) |
| "cannot be returned within 7 days", "within 30 days", "even if the seal is broken" all passed the guard | Guard only counted overlapping words | Each sentence must match ONE context sentence on coverage **and** numbers, negation, contradictory word pairs and unsupported absolutes; re-run after the AutoGen revision | [agents/guardrails.py](agents/guardrails.py), [agents/service.py](agents/service.py) |
| AutoGen reviewer approved an irrelevant home sentence | Reviewer never saw the question and only checked grounding | Reviewer receives QUESTION + DRAFT + CONTEXT and reports unsupported / irrelevant / unsafe / missing; Final Editor removes only flagged sentences | [llm/autogen_mock.py](llm/autogen_mock.py), [review/autogen_review.py](review/autogen_review.py) |

**Chatbot answers before → after**

| Question | Before | After |
|---|---|---|
| What is the return policy for beauty products? | "Beauty … unbroken. **Home products can be returned within 14 days of delivery.**" (sources: return_window, warranty_terms) | "Beauty products can be returned within 7 days of delivery, and only if the seal is unbroken." (sources: return_window) |
| What is the status of order ORD-9999? | "An order can be cancelled for free while its status is Placed." | "I could not find an order with id ORD-9999, so please check the order number." |
| Where is my order? | two unrelated policy sentences | "Please share your order id (for example ORD-1001) so I can check the status." |
| My parcel is very late, how long should delivery take and who do I escalate to? | "A damaged or wrong item must be reported within 48 hours of delivery." | delivery SLA sentences + the escalation sentence (sources: delivery_sla, escalation_matrix) |
| How do I order a pizza online? | would be answered (top-1 0.368 > 0.32) | "I don't know … connect you to a human agent." |

## Round 4 — final audit (defects reproduced, fixed, tested)

Audit method: ran the 157-test suite (green), reproduced each known README limitation on the unfixed code, and wrote 14 **unseen** probe questions ([eval/unseen_probe.py](eval/unseen_probe.py)) *before* any fix. First run on the unfixed code: **12/14** (`transcripts/unseen_probe_initial.txt`). After the fixes: **14/14** (`transcripts/unseen_probe_after_fixes.txt`). The probe set is a rule-based fact check written once, so it is a smoke test, not a benchmark; it is now also tuned-on for the fixes below.

| Defect (reproduced first) | Root cause | Fix | Regression tests |
|---|---|---|---|
| "How long does delivery take to remote areas?" returned the metro / other-town sentences | Customer wording "remote areas / villages" is not linked to the policy's "remote pin codes" by the embedding model | `normalize_phrases` rewrites those phrases for retrieval only ([rag/relevance.py](rag/relevance.py)) | `test_remote_area_wording_finds_remote_pin_code_rule` (3 phrasings) |
| "When will my cash on delivery refund reach my bank account?" returned only the generic COD sentence | "when will …" was not a duration question | `WHEN_ARRIVES_RE`: "when will/would/should … reach/arrive/credited/received/refunded/delivered/processed" counts; "When can I cancel?" still does not | `test_when_will_cod_refund_arrive_returns_the_timeline`, `test_when_question_detection_is_narrow` |
| "When will the money for my COD return be credited to my bank?" matched the payment-failure policy (found by the unseen probe) | "COD" is unknown to the embedding model; the timeline sentence also ranked 4th (outside top 3) | "COD" → "cash on delivery" for retrieval; duration questions retrieve 2 chunks deeper | `test_cod_abbreviation_is_expanded_for_retrieval_only` |
| Hair dryer / kettle / microwave warranty answered with one category, or fell back | The policy never says whether such an item is Electronics or Home appliances | For a warranty or return/exchange question about a listed appliance with no stated category, the answer gives **both written rules** plus the fixed sentence "The policy does not say whether this item counts as Electronics or Home appliances, so please confirm its category." (no KB facts changed) | `test_ambiguous_appliance_warranty_states_both_rules_and_asks` (4 phrasings), `test_stated_category_is_not_ambiguous` |
| "Do you deliver to remote pin codes and how long will it take?" added an unrelated return-window sentence | The vague second part ("how long will it take?") was searched alone and matched any sentence with a duration | A part with no product/policy anchor and a vague pronoun is joined to the part before it (only when the question has no earlier product word that the existing reference resolution can use) | `test_vague_second_part_does_not_add_unrelated_sentence`, `test_genuine_two_topic_question_still_gets_both_topics` |

**A regression I introduced and caught:** my first version of the merge joined *every* anchorless "it" part, so "warranty on electronics and who do I contact if it fails" lost its escalation answer (`tests/test_stress_defects.py` failed). The merge now applies only when the question has no earlier product/category word. Separately, the held-out set's hair-dryer question dropped to accuracy 3.67 (set average 4.87) because the judge counted the new fixed sentence as off-topic; the judge's on-topic rule now accepts that one fixed system sentence (`test_judge_counts_the_fixed_category_sentence_as_on_topic` also checks that a genuinely unrelated sentence is still penalised). That is a judge-definition change, stated here rather than hidden.

**Test updates (not hidden):** four older tests encoded the previous, documented-as-wrong behaviour and were updated: the 32-answer snapshot (`tests/answer_snapshot.json`; old copy kept as `answer_snapshot_before_round4.json`; exactly the 2 known-bad answers changed, the other 30 are identical), the "changed vs before the duration fix" set, `test_duration_detection` ("When will my refund arrive?" is now a duration question), and the hair-dryer test. No KB text, seed, dataset, threshold (0.44 / floor 0.32) or dependency was changed; the calibration output is unchanged.

**Final results (this round, all executed):** pytest 177 passed; 15 required queries 5.00 / 5.00 / 5.00 / 5.00; stress set (12) 5.00 x4; held-out set (10) 5.00 x4 (**not** an independent measure, see above); precision/recall unchanged; network guard: 0 outbound connection attempts.

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
* **The cost of the stricter threshold, and how it was reduced without lowering it:** short legitimate questions that name a *product* ("How do I return my earphones?" 0.33) scored below the junk, so no single cosine threshold separated everything. The threshold stays at 0.44; the product-glossary fix in the next section removes the cause (see the evidence table there).

## Second improvement round — the two stress-evaluation defects

### Defect 1 — "How do I return my earphones?" was falsely rejected
* **Root cause (measured):** not the threshold. The embedding model does not link product nouns to our category words: "…my earphones" 0.33, "…my headphones" 0.35, "…my kurta" 0.35, "…my cushion" 0.33, but "…my electronics" 0.556, "…my lipstick" 0.563, "…my sandals" 0.598. Lowering the threshold to ~0.33 would have let "order a pizza online" (0.368) through.
* **Fix:** a product glossary ([rag/relevance.py](rag/relevance.py), `PRODUCT_NOUNS`, `normalize_products`) replaces a product noun with its category word *for retrieval only* ("my earphones" → "my electronics"). Glossary words must not occur in the knowledge base (tested), so exact matches such as "nail polish" are untouched. The threshold stays **0.44**.
* **Evidence** (`python -m rag.calibrate`, sentence collection):

| Question | Top-1 before | Top-1 after | Result before → after |
|---|---|---|---|
| How do I return my earphones? | 0.3316 | 0.5563 | rejected → answered (electronics, 10 days) |
| How can I return my headphones? | 0.3508 | 0.5575 | rejected → answered |
| How do I return my trimmer? | 0.2507 | 0.5563 | rejected → answered |
| How do I return my kurta? | 0.3490 | 0.6871 | rejected → answered (apparel, 15 days) |
| How do I return my cushion? | 0.3307 | 0.7101 | rejected → answered (home, 14 days) |
| Can I return my shampoo? | 0.4171 | 0.7580 | rejected → answered (beauty, 7 days) |
| How do I return my sneakers? / mascara? | 0.5768 / 0.4784 | 0.6566 / 0.7275 | already answered, now higher |

  The glossary changed **0 of the 8** out-of-scope questions; the highest out-of-scope top-1 is still 0.3682 (pizza), still rejected.

### Defect 2 — "Who do I contact if it fails?" got an unrelated sentence
* **Where the problem was (all five suspects checked):**
  * *Understanding:* "it" had nothing to point at, and nothing noticed.
  * *Retrieval:* the fragment "who do I contact if it fails" matches any sentence containing "fail", so `reverse_pickup` ("If both attempts fail, the return request is closed.") and `damaged_item_claim` outranked the escalation matrix (which never uses the word "contact").
  * *Memory:* only order ids were remembered, never the topic.
  * *Composition:* the Composer faithfully passed on what it was given.
  * *AutoGen review:* it could not tell that a "who" question needs a role in the answer, and the sentence was grounded and in the right category.
* **Fixes:**
  1. **Clarify instead of guessing** ([agents/routing.py](agents/routing.py)): a question with a vague "it/they/them" (or a trailing "that/this") and no product, category or policy-topic word is routed to `clarify`; the answer is "Could you tell me which item or order you are asking about…". No tool runs and nothing is invented.
  2. **Use the conversation** ([agents/memory.py](agents/memory.py)): if the session has an order id, "it" means that order; otherwise "it" is pointed at the product/category of the most recent earlier question ("…if my earphones fails?"). A new or reset session has nothing to resolve, so it asks.
  3. **Resolve within one message** ([rag/generate.py](rag/generate.py)): "warranty on electronics and who do I contact if it fails" → the second part becomes "…if my electronics fails".
  4. **Answer type + topic hint**: a "who …" question may only be answered by a sentence that names a role (agent, lead, manager, team, support, level); contact/escalate/complain words search the escalation matrix directly and give it a small boost.
  5. **Reviewer** ([llm/autogen_mock.py](llm/autogen_mock.py)): a single-part "who" question whose draft names no role is flagged `irrelevant` and removed.
* **Results** (`transcripts/full_demo.txt`, "STRESS-DEFECT FIXES"): no context → clarification; same message → warranty sentence + the support-agent/manager sentences; after an earphones question → resolved to "my earphones" and answered from the escalation matrix; fresh session → clarification; "How do I contact the pizza place?" → still refused.

### Problems this round introduced or exposed, and their fixes
* **Regression caught by the 15-query evaluation:** my first "vague pronoun" pattern treated the relative "that" in "Who handles a case **that** is older than 72 hours?" as a reference, so query 12 asked for clarification (accuracy/completeness 5.00 → 1.00, averages 4.73). The pattern now only counts it/they/them, a trailing "that/this", or "is that …". Permanent test: `test_relative_that_is_not_a_dangling_reference`.
* **Junk sub-question hole:** with the 0.32 floor a junk part inside a valid question would have been answered; non-hinted sub-questions now need 0.44 and are reported as unanswered ("I could not find a policy answer for the rest of your question…").
* **Found by the held-out set:** the order-only part of "What is the status of order ORD-1047 and who do I contact if it is late?" was searched in the policy KB and pulled in a cancellation sentence; order-only parts are now skipped (`test_order_part_of_a_mixed_question_does_not_search_the_policy_kb`).

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
