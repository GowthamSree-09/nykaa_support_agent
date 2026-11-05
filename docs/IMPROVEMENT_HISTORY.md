# Improvement history (detailed)

This file keeps the full before/after history of the improvement rounds that was previously in the main README.
The main [README](../README.md) holds the current design, results and verification summary; the capstone requirements
are tracked task by task in [REQUIREMENT_AUDIT.md](../REQUIREMENT_AUDIT.md). Nothing here has been shortened or rewritten.
Several tables compare an earlier state with the current one; the numbers are the ones recorded at the time.

## Improvement round — what changed and why (before → after)

| Problem found | Root cause | Fix | Where |
|---|---|---|---|
| Beauty question also returned the **home** return sentence | Answer picker scored sentences by shared words ("return", "products" match every category) and appended any sentence above 60% of the best | Sentences are scored by embedding similarity; sentences about *other categories only* are dropped; extra sentences must come from the best sentence's own document **and** score within 0.10 | [rag/relevance.py](../rag/relevance.py), [llm/mock_llm.py](../llm/mock_llm.py), [rag/generate.py](../rag/generate.py) |
| `sources` listed unused docs (e.g. `warranty_terms`) | Sources were taken from all retrieved chunks | Sources = documents of the sentences actually used | [rag/generate.py](../rag/generate.py) |
| Missing order `ORD-9999` returned a cancellation-policy sentence; "Where is my order?" returned two random policy sentences | Retrieval ran for every question and the Composer always appended policy text | Question routing (`order` / `policy` / `mixed` / `order_id_missing`); Retrieval agent skips RAG for order-only questions; Composer writes only what the intent needs; clear "not found" / "please share your order id" sentences | [agents/routing.py](../agents/routing.py), [llm/crew_mock.py](../llm/crew_mock.py) |
| Late parcel + escalation question: recall 0.5 | One vector for a two-topic question | Multi-query retrieval: split into sub-questions, retrieve each | [rag/relevance.py](../rag/relevance.py), [rag/evaluate.py](../rag/evaluate.py) |
| "order a pizza online" would pass the 0.32 threshold (top-1 = 0.368) | Threshold was calibrated only on clearly unrelated questions | Re-calibrated with near-domain junk; question-level threshold 0.44 (sub-questions without a topic hint also need 0.44; see below) | [rag/generate.py](../rag/generate.py) |
| "cannot be returned within 7 days", "within 30 days", "even if the seal is broken" all passed the guard | Guard only counted overlapping words | Each sentence must match ONE context sentence on coverage **and** numbers, negation, contradictory word pairs and unsupported absolutes; re-run after the AutoGen revision | [agents/guardrails.py](../agents/guardrails.py), [agents/service.py](../agents/service.py) |
| AutoGen reviewer approved an irrelevant home sentence | Reviewer never saw the question and only checked grounding | Reviewer receives QUESTION + DRAFT + CONTEXT and reports unsupported / irrelevant / unsafe / missing; Final Editor removes only flagged sentences | [llm/autogen_mock.py](../llm/autogen_mock.py), [review/autogen_review.py](../review/autogen_review.py) |

**Chatbot answers before → after**

| Question | Before | After |
|---|---|---|
| What is the return policy for beauty products? | "Beauty … unbroken. **Home products can be returned within 14 days of delivery.**" (sources: return_window, warranty_terms) | "Beauty products can be returned within 7 days of delivery, and only if the seal is unbroken." (sources: return_window) |
| What is the status of order ORD-9999? | "An order can be cancelled for free while its status is Placed." | "I could not find an order with id ORD-9999, so please check the order number." |
| Where is my order? | two unrelated policy sentences | "Please share your order id (for example ORD-1001) so I can check the status." |
| My parcel is very late, how long should delivery take and who do I escalate to? | "A damaged or wrong item must be reported within 48 hours of delivery." | delivery SLA sentences + the escalation sentence (sources: delivery_sla, escalation_matrix) |
| How do I order a pizza online? | would be answered (top-1 0.368 > 0.32) | "I don't know … connect you to a human agent." |

## Round 4 — final audit (defects reproduced, fixed, tested)

Audit method: ran the 157-test suite (green), reproduced each known README limitation on the unfixed code, and wrote 14 **unseen** probe questions ([eval/unseen_probe.py](../eval/unseen_probe.py)) *before* any fix. First run on the unfixed code: **12/14** (`transcripts/unseen_probe_initial.txt`). After the fixes: **14/14** (`transcripts/unseen_probe_after_fixes.txt`). The probe set is a rule-based fact check written once, so it is a smoke test, not a benchmark; it is now also tuned-on for the fixes below.

| Defect (reproduced first) | Root cause | Fix | Regression tests |
|---|---|---|---|
| "How long does delivery take to remote areas?" returned the metro / other-town sentences | Customer wording "remote areas / villages" is not linked to the policy's "remote pin codes" by the embedding model | `normalize_phrases` rewrites those phrases for retrieval only ([rag/relevance.py](../rag/relevance.py)) | `test_remote_area_wording_finds_remote_pin_code_rule` (3 phrasings) |
| "When will my cash on delivery refund reach my bank account?" returned only the generic COD sentence | "when will …" was not a duration question | `WHEN_ARRIVES_RE`: "when will/would/should … reach/arrive/credited/received/refunded/delivered/processed" counts; "When can I cancel?" still does not | `test_when_will_cod_refund_arrive_returns_the_timeline`, `test_when_question_detection_is_narrow` |
| "When will the money for my COD return be credited to my bank?" matched the payment-failure policy (found by the unseen probe) | "COD" is unknown to the embedding model; the timeline sentence also ranked 4th (outside top 3) | "COD" → "cash on delivery" for retrieval; duration questions retrieve 2 chunks deeper | `test_cod_abbreviation_is_expanded_for_retrieval_only` |
| Hair dryer / kettle / microwave warranty answered with one category, or fell back | The policy never says whether such an item is Electronics or Home appliances | For a warranty or return/exchange question about a listed appliance with no stated category, the answer gives **both written rules** plus the fixed sentence "The policy does not say whether this item counts as Electronics or Home appliances, so please confirm its category." (no KB facts changed) | `test_ambiguous_appliance_warranty_states_both_rules_and_asks` (4 phrasings), `test_stated_category_is_not_ambiguous` |
| "Do you deliver to remote pin codes and how long will it take?" added an unrelated return-window sentence | The vague second part ("how long will it take?") was searched alone and matched any sentence with a duration | A part with no product/policy anchor and a vague pronoun is joined to the part before it (only when the question has no earlier product word that the existing reference resolution can use) | `test_vague_second_part_does_not_add_unrelated_sentence`, `test_genuine_two_topic_question_still_gets_both_topics` |

**A regression I introduced and caught:** my first version of the merge joined *every* anchorless "it" part, so "warranty on electronics and who do I contact if it fails" lost its escalation answer (`tests/test_stress_defects.py` failed). The merge now applies only when the question has no earlier product/category word. Separately, the held-out set's hair-dryer question dropped to accuracy 3.67 (set average 4.87) because the judge counted the new fixed sentence as off-topic; the judge's on-topic rule now accepts that one fixed system sentence (`test_judge_counts_the_fixed_category_sentence_as_on_topic` also checks that a genuinely unrelated sentence is still penalised). That is a judge-definition change, stated here rather than hidden.

**Test updates (not hidden):** four older tests encoded the previous, documented-as-wrong behaviour and were updated: the 32-answer snapshot (`tests/answer_snapshot.json`; old copy kept as `answer_snapshot_before_round4.json`; exactly the 2 known-bad answers changed, the other 30 are identical), the "changed vs before the duration fix" set, `test_duration_detection` ("When will my refund arrive?" is now a duration question), and the hair-dryer test. No KB text, seed, dataset, threshold (0.44 / floor 0.32) or dependency was changed; the calibration output is unchanged.

**Final results (this round, all executed):** pytest 177 passed; 15 required queries 5.00 / 5.00 / 5.00 / 5.00; stress set (12) 5.00 x4; held-out set (10) 5.00 x4 (**not** an independent measure, see above); precision/recall unchanged; network guard: 0 outbound connection attempts.

## Second improvement round — the two stress-evaluation defects

### Defect 1 — "How do I return my earphones?" was falsely rejected
* **Root cause (measured):** not the threshold. The embedding model does not link product nouns to our category words: "…my earphones" 0.33, "…my headphones" 0.35, "…my kurta" 0.35, "…my cushion" 0.33, but "…my electronics" 0.556, "…my lipstick" 0.563, "…my sandals" 0.598. Lowering the threshold to ~0.33 would have let "order a pizza online" (0.368) through.
* **Fix:** a product glossary ([rag/relevance.py](../rag/relevance.py), `PRODUCT_NOUNS`, `normalize_products`) replaces a product noun with its category word *for retrieval only* ("my earphones" → "my electronics"). Glossary words must not occur in the knowledge base (tested), so exact matches such as "nail polish" are untouched. The threshold stays **0.44**.
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
  1. **Clarify instead of guessing** ([agents/routing.py](../agents/routing.py)): a question with a vague "it/they/them" (or a trailing "that/this") and no product, category or policy-topic word is routed to `clarify`; the answer is "Could you tell me which item or order you are asking about…". No tool runs and nothing is invented.
  2. **Use the conversation** ([agents/memory.py](../agents/memory.py)): if the session has an order id, "it" means that order; otherwise "it" is pointed at the product/category of the most recent earlier question ("…if my earphones fails?"). A new or reset session has nothing to resolve, so it asks.
  3. **Resolve within one message** ([rag/generate.py](../rag/generate.py)): "warranty on electronics and who do I contact if it fails" → the second part becomes "…if my electronics fails".
  4. **Answer type + topic hint**: a "who …" question may only be answered by a sentence that names a role (agent, lead, manager, team, support, level); contact/escalate/complain words search the escalation matrix directly and give it a small boost.
  5. **Reviewer** ([llm/autogen_mock.py](../llm/autogen_mock.py)): a single-part "who" question whose draft names no role is flagged `irrelevant` and removed.
* **Results** (`transcripts/full_demo.txt`, "STRESS-DEFECT FIXES"): no context → clarification; same message → warranty sentence + the support-agent/manager sentences; after an earphones question → resolved to "my earphones" and answered from the escalation matrix; fresh session → clarification; "How do I contact the pizza place?" → still refused.

### Problems this round introduced or exposed, and their fixes
* **Regression caught by the 15-query evaluation:** my first "vague pronoun" pattern treated the relative "that" in "Who handles a case **that** is older than 72 hours?" as a reference, so query 12 asked for clarification (accuracy/completeness 5.00 → 1.00, averages 4.73). The pattern now only counts it/they/them, a trailing "that/this", or "is that …". Permanent test: `test_relative_that_is_not_a_dangling_reference`.
* **Junk sub-question hole:** with the 0.32 floor a junk part inside a valid question would have been answered; non-hinted sub-questions now need 0.44 and are reported as unanswered ("I could not find a policy answer for the rest of your question…").
* **Found by the held-out set:** the order-only part of "What is the status of order ORD-1047 and who do I contact if it is late?" was searched in the policy KB and pulled in a cancellation sentence; order-only parts are now skipped (`test_order_part_of_a_mixed_question_does_not_search_the_policy_kb`).


## Evaluation history and held-out findings (moved verbatim from the former README Part 3)

These tables and notes record how the evaluation numbers changed across the improvement rounds. Statements such as "still wrong, not fixed" describe the state at the time they were written; the remote-area, COD-timing and hair-dryer gaps were closed later in round 4 (see the Round 4 section above).

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
