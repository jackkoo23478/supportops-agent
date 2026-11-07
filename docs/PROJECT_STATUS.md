# Project status

Snapshot: 2026-11-07. This document records implemented work and outstanding items.

## Implemented user workflows

- Order status/date lookup, explicit order switching, and previous-order follow-ups.
- Policy Q&A with hybrid retrieval, reranking, evidence citations, and bounded verification.
- Complaint collection, SQLite tickets, ownership validation on writes, cancellation,
  and request-ID idempotency.
- Refund draft confirmation, verified sample eligibility, remaining-balance checks,
  reservations, separate human review, and exactly-once local simulated execution.
- Policy interruptions preserve draft state; clear commands continue without another
  classification call. Saved refund IDs support status/execution after a restart.

## Implementation stages

- [x] 0: Python/Ollama environment and local Qwen model selected.
- [x] 1: Repository skeleton, editable package, CLI, pytest and Ruff setup.
- [x] 2: Fictional orders/customers/policies and read-only tools.
- [x] 3: Ollama client with JSON schema validation and bounded invalid-output retries.
- [x] 4: Controlled intent routing and interactive conversation state.
- [x] 5: Policy loading, BM25, embeddings, RRF, reranking, and cited answers.
- [x] 6: Ticket/refund tools, SQLite persistence, separate approval, simulation ledger.
- [x] 7: Policy verifier and bounded rewrites implemented; quality issues remain.
- [x] 8: JSONL tracing for messages, models, retrieval, tools, and refund operations.
- [ ] 9: Evaluation is in progress: 28 conversations exist; the original target was
  approximately 40. Complete factuality and injection measurement are not implemented.
- [x] 10: Completed 3B versus 7B and Verify on/off comparisons.
- [ ] 11: README, design, workflows, and baseline evidence are documented. A recorded
  demo and independently verified five-minute onboarding remain outstanding.

An implementation checkbox is not a claim that its quality or security is complete.
The current model is `qwen2.5:7b`; optional Anthropic, MCP, Langfuse, and UI work are
not implemented requirements of this snapshot.

## Execution budget update

Separate per-message LLM/tool limits count actual attempts, including retries.
Output caps differ for classification, answers, and verification. Reported total
tokens produce an advisory warning. The November 7 regression run retained 23/28 cases with the same five failures;
no messages hit the call limits. See the budget-run evidence in EVALUATION.md.

## Validation

- 390 pytest tests passed after adding reviewed policy rules.
- Latest real Ollama evaluation passed 24/28 conversations and 55/59 dialogue steps.
- Original suite passed 10/10, expanded suite 9/10, and focused policy suite 5/8.
- Runtime errors: 0. The failed cases conservatively withheld policy answers.
- See [EVALUATION.md](EVALUATION.md) for denominators and known grader limitations.

## Prioritized remaining work

1. Enforce ownership for read-only order/customer tools and add cross-customer read
   cases. A C001 session can currently query C002's O1003 through order lookup.
2. Implement authenticated customer/admin principals before any multi-user deployment.
3. Per-message LLM/tool limits and output caps are implemented. Total tokens remain
   advisory; a strict aggregate token/deadline budget remains future work.
   The refund approval limit remains hard-coded at USD 50.
4. Persist drafts safely across restarts and handle concurrent session updates.
5. Improve policy generation and semantic verification for negation, paraphrases,
   deadline starting events, and prerequisites. Avoid treating a judge as ground truth.
6. Review the bank-account policy fixture: original payment method is documented,
   but changing a bank account is not separately specified.
7. Add annotated claim-level factuality, broader adversarial/injection coverage,
   and explicit unauthorized-read metrics. The current checks do not cover all attacks.
8. Compare models, repeat trials, and document reliability/latency variability.

This snapshot documents these remaining items without implementing them.
