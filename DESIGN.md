# Design

## Controlled workflow

SupportOps uses a fixed Python workflow around a local model. General messages
are classified into `order_status`, `policy_question`, `refund`, `complaint`, or
`other`. Pydantic validates structured model output. The model does not decide
administrator authority and does not execute arbitrary tool names or arguments.

Conversation state tracks the demo customer, last order, pending complaint/refund,
and most recent saved action IDs. Explicit continuation commands are handled before
classification. Policy questions can interrupt a draft without discarding it.

## Retrieval

Policy Markdown level-two headings are stable source IDs. The index includes
source IDs and passage text. Each retriever supplies up to five positive-scoring
candidates; reciprocal-rank fusion with constant 60 retains five; a cross-encoder
reranks those and returns three. Ranking values are not confidence probabilities.

Embedding and reranker models load lazily and are reused. The default embedding
model is all-MiniLM-L6-v2 and the reranker is bge-reranker-v2-m3. No vector database
is needed for the small fictional corpus. Rebuild the index when source content
or the embedding model changes.

## Policy verification

Generation receives only the question and retrieved policy passages. The verifier
first rejects detectable citation, quantity, currency, unit, and action-claim
problems. It then requests sentence-level evidence from a separate structured
call to the same local model. Quotes must exist in cited passages, and every answer
sentence must have evidence before a positive verdict can pass.

Negative verdicts also require real evidence; ungrounded criticism can trigger
one additional judging call, then a conservative fallback. Valid criticism is
used to regenerate from the original sources. There are at most three drafts;
retrieval and business actions are not repeated during rewriting. Exact quote
membership does not prove entailment, and correlated model errors remain.

Quantity logic extracts limited approval rules from cited text, uses Decimal to
check a question-supplied USD amount, and preserves duration bounds. Missing
qualifiers inherit only a unique cited interpretation. This is deliberately
narrow; it is not a general arithmetic or natural-language theorem prover.

## Tickets and refunds

Tools validate input models and check customer/order ownership before writes.
Ticket and refund request IDs are unique SQLite keys. Identical retries return
the saved result; reusing an ID with different details fails.

Refund facts come from a server-maintained sample context file. Delivery dates
come from the order store. The eligibility evaluator checks the original request
date, standard-return conditions, receipt/inspection, amount, and available balance.
Damage exceptions require manual review. USD 50 or less can be automatically
approved when eligible; larger amounts await a separate local review command.

Approval is distinct from execution. Pending/approved amounts reserve balance;
rejection releases it. Execution reloads verified facts, checks approval, and
writes a unique simulation transaction and `executed` status in one transaction.
Executed amounts remain consumed. A failed update rolls back the ledger insert.
No external payment request is made.

The baseline balance excludes external historical refunds but must not also
subtract this local store's reservations or simulated transactions. A future real
payment integration needs provider idempotency and recovery beyond local SQLite
atomicity.

## Observability and evaluation

Context-local trace spans preserve message/session IDs, parentage, inputs, outputs,
errors, elapsed times, and model usage. Known secret field names are redacted;
free text can still contain sensitive data. Logging is best effort, not an atomic
database audit log or crash-recovery system.

Evaluation runs the actual workflow with isolated case databases and fixed dates.
It grades intents, selected tool sequences, references, annotated fact patterns,
state transitions, and persisted outcomes. Model-generated verification verdicts
are not treated as independent evaluation truth. See [evaluation details](docs/EVALUATION.md).

## Current boundaries

This snapshot has no real customer/admin authentication, durable draft sessions,
global tool/token budget, or real payment integration. Read-only order lookup is
not yet ownership-scoped. The environment refund limit is not wired into the
hard-coded USD 50 rule. These remain explicit backlog items, not completed safeguards.
