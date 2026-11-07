# Evaluation

## Current results (November 7, 2026)

The policy workflow now renders reviewed return-window and refund-processing rules
when the question and retrieved source match. The latest Qwen2.5 7B run passed
**24/28 cases**, **55/59 steps**, and **59/59 persisted safety checks**. The return
deadline and refund-processing start cases both passed with one model call each
for intent classification. The four remaining failures were two original-packaging
questions, the USD 50.00 approval explanation, and the alternative refund account
question. The USD 50.00 and USD 50.01 action cases both passed. Ruff and all 390
pytest tests passed.

## Model and Verify comparison

Before the two reviewed policy rules were added, the same 28 cases with Verify
enabled passed 14/28 with Qwen2.5 3B and 23/28 with Qwen2.5 7B. Both models passed
59/59 persisted safety checks. Two 3B intent errors sent policy questions to the
refund workflow while a refund draft was active.

The eight focused policy cases were also run in four configurations. The off
configuration retained citation ID checks and skipped claim verification and
rewrites. Each model and the shared retrieval index were warmed before timing.

- 3B with Verify on passed 0/8, with 7 refusals and a 4.90-second median.
- 3B with Verify off passed 4/8, with 0 refusals and a 1.23-second median. Two
  answers directly contradicted the policy.
- 7B with Verify on passed 3/8, with 5 refusals and a 4.71-second median.
- 7B with Verify off passed 7/8, with 1 refusal and a 1.95-second median.

Each figure represents one run on a small policy set. The comparison describes
observed answer quality and latency under these conditions.

The pre-budget baseline Ollama run used `qwen2.5:7b` on November 6, 2026.
It passed **23 of 28 cases** and **54 of 59 conversation steps**. The deterministic
test suite separately passed **362 tests** during PR preparation. Unit tests do
not establish real-model answer quality.

The [sanitized baseline](evaluation/baseline-2026-11-06.json) contains run metadata,
source and dataset hashes, aggregate metrics, and case outcomes. Runtime databases,
raw traces, local paths, and conversation transcripts are excluded.

## Execution-budget regression run (November 7, 2026)

The same 28 cases were rerun with the configured 10 LLM calls / 5 tool calls per
message and role-specific generation caps. Results remained **23/28 cases** and
**54/59 steps**, with the same five failed case IDs. The test suite now passes
**381 tests**, including 19 budget tests. See the
[sanitized budget run](evaluation/budget-2026-11-07.json).

Across 57 workflow messages (the other two steps are administrative reviews):

- No messages hit a call limit or the advisory token threshold.
- Maximum observed usage: 7 LLM calls, 2 public tool calls, 5,948 reported tokens.
- No JSON-validation retries were recorded with these output caps.
- 74 LLM calls; 47,437 prompt tokens; 4,181 completion tokens; no runtime errors.
- Elapsed time was approximately 252 seconds. This is not a controlled latency
  comparison; loading, caching, and hardware load can differ between runs.

Deterministic tests separately force exhaustion to verify that retries count,
HTTP/tool work stops before execution, independent messages reset their budgets,
and a saved refund can be resumed without duplicate execution. Passing the real
suite without exhaustion alone does not establish that blocking works.

## Pre-budget recorded results

- Smoke: 10/10 cases.
- Extended: 9/10 cases.
- Verifier questions: 4/8 cases.
- Intent checks: 33/33 annotated classifications.
- Routing checks: 26/26 annotated deterministic routing decisions.
- Tool checks: 38/38 steps with an expected tool sequence.
- Citation checks: 9/14 annotated required-source checks.
- Fact checks: 7/12 annotated answer checks.
- Persisted safety checks: 59/59 steps checked against the suite's assertions.
- Runtime errors: 0; unrun steps: 0.
- Reported model usage: 74 calls, 47,437 prompt tokens, 4,181 completion tokens.
- Total elapsed time: approximately 639 seconds on the development machine.

Each denominator counts only applicable annotations. Intent and tool accuracy
therefore have different denominators. Tool checks compare observed tool sequences
with expected sequences; Python selects the workflow after classification.
Fact checks use selected expected/forbidden answer patterns. They are not an
exhaustive claim-level hallucination measurement. Safety checks inspect specified
persisted outcomes; they do not cover unauthorized read access or every injection.
Latency depends on hardware, model loading, and workload; this is one run.

## Five failed cases

All five ended with a conservative insufficient-policy-information response:

- `policy_without_packaging`: the verifier rejected correct conditional wording
  about original packaging.
- `verify_packaging_negative`: the draft did not provide an acceptable supported
  answer and was withheld.
- `verify_deadline_purchase_premise`: the semantic verifier rejected a corrected
  answer measuring the return deadline from delivery.
- `verify_processing_start_event`: generation and verification were inconsistent
  about receipt/inspection versus approval as the processing start event.
- `verify_alternative_refund_account`: the policy specifies the original payment
  method but does not explicitly address changing bank details. The expected
  answer and policy coverage need review before treating this as a model-only bug.

The generator and verifier use separate calls to the same Qwen model, so their
errors can be correlated. Refusal avoids publishing some unsupported drafts but
reduces task completion. Remaining limitations are tracked in
[PROJECT_STATUS.md](PROJECT_STATUS.md).

## Reproduce

Install the project and start Ollama as described in [README.md](../README.md).
Embedding and reranker models must already be cached for `--local-files-only`.

```powershell
python -m pytest -q
python -m ruff check src scripts evals tests
python -m evals.run --cases evals/cases/smoke.json --cases evals/cases/extended.json --cases evals/cases/verifier_questions.json --local-files-only
```

The runner writes reports under ignored `data/runtime/evals/`. Each case uses an
isolated database and records tool calls, model decisions, timings, and checks.
The suites specify a fixed date for reproducible refund eligibility. A new run
may differ because real-model generation is not deterministic.

Broader claim-level annotation, repeated trials, held-out questions, adversarial
paraphrases, and cross-customer read tests remain future work. Do not interpret
this small regression set as a general security or production-readiness score.
