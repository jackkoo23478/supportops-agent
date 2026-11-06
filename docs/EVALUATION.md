# Evaluation

The latest recorded Ollama run used `qwen2.5:7b` on November 6, 2026.
It passed **23 of 28 cases** and **54 of 59 conversation steps**. The deterministic
test suite separately passed **362 tests** during PR preparation. Unit tests do
not establish real-model answer quality.

The [sanitized baseline](evaluation/baseline-2026-11-06.json) contains run metadata,
source and dataset hashes, aggregate metrics, and case outcomes. Runtime databases,
raw traces, local paths, and conversation transcripts are excluded.

## Recorded results

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
