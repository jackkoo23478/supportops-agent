# Workflow commands

Run commands from the repository root after installing `.[dev]` into the active
Python environment. The CLI uses the configured Ollama model; standalone ticket
and refund tools do not require model inference.

## Interactive state and commands

```powershell
python -m supportops.cli --customer-id C001 --ticket-db data/runtime/tickets-demo.db --refund-db data/runtime/refunds-demo.db
```

- `exit` / `quit`: leave the CLI.
- `cancel complaint`: discard a pending complaint, not a saved ticket.
- `confirm refund`: submit the collected order and amount; execute if automatically approved.
- `cancel refund`: discard an unsubmitted draft. An unresolved submission stays locked for retry.
- `retry refund`: retry the same draft or continue the latest saved refund.
- `refund status R-...`: read a saved refund for the current demo customer.
- `execute refund R-...`: attempt a saved approved simulation; pending/rejected requests cannot execute.

Refund IDs can be omitted for the most recent refund in the current session.
After restarting, provide the complete ID. Generic `retry` follows the active
workflow or most recent action. Pending draft state does not survive a restart.

## Complaint tool

```powershell
python -m scripts.try_ticket --db data/runtime/ticket-tool-demo.db
```

The fixture creates a damaged-item complaint for C001/O1001. Repeating the command
uses the same demo request ID and returns the same ticket. A genuinely new request
needs a new `--request-id`.

## Refund eligibility without persistence

```powershell
python -m scripts.try_refund
```

This prints decisions for USD 30, 50, and 80 using fixed eligible sample facts.
It does not create requests or payments.

## Persistent request and automatic simulation

```powershell
python -m scripts.try_refund_request --db data/runtime/refund-tool-demo.db --requested-on 2026-11-06
```

The default request is C001/O1001 for USD 30. Copy its `refund_id` and replace
`R-REPLACE-ME` below. Use the same database path:

```powershell
python -m scripts.execute_refund --db data/runtime/refund-tool-demo.db --refund-id R-REPLACE-ME --customer-id C001
```

The result contains `decision.status: executed`, a `SIM-...` transaction ID, and
`simulated: true`. Repeating execution returns the original transaction, not a
second refund. `ok: true` on request creation means saved, not necessarily approved.

## Human approval and rejection

Use a separate database so earlier reservations do not consume the fixture balance:

```powershell
python -m scripts.try_refund_request --db data/runtime/refund-approval-demo.db --amount 59.90 --requested-on 2026-11-06
python -m scripts.review_refund --db data/runtime/refund-approval-demo.db --refund-id R-REPLACE-ME --decision approve --reviewer admin-jack --reason "Return documents checked."
python -m scripts.execute_refund --db data/runtime/refund-approval-demo.db --refund-id R-REPLACE-ME --customer-id C001
```

Substitute the actual saved ID before reviewing/executing. Use `--decision reject`
for rejection. Only pending approvals can be reviewed; identical reviews are
idempotent and conflicting decisions cannot overwrite the original audit.
`--reviewer` is a local audit label, not authenticated administrator identity.
Customer chat does not expose this review operation.

For a request created in chat, review the same `--refund-db` file from a separate
trusted terminal, then use `execute refund R-...` in chat. Approval alone sends no
payment and does not automatically execute a saved request.

## Retrieval and trace inspection

```powershell
python -m scripts.try_search "original packaging" --local-files-only
python -m scripts.try_search "original packaging" --no-rerank --top-k 3
python -m scripts.show_trace --last 5
python -m scripts.show_trace --trace-id REPLACE-ME --details
python -m scripts.show_trace --refund-id R-REPLACE-ME --last 10
```

The search demo loads policies and ranking models. The inspector only reads JSONL;
it never invokes tools or model inference. Use a positional path to inspect a
non-default trace file. Refund request/review/execution scripts also support
`--trace-file PATH` and `--no-trace`.
