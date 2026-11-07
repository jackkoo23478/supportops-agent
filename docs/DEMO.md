# Record a SupportOps demo

Record a three-to-four-minute terminal walkthrough using fictional customer data.
Start Ollama and install the project using the [quick start](../README.md#quick-start).
The recording uses `qwen2.5:7b` and a fresh local database for each take.

## Prepare the terminal

```powershell
$env:LLM_MODEL = 'qwen2.5:7b'
$demoDir = Join-Path 'data/runtime' ('demo-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $demoDir | Out-Null
$demoDb = Join-Path $demoDir 'actions.db'
$demoTrace = Join-Path $demoDir 'trace.jsonl'
python -m supportops.cli --customer-id C001 --ticket-db $demoDb --refund-db $demoDb --trace-file $demoTrace
```

Ask the return-window question once in the same session before recording to load
the embedding and reranking models. Enlarge terminal text so the audience can
read the answers and citations.

## Show three customer requests

```text
What is the status of order O1001?
When was it ordered?
Does the 30-day return window start from the purchase date?
My item arrived damaged. Please open a complaint ticket.
O1001
exit
```

Show that the follow-up retains the order ID, the policy reply cites
`returns.eligibility`, and the complaint produces a ticket ID. If the complaint
already used the remembered order and returned a ticket ID, skip the `O1001` turn.

## Show approval and simulated execution

Use a fixed request date to keep the same sample order eligible on every take.
Run these commands in the original PowerShell terminal after exiting customer chat.

```powershell
$request = python -m scripts.try_refund_request --db $demoDb --trace-file $demoTrace --amount 59.90 --requested-on 2026-11-06 | ConvertFrom-Json
$refundId = $request.data.refund_id
$request.data.decision.status
python -m scripts.execute_refund --db $demoDb --trace-file $demoTrace --refund-id $refundId --customer-id C001
python -m scripts.review_refund --db $demoDb --trace-file $demoTrace --refund-id $refundId --decision approve --reviewer demo-reviewer --reason 'Sample return inspected.'
python -m scripts.execute_refund --db $demoDb --trace-file $demoTrace --refund-id $refundId --customer-id C001
```

Pause at `pending_approval`, show the blocked execution, then show the approved
execution with a `SIM-...` transaction ID. This uses a separate trusted local
review command and a simulated ledger.

## Close with the trace and evaluation

```powershell
python -m scripts.show_trace $demoTrace --last 5
```

Show the policy retrieval or refund sequence and mention the latest 7B run passed
24 of 28 scenarios. Keep the trace, databases, and `.env` outside the recording
uploads. The [evaluation notes](EVALUATION.md) describe the remaining policy failures.
