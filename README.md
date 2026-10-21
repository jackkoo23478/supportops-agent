# SupportOps Agent

A customer support agent that looks up policies and orders, handles complaints and creates tickets. Refunds above a limit need human approval. Runs on local models (Ollama).

## Status

Work in progress. Project skeleton only.

## Setup

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -e ".[dev]"
    pytest