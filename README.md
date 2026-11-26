# Airline Support Agent

A tool-using airline support assistant built with Python. It searches a small
flight dataset, retrieves airline-rule passages, and returns source references
so the answer can be checked.

This is a portfolio/demo project, not a booking service. Flight data and rule
documents are illustrative and must not be treated as current airline policy.

## What it demonstrates

- Tool calling for flight search and airline-rule lookup.
- Retrieval with lexical ranking and optional Yandex embeddings.
- Input validation, prompt-injection defenses, and response guardrails.
- MCP server over stdio, suitable for connecting to an MCP client.
- Completion caching, usage/cost monitoring, and a reproducible evaluation
  harness with separate golden and holdout datasets.
- Offline unit tests for core behavior.

## Quick start

Requires Python 3.11 or newer.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -v
```

The tests use fakes and local sample data; they do not require API credentials.

## Run the assistant

Copy `.env.example` to `.env` and set `YANDEX_API_KEY` and
`YANDEX_FOLDER_ID`. Keep `.env` private.

```bash
python -m alisa.chat_demo
python -m alisa.chat_demo --semantic
```

Semantic mode and live model calls require a configured Yandex AI Studio
account. The default retrieval mode works locally.

## MCP server

Run the stdio server from this directory:

```bash
python -m alisa.mcp_server
```

To smoke-test the server locally:

```bash
python check_mcp.py
```

The server exposes flight search and airline-rule search tools. Configure the
MCP client to launch `python -m alisa.mcp_server` with this project directory
as its working directory.

## Evaluation

The evaluation harness can call a live model and may incur API costs:

```bash
python scripts/run_eval.py \
  --dataset alisa/golden/golden_set.json \
  --out reports/local_eval.json
```

The checked-in `reports/first_holdout_summary.json` records a historical
independent run: **15/20 scenarios passed (18/23 turns)**. This is evidence
from the specific evaluated revision noted in that file, not a claim about
every later code revision. The result is intentionally reported as-is; inspect
the failed cases and rerun evaluation after making changes.

## Project layout

```text
alisa/       agent, tools, retrieval, guardrails, MCP server, sample data
tests/       unit and integration-style tests
scripts/     evaluation CLI
reports/     selected, compact evaluation evidence
```

## Configuration

| Variable | Purpose |
| --- | --- |
| `YANDEX_API_KEY` | API credential; never commit it |
| `YANDEX_FOLDER_ID` | Yandex Cloud folder used for model IDs |
| `ALISA_SEMANTIC=1` | Enable embeddings in the MCP server |
| `ALISA_LLM_GUARD=1` | Enable the optional model-based MCP guard |

Without API credentials, offline tests, local data search, and the lexical
retrieval implementation remain usable.
