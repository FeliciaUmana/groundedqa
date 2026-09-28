# groundedqa

A small Python package that answers questions using **only** a dataset you
supply, via the Groq chat completions REST endpoint (called directly with
`requests` — no Groq SDK). If the answer isn't in the data, the model is
instructed to reply `NOT_IN_DATA` instead of guessing.

Built for the Data Epic Cohort 5 "Grounded LLM Assistant" task, using the
`mock_data.csv` customer dataset from the segmentation task as the grounding
data.

## Project layout

```
groundedqa/
├── README.md
├── REPORT.md                  # required summary report
├── .env.example                # copy to .env and add your real key
├── .gitignore
├── pyproject.toml              # Poetry config + CLI entry point
├── groundedqa/
│   ├── client.py                # GroqClient — the REST wrapper
│   ├── prompts.py                # all prompt templates, in one place
│   ├── context.py                # picks relevant rows instead of the whole CSV
│   ├── schema.py                  # parses + validates the model's JSON output
│   ├── qa.py                       # ties context + prompts + client + schema together
│   ├── token_tracker.py            # running prompt/completion token totals
│   ├── exceptions.py               # GroqClientError / GroqServerError / GroqRateLimitError
│   └── cli.py                       # `ask` and `compare` commands
└── tests/                       # pytest suite — mocks every HTTP call, never hits Groq
```

## Install

```bash
poetry install
```

(If you're not using Poetry, `pip install requests python-dotenv pandas pytest pytest-cov` works too — the package just won't have the `groundedqa` CLI script installed globally; run it as `python -m groundedqa.cli` instead.)

## Set up your API key

1. Get a free key at [console.groq.com](https://console.groq.com).
2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` and paste your real key in place of the placeholder.
   **Never commit `.env`** — it's already in `.gitignore`.

## Get the dataset

A tiny 10-row sample is included at `data/mock_data.csv` so you can smoke-test
the CLI's plumbing immediately. For the real assignment, replace it with the
full `mock_data.csv` customer dataset from the segmentation task (or your
Week 6 package's data), either by overwriting `data/mock_data.csv` or by
pointing at it directly with `--data path/to/your.csv`.

## Usage

Ask a single question:

```bash
poetry run groundedqa ask "What is the average total purchases for customers in Canada?"
```

Compare temperature 0 vs temperature 1 (3 runs each) for the same question:

```bash
poetry run groundedqa compare "What is the most common loyalty score for USA customers?"
```

Both commands accept `--data` (defaults to `mock_data.csv`) and `--max-tokens`
(defaults to 512). `ask` additionally accepts `--temperature` (defaults to 0.0).

Without Poetry, replace `poetry run groundedqa` with `python -m groundedqa.cli`.

## Running the tests

Tests never call the real Groq API — every HTTP call is mocked with
`unittest.mock`, so the suite runs instantly and needs no API key or network
access.

```bash
pytest tests/ -v --cov=groundedqa --cov-report=term-missing
```

### Coverage output

```
Name                          Stmts   Miss  Cover   Missing
-----------------------------------------------------------
groundedqa/__init__.py            0      0   100%
groundedqa/cli.py                39      1    97%   66
groundedqa/client.py             40      1    98%   96
groundedqa/context.py            24      0   100%
groundedqa/exceptions.py          8      0   100%
groundedqa/prompts.py             7      0   100%
groundedqa/qa.py                 15      0   100%
groundedqa/schema.py             19      0   100%
groundedqa/token_tracker.py      14      0   100%
-----------------------------------------------------------
TOTAL                           166      2    99%
============================== 35 passed in 0.61s ==============================
```

99% coverage, well above the 80% minimum. The two uncovered lines are the
`if __name__ == "__main__":` guard in `cli.py` and one unreachable fallback
branch in `client.py` (a response status code outside 2xx/4xx/5xx, which
`requests` itself would already have raised on).

Test cases covered, per the task's checklist:
- ✅ Successful JSON response (`test_chat_success_records_usage`, `test_ask_returns_parsed_answer_for_valid_json`)
- ✅ Non-JSON response, handled without crashing (`test_ask_handles_non_json_response_without_crashing`)
- ✅ 429 followed by success (`test_chat_429_then_success_retries`)
- ✅ 401 (`test_chat_401_raises_client_error`)
- ✅ NOT_IN_DATA path (`test_ask_not_in_data_path`)
- ✅ Every public method has both a success and a failure path tested (client, schema, context, qa, cli)

## Report

See [REPORT.md](./REPORT.md) for prompt design rationale, the temperature
comparison output, a case where the model was confidently wrong, token
totals, and cost-reduction notes.

## Rules followed

- No API key in code or committed to git — loaded from `.env` via `python-dotenv`.
- `.env` is git-ignored; `.env.example` is committed instead.
- `max_tokens` is set on every single request (`client.py`'s `chat()` always includes it in the payload; it's a required parameter with a default, never omitted).
- No private customer data is logged — only token counts and the model's own answers are printed/logged, never full row dumps beyond what's shown to the model in-session.
