# REPORT.md

> **Before submitting:** this file has placeholders in the sections marked
> `TODO` — those require actually running `groundedqa` against the real
> Groq API with your own key, which can't be done ahead of time here. Run
> the commands shown in each section and paste the real output in.

## 1. Prompt templates and why they're structured this way

All templates live in `groundedqa/prompts.py`, in one place, so they're easy
to read and review without hunting through the rest of the code.

**System message** (`build_system_prompt`) sets the model's role ("a data
assistant"), the grounding constraint (answer only from the data provided,
no outside knowledge), the exact fallback phrase to use when the answer
isn't present (`NOT_IN_DATA`), and the required output shape (a single JSON
object with `answer`, `source_rows`, `confidence` — no extra text). Putting
all of this in the system message, rather than the user message, keeps it
stable across every question — it doesn't need to be re-stated or
re-worded per call.

**User message** (`build_user_prompt`) is ordered instructions → labelled
data → question, per the task's requirement. Instructions come first so the
model reads its task before seeing any data (helps it interpret the data
correctly rather than pattern-matching on it blind). The labelled data rows
come next, each tagged with a row index (`Row 0: Country=USA, ...`) so the
model can cite specific `source_rows` in its answer. The question comes
last, immediately before the model has to answer — keeping it close to the
generation point.

## 2. Grounding — three questions the model correctly refuses

TODO — the system prompt instructs the model to reply `NOT_IN_DATA` when
the answer isn't in the supplied data. Run at least three questions that
have no answer in `mock_data.csv` (e.g. asking about a column that doesn't
exist, a country not present in the data, or something entirely unrelated
like general trivia) and paste each question + the model's JSON response
below, showing it refusing correctly:

```bash
poetry run groundedqa ask "TODO: an unanswerable question #1"
poetry run groundedqa ask "TODO: an unanswerable question #2"
poetry run groundedqa ask "TODO: an unanswerable question #3"
```

```
TODO: paste all three question/response pairs here.
```

## 3. Temperature comparison

TODO — run this and paste the real output below:

```bash
poetry run groundedqa compare "TODO: put a real question about mock_data.csv here"
```

```
TODO: paste the full console output here — all 3 runs at temperature 0,
all 3 runs at temperature 1, and the final token summary line.
```

**What I observed:** TODO — e.g., did temperature 0 give identical or
near-identical answers across all 3 runs? Did temperature 1 show more
variation in wording, or in which rows it cited, or even in confidence
level? Note anything surprising.

## 4. A case where the model was confidently wrong

TODO — ask a question where you can verify the correct answer yourself
from the data, get the model to answer with `"confidence": "high"`, and
show a case where it was wrong. Paste the question, the model's JSON
response, and the actual correct answer, then briefly say how you caught
it (e.g. manually checking the filtered rows sent in the prompt, or
noticing the `source_rows` it cited didn't actually support its stated
answer).

## 5. Token totals

TODO — after running a full session (`ask` + `compare` calls), paste the
final line the CLI prints on exit, e.g.:

```
Total calls: 8 | Prompt tokens: 1,240 | Completion tokens: 310 | Total tokens: 1,550
```

**How much context filtering saved:** `context.py`'s `filter_relevant_rows`
narrows the dataset down to only rows matching a country or loyalty score
mentioned in the question (falling back to a small bounded sample —
`max_rows=20` — otherwise), instead of sending the entire CSV on every
call. TODO — state roughly how many tokens this saved, e.g. by comparing
the prompt token count for a filtered call against what the same call
would have cost sending the full dataset (you can estimate the full-dataset
token count with an online tokenizer or `tiktoken`, then compare against
the `usage.prompt_tokens` the API actually reported for the filtered
version).

## 6. What I'd change under a fixed cost budget

TODO — a few concrete ideas to consider and pick from:
- Cache repeated questions (same question + same filtered data → skip the
  API call entirely and return the cached answer).
- Lower `max_tokens` further for questions that only need a short answer.
- Narrow `filter_relevant_rows`' fallback sample size (`max_rows`) even
  further when no country/loyalty-score match is found.
- Batch multiple related questions into a single prompt instead of one
  call per question, where the task allows it.
- Stick with `llama-3.1-8b-instant` (the smallest model) rather than
  upgrading, unless accuracy testing shows it's insufficient.
