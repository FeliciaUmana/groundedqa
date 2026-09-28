"""
Context management: instead of dumping the whole CSV into every prompt
(expensive and wasteful in tokens), pick out only the rows that are
plausibly relevant to the question being asked.
"""

import re

import pandas as pd


def filter_relevant_rows(df: pd.DataFrame, question: str, max_rows: int = 20) -> pd.DataFrame:
    """
    Select a subset of rows relevant to `question`.

    Strategy: look for a country name (from the data itself, not a
    hardcoded list) or a loyalty score number mentioned in the
    question text, and filter down to matching rows. If nothing in
    the question matches anything filterable, fall back to a small
    head-of-table sample rather than sending everything.
    """
    q = question.lower()
    filtered = df

    if "Country" in df.columns:
        matched_country = next(
            (c for c in df["Country"].dropna().unique() if str(c).lower() in q),
            None,
        )
        if matched_country:
            filtered = filtered[filtered["Country"] == matched_country]

    if "Loyalty Score" in df.columns:
        score_match = re.search(r"loyalty score (?:of )?(\d)", q)
        if score_match:
            score = int(score_match.group(1))
            filtered = filtered[filtered["Loyalty Score"] == score]

    if filtered.empty or len(filtered) == len(df):
        filtered = df.head(max_rows)
    else:
        filtered = filtered.head(max_rows)

    return filtered


def rows_to_text(df: pd.DataFrame) -> str:
    """
    Render a (small, already-filtered) DataFrame as compact labelled
    text for the prompt, one line per row, each field name=value, with
    the row's original index included so the model can cite source_rows.
    """
    lines = []
    for idx, row in df.iterrows():
        fields = ", ".join(f"{col}={row[col]}" for col in df.columns)
        lines.append(f"Row {idx}: {fields}")
    return "\n".join(lines)