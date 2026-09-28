import pandas as pd


from .client import GroqClient
from .context import filter_relevant_rows, rows_to_text
from .prompts import build_messages
from .schema import parse_and_validate, InvalidResponseError



def ask(question: str, df: pd.DataFrame, client: GroqClient, temperature: float = 0.0, max_tokens: int = 512) -> dict:
    """
    Answer `question` grounded only in `df`. Returns a dict with
    answer / source_rows / confidence. If the model's response isn't
    valid JSON, this does NOT crash — it degrades to a NOT_IN_DATA-style
    result with the parsing error attached for debugging.
    """
    relevant = filter_relevant_rows(df, question)
    data_text = rows_to_text(relevant)
    messages = build_messages(question, data_text)


    response = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
    raw_answer = response["choices"][0]["message"]["content"]


    try:
        return parse_and_validate(raw_answer)
    except InvalidResponseError as exc:
        return {
            "answer": "NOT_IN_DATA",
            "source_rows": [],
            "confidence": "low",
            "error": str(exc),
            "raw": raw_answer,
        }
