"""
All prompt templates live in this one module so they're easy to read,
diff and review in one place, rather than scattered through the code
that calls the API.
"""


SYSTEM_PROMPT_TEMPLATE = """You are a data assistant that answers questions using ONLY the customer \
data provided to you in the user message below. You must not use outside knowledge, general \
world facts, or assumptions not present in the data.


Rules:
- Answer strictly from the labelled data rows provided below the instructions.
- If the answer cannot be determined from the data provided, respond with exactly: NOT_IN_DATA
- Keep the "answer" field concise (1-3 sentences).
- Respond with a single JSON object and nothing else — no preamble, no markdown fences, \
no explanation outside the JSON. The object must have exactly these fields:
  {{"answer": string, "source_rows": [list of row index integers you used, or an empty list], \
"confidence": one of "low", "medium", "high"}}
"""


def build_system_prompt():
  """The system message: role + output-format + grounding constraints."""
  return  SYSTEM_PROMPT_TEMPLATE


def build_user_prompt(question: str, data_rows_text: str) -> str:
    """
    The user message, ordered as the task requires:
    instructions -> labelled data -> question.
    """
    return (
        "Instructions: Use only the data below to answer the question. "
        "If the question cannot be answered from this data, reply NOT_IN_DATA.\n\n"
        f"Data:\n{data_rows_text}\n\n"
        f"Question: {question}"
    )
    

def build_messages(question:str, data_rows_text: str) -> list:
  """Assemble the full messages list (system + user) for one API call."""
  return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(question, data_rows_text)},
    ]
 