from unittest.mock import MagicMock


from groundedqa.qa import ask
from groundedqa.client import GroqClient




def make_client_with_response(content):
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5},
    }
    session.post.return_value = resp
    return GroqClient(api_key="test-key", session=session)




def test_ask_returns_parsed_answer_for_valid_json(sample_df):
    client = make_client_with_response(
        '{"answer": "3 customers", "source_rows": [0, 1], "confidence": "high"}'
    )
    result = ask("How many customers are there?", sample_df, client)
    assert result["answer"] == "3 customers"
    assert result["source_rows"] == [0, 1]
    assert result["confidence"] == "high"




def test_ask_not_in_data_path(sample_df):
    client = make_client_with_response(
        '{"answer": "NOT_IN_DATA", "source_rows": [], "confidence": "low"}'
    )
    result = ask("What is the weather today?", sample_df, client)
    assert result["answer"] == "NOT_IN_DATA"




def test_ask_handles_non_json_response_without_crashing(sample_df):
    client = make_client_with_response("I think the answer is probably 42.")
    result = ask("How many customers?", sample_df, client)
    assert result["answer"] == "NOT_IN_DATA"
    assert "error" in result
    assert "raw" in result




def test_ask_passes_temperature_and_max_tokens_through(sample_df):
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": "{}"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    session.post.return_value = resp
    client = GroqClient(api_key="test-key", session=session)


    ask("test question", sample_df, client, temperature=0.7, max_tokens=99)


    _, kwargs = session.post.call_args
    assert kwargs["json"]["temperature"] == 0.7
    assert kwargs["json"]["max_tokens"] == 99



