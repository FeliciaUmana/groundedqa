import json
from unittest.mock import MagicMock

import pytest

from groundedqa.client import GroqClient
from groundedqa.exceptions import GroqClientError, GroqServerError, GroqRateLimitError


def make_response(status_code, json_data=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text or json.dumps(json_data or {})
    resp.json.return_value = json_data or {}
    return resp


def test_chat_success_records_usage(fake_env):
    session = MagicMock()
    session.post.return_value = make_response(200, {
        "choices": [{"message": {"content": '{"answer": "x", "source_rows": [], "confidence": "high"}'}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    })
    client = GroqClient(session=session)

    result = client.chat([{"role": "user", "content": "hi"}])

    assert result["usage"]["prompt_tokens"] == 10
    assert client.tracker.prompt_tokens == 10
    assert client.tracker.completion_tokens == 5
    assert client.tracker.calls == 1


def test_chat_sends_bearer_token_and_max_tokens(fake_env):
    session = MagicMock()
    session.post.return_value = make_response(200, {
        "choices": [{"message": {"content": "{}"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    })
    client = GroqClient(api_key="my-secret-key", session=session)

    client.chat([{"role": "user", "content": "hi"}], max_tokens=123)

    _, kwargs = session.post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer my-secret-key"
    assert kwargs["json"]["max_tokens"] == 123


def test_chat_401_raises_client_error(fake_env):
    session = MagicMock()
    session.post.return_value = make_response(401, text="Unauthorized")
    client = GroqClient(session=session)

    with pytest.raises(GroqClientError) as exc_info:
        client.chat([{"role": "user", "content": "hi"}])
    assert exc_info.value.status_code == 401


def test_chat_500_raises_server_error(fake_env):
    session = MagicMock()
    session.post.return_value = make_response(500, text="Server error")
    client = GroqClient(session=session)

    with pytest.raises(GroqServerError) as exc_info:
        client.chat([{"role": "user", "content": "hi"}])
    assert exc_info.value.status_code == 500


def test_chat_429_then_success_retries(fake_env, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)  # skip real backoff delay in tests
    session = MagicMock()
    rate_limited = make_response(429, text="Too many requests")
    success = make_response(200, {
        "choices": [{"message": {"content": "{}"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    })
    session.post.side_effect = [rate_limited, success]
    client = GroqClient(session=session)

    result = client.chat([{"role": "user", "content": "hi"}], max_retries=3)

    assert session.post.call_count == 2
    assert result["usage"]["prompt_tokens"] == 1


def test_chat_429_exhausts_retries_then_raises(fake_env, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    session = MagicMock()
    session.post.return_value = make_response(429, text="Too many requests")
    client = GroqClient(session=session)

    with pytest.raises(GroqRateLimitError):
        client.chat([{"role": "user", "content": "hi"}], max_retries=2)

    assert session.post.call_count == 3  # initial attempt + 2 retries


def test_missing_api_key_raises_value_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("groundedqa.client.load_dotenv", lambda *a, **kw: None)
    with pytest.raises(ValueError):
        GroqClient(session=MagicMock())