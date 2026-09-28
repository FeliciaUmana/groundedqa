import os
import time
import requests
from dotenv import load_dotenv

from .exceptions import GroqClientError, GroqRateLimitError, GroqServerError
from .token_tracker import TokenTracker

GROQ_CHAT_COMPLETIONS_URL = 'https://api.groq.com/openai/v1/chat/completions'
DEFAULT_MODEL = "llama-3.1-8b-instant"


class GroqClient:
    """
    Thin wrapper around the Groq chat completions REST endpoint, using
    `requests` directly (no Groq SDK), as required by the task.
    """

    def __init__(self, api_key=None, model=DEFAULT_MODEL, tracker=None, session=None):
        load_dotenv()
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError(
                "No Groq API key found. Set GROQ_API_KEY in a .env file "
                "(see .env.example) or pass api_key= explicitly."
            )
        self.model = model
        self.tracker = tracker or TokenTracker()
        self.session = session or requests.Session()

    def _headers(self):
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

    def chat(self, messages, temperature: float = 0.0, max_tokens: int = 512, max_retries: int = 3):
        """
        Send a chat completion request. Always sets max_tokens (required
        by the task). Retries up to `max_retries` times with exponential
        backoff on 429, then raises GroqRateLimitError. Any other 4xx
        raises GroqClientError; any 5xx raises GroqServerError.
        """
        payload = {
            'model': self.model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
        }

        attempt = 0
        while True:
            response = self.session.post(
                GROQ_CHAT_COMPLETIONS_URL,
                headers=self._headers(),
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                usage = data.get('usage', {})
                self.tracker.record(
                    usage.get('prompt_tokens', 0),
                    usage.get('completion_tokens', 0),
                )
                return data

            if response.status_code == 429:
                attempt += 1
                if attempt > max_retries:
                    raise GroqRateLimitError(
                        "Rate limited by Groq after maximum retries.",
                        status_code=429,
                        response_body=response.text,
                    )
                time.sleep(2 ** attempt)
                continue

            if 400 <= response.status_code < 500:
                raise GroqClientError(
                    f'Groq API client error ({response.status_code}).',
                    status_code=response.status_code,
                    response_body=response.text,
                )

            if 500 <= response.status_code < 600:
                raise GroqServerError(
                    f'Groq API server error ({response.status_code}).',
                    status_code=response.status_code,
                    response_body=response.text,
                )
                
            response.raise_for_status()
            