import pandas as pd
import pytest




@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Customer ID": [1, 2, 3, 4],
        "Country": ["USA", "Canada", "USA", "Mexico"],
        "Total Purchases": [25, 15, 8, 3],
        "Loyalty Score": [3, 2, 1, 1],
    })




@pytest.fixture
def fake_env(monkeypatch):
    """Ensures a GROQ_API_KEY is present in the environment for tests
    that construct a GroqClient without passing api_key= explicitly."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key-123")
