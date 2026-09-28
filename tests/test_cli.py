import argparse
from unittest.mock import MagicMock


import pandas as pd


from groundedqa import cli




def make_fake_df():
    return pd.DataFrame({
        "Country": ["USA", "Canada"],
        "Total Purchases": [10, 20],
        "Loyalty Score": [2, 3],
    })




def test_cmd_ask_prints_result_and_summary(monkeypatch, capsys):
    fake_client = MagicMock()
    fake_client.tracker.summary.return_value = "Total calls: 1 | ..."
    monkeypatch.setattr(cli, "load_dataset", lambda path: make_fake_df())
    monkeypatch.setattr(cli, "GroqClient", lambda: fake_client)
    monkeypatch.setattr(
        cli, "ask_question",
        lambda question, df, client, temperature, max_tokens: {
            "answer": "42", "source_rows": [0], "confidence": "high",
        },
    )


    args = argparse.Namespace(data="mock_data.csv", question="how many?", temperature=0.0, max_tokens=100)
    cli.cmd_ask(args)


    captured = capsys.readouterr()
    assert "42" in captured.out
    assert "Total calls" in captured.out




def test_cmd_compare_runs_six_times_and_prints_summary(monkeypatch, capsys):
    fake_client = MagicMock()
    fake_client.tracker.summary.return_value = "Total calls: 6 | ..."
    monkeypatch.setattr(cli, "load_dataset", lambda path: make_fake_df())
    monkeypatch.setattr(cli, "GroqClient", lambda: fake_client)


    call_count = {"n": 0}


    def fake_ask(question, df, client, temperature, max_tokens):
        call_count["n"] += 1
        return {"answer": f"answer-{call_count['n']}", "source_rows": [], "confidence": "medium"}


    monkeypatch.setattr(cli, "ask_question", fake_ask)


    args = argparse.Namespace(data="mock_data.csv", question="compare this?", max_tokens=100)
    cli.cmd_compare(args)


    captured = capsys.readouterr()
    assert call_count["n"] == 6  # 2 temperatures x 3 runs each
    assert "Temperature 0" in captured.out
    assert "Temperature 1" in captured.out
    assert "Total calls" in captured.out




def test_load_dataset_reads_csv(tmp_path):
    df = make_fake_df()
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)


    result = cli.load_dataset(str(csv_path))
    assert len(result) == 2
    assert list(result.columns) == list(df.columns)




def test_main_dispatches_to_cmd_ask(monkeypatch):
    monkeypatch.setattr("sys.argv", ["groundedqa", "ask", "how many?"])
    calls = {}
    monkeypatch.setattr(cli, "cmd_ask", lambda args: calls.setdefault("ask", args))
    monkeypatch.setattr(cli, "cmd_compare", lambda args: calls.setdefault("compare", args))


    cli.main()


    assert "ask" in calls
    assert calls["ask"].question == "how many?"




def test_main_dispatches_to_cmd_compare(monkeypatch):
    monkeypatch.setattr("sys.argv", ["groundedqa", "compare", "compare this?"])
    calls = {}
    monkeypatch.setattr(cli, "cmd_ask", lambda args: calls.setdefault("ask", args))
    monkeypatch.setattr(cli, "cmd_compare", lambda args: calls.setdefault("compare", args))


    cli.main()


    assert "compare" in calls
    assert calls["compare"].question == "compare this?"



