from groundedqa.token_tracker import TokenTracker


def test_records_and_accumulates_totals():
    tracker = TokenTracker()
    tracker.record(10, 5)
    tracker.record(20, 10)

    assert tracker.prompt_tokens == 30
    assert tracker.completion_tokens == 15
    assert tracker.total_tokens == 45
    assert tracker.calls == 2


def test_starts_at_zero():
    tracker = TokenTracker()
    assert tracker.total_tokens == 0
    assert tracker.calls == 0


def test_summary_contains_the_numbers():
    tracker = TokenTracker()
    tracker.record(10, 5)
    summary = tracker.summary()
    assert "10" in summary
    assert "5" in summary
    assert "15" in summary  # total