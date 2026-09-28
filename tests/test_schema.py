from groundedqa.context import filter_relevant_rows, rows_to_text


def test_filters_by_country_mentioned_in_question(sample_df):
    result = filter_relevant_rows(sample_df, "What is the average purchases in Canada?")
    assert len(result) > 0
    assert (result["Country"] == "Canada").all()


def test_filters_by_loyalty_score_mentioned_in_question(sample_df):
    result = filter_relevant_rows(sample_df, "Who has loyalty score 1?")
    assert len(result) > 0
    assert (result["Loyalty Score"] == 1).all()


def test_falls_back_to_bounded_sample_when_nothing_matches(sample_df):
    result = filter_relevant_rows(sample_df, "What is the capital of France?")
    assert len(result) > 0
    assert len(result) <= len(sample_df)


def test_respects_max_rows(sample_df):
    result = filter_relevant_rows(sample_df, "Tell me everything", max_rows=2)
    assert len(result) <= 2


def test_rows_to_text_includes_row_index_and_fields(sample_df):
    text = rows_to_text(sample_df.head(1))
    assert "Row 0" in text
    assert "Customer ID=1" in text
    assert "Country=USA" in text

import pytest

from groundedqa.schema import parse_and_validate, InvalidResponseError


def test_valid_response_parses():
    raw = '{"answer": "42", "source_rows": [1, 2], "confidence": "high"}'
    result = parse_and_validate(raw)
    assert result["answer"] == "42"
    assert result["source_rows"] == [1, 2]
    assert result["confidence"] == "high"


def test_not_in_data_is_a_valid_answer():
    raw = '{"answer": "NOT_IN_DATA", "source_rows": [], "confidence": "low"}'
    result = parse_and_validate(raw)
    assert result["answer"] == "NOT_IN_DATA"


def test_non_json_raises():
    with pytest.raises(InvalidResponseError):
        parse_and_validate("this is not json at all")


def test_missing_field_raises():
    raw = '{"answer": "42", "confidence": "high"}'  # source_rows missing
    with pytest.raises(InvalidResponseError):
        parse_and_validate(raw)


def test_wrong_type_raises():
    raw = '{"answer": "42", "source_rows": "not a list", "confidence": "high"}'
    with pytest.raises(InvalidResponseError):
        parse_and_validate(raw)


def test_invalid_confidence_value_raises():
    raw = '{"answer": "42", "source_rows": [], "confidence": "extremely-high"}'
    with pytest.raises(InvalidResponseError):
        parse_and_validate(raw)


def test_non_object_json_raises():
    raw = '[1, 2, 3]'  # valid JSON, but not an object
    with pytest.raises(InvalidResponseError):
        parse_and_validate(raw)
def test_rows_to_text_empty_df_returns_empty_string(sample_df):
    empty = sample_df.iloc[0:0]
    assert rows_to_text(empty) == ""