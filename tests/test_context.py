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


def test_rows_to_text_empty_df_returns_empty_string(sample_df):
    empty = sample_df.iloc[0:0]
    assert rows_to_text(empty) == ""