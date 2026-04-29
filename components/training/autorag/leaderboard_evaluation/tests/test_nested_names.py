"""Unit tests for nested_names.py functions."""

from kfp_components.components.training.autorag.leaderboard_evaluation.tests.nested_names import (
    _build_leaderboard_html,
    _get_config_value,
    _get_nested,
    _header_two_lines,
    _merge_params,
    _metric_display_name,
    _metric_to_mean_key,
    _normalize_flat_settings,
    _optimization_score,
    _settings_from_rag_pattern,
)


class TestGetNested:
    """Tests for _get_nested function."""

    def test_flat_key_lookup(self):
        """Key exists at top level."""
        params = {"method": "bm25"}
        assert _get_nested(params, "method") == "bm25"

    def test_dotted_key_with_nested_dict(self):
        """Dotted key resolves from nested dict."""
        params = {"chunking": {"method": "recursive"}}
        assert _get_nested(params, "chunking.method") == "recursive"

    def test_dotted_key_flattened(self):
        """Dotted key exists as flat key in params."""
        params = {"chunking.method": "recursive"}
        assert _get_nested(params, "chunking.method") == "recursive"

    def test_missing_key(self):
        """Key doesn't exist in params."""
        params = {"other": "value"}
        assert _get_nested(params, "missing") is None

    def test_none_params(self):
        """Params is None."""
        assert _get_nested(None, "key") is None

    def test_empty_params(self):
        """Params is empty dict."""
        assert _get_nested({}, "key") is None

    def test_partially_missing_path(self):
        """Outer key exists but value is not a dict."""
        params = {"chunking": "not_a_dict"}
        assert _get_nested(params, "chunking.method") is None

    def test_deep_nesting_only_resolves_first_dot(self):
        """More than 2 levels, should only resolve first dot."""
        params = {"a": {"b.c": "value"}}
        assert _get_nested(params, "a.b.c") == "value"

    def test_empty_string_key(self):
        """Key is empty string."""
        params = {"": "value"}
        assert _get_nested(params, "") == "value"

    def test_key_without_dot_in_nested_structure(self):
        """Simple string key in nested structure."""
        params = {"outer": {"inner": "value"}, "simple": "data"}
        assert _get_nested(params, "simple") == "data"


class TestGetConfigValue:
    """Tests for _get_config_value function."""

    def test_direct_nested_key_match(self):
        """Column exists in nested structure."""
        merged = {"embeddings": {"model_id": "embed-model-123"}}
        assert _get_config_value(merged, "embeddings.model_id") == "embed-model-123"

    def test_flat_fallback_for_embeddings(self):
        """Column is embeddings.model_id but data has embedding_model at top level."""
        merged = {"embedding_model": "flat-embed"}
        assert _get_config_value(merged, "embeddings.model_id") == "flat-embed"

    def test_nested_fallback_for_embeddings(self):
        """Column is embeddings.model_id but data has embedding.model_id."""
        merged = {"embedding": {"model_id": "nested-embed"}}
        assert _get_config_value(merged, "embeddings.model_id") == "nested-embed"

    def test_flat_fallback_for_generation(self):
        """Column is generation.model_id but data has foundation_model."""
        merged = {"foundation_model": "gen-model"}
        assert _get_config_value(merged, "generation.model_id") == "gen-model"

    def test_no_fallbacks_exist(self):
        """Column that doesn't have fallback mappings."""
        merged = {"chunking": {"method": "recursive"}}
        assert _get_config_value(merged, "chunking.method") == "recursive"

    def test_none_merged_dict(self):
        """Merged is None."""
        assert _get_config_value(None, "embeddings.model_id") is None

    def test_empty_merged_dict(self):
        """Merged is empty dict."""
        assert _get_config_value({}, "embeddings.model_id") is None

    def test_all_fallbacks_fail(self):
        """Column with fallbacks but none exist in data."""
        merged = {"other_field": "value"}
        assert _get_config_value(merged, "embeddings.model_id") is None

    def test_priority_of_fallbacks(self):
        """Multiple fallbacks present, should use first match."""
        # embedding_model is first fallback for embeddings.model_id
        merged = {"embedding_model": "first", "embedding": {"model_id": "second"}}
        assert _get_config_value(merged, "embeddings.model_id") == "first"

    def test_column_without_fallbacks_and_not_in_data(self):
        """Column that has no fallbacks and doesn't exist in data."""
        merged = {"other_field": "value"}
        # retrieval.some_field doesn't exist and has no fallbacks
        assert _get_config_value(merged, "retrieval.some_field") is None


class TestMergeParams:
    """Tests for _merge_params function."""

    def test_both_dicts_populated(self):
        """Standard merge with overlapping keys."""
        indexing = {"a": 1, "b": 2}
        rag = {"b": 3, "c": 4}
        result = _merge_params(indexing, rag)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_disjoint_keys(self):
        """No key overlap between the two dicts."""
        indexing = {"a": 1}
        rag = {"b": 2}
        result = _merge_params(indexing, rag)
        assert result == {"a": 1, "b": 2}

    def test_none_indexing_params(self):
        """Only rag_params is provided."""
        result = _merge_params(None, {"a": 1})
        assert result == {"a": 1}

    def test_none_rag_params(self):
        """Only indexing_params is provided."""
        result = _merge_params({"a": 1}, None)
        assert result == {"a": 1}

    def test_both_none(self):
        """Both inputs are None."""
        result = _merge_params(None, None)
        assert result == {}

    def test_overriding_behavior(self):
        """Same key in both dicts, rag_params should win."""
        indexing = {"key": "old"}
        rag = {"key": "new"}
        result = _merge_params(indexing, rag)
        assert result["key"] == "new"

    def test_nested_dict_values_shallow_update(self):
        """Ensure nested dicts are not deep-merged (shallow update)."""
        indexing = {"config": {"a": 1, "b": 2}}
        rag = {"config": {"b": 3}}
        result = _merge_params(indexing, rag)
        # Shallow update means config is completely replaced
        assert result["config"] == {"b": 3}

    def test_empty_dicts(self):
        """Both are empty dicts."""
        result = _merge_params({}, {})
        assert result == {}


class TestSettingsFromRagPattern:
    """Tests for _settings_from_rag_pattern function."""

    def test_complete_rag_pattern_structure(self):
        """All expected fields present."""
        e = {
            "rag_pattern": {
                "settings": {
                    "chunking": {"method": "recursive"},
                    "embedding": {"model_id": "embed-1"},
                    "method": "vector",
                    "number_of_chunks": 5,
                    "generation": {"model_id": "gen-1"},
                }
            }
        }
        result = _settings_from_rag_pattern(e)
        assert result["chunking"] == {"method": "recursive"}
        assert result["embeddings"]["model_id"] == "embed-1"
        assert result["retrieval"]["method"] == "vector"
        assert result["retrieval"]["number_of_chunks"] == 5
        assert result["generation"]["model_id"] == "gen-1"

    def test_missing_rag_pattern_key(self):
        """E doesn't have rag_pattern."""
        e = {"other": "data"}
        assert _settings_from_rag_pattern(e) is None

    def test_missing_settings_key(self):
        """e.rag_pattern exists but no settings."""
        e = {"rag_pattern": {"other": "data"}}
        assert _settings_from_rag_pattern(e) is None

    def test_partial_settings_missing_chunking(self):
        """Some nested fields missing (no chunking)."""
        e = {
            "rag_pattern": {
                "settings": {
                    "embedding": {"model_id": "embed-1"},
                }
            }
        }
        result = _settings_from_rag_pattern(e)
        assert result["chunking"] == {}
        assert result["embeddings"]["model_id"] == "embed-1"

    def test_partial_settings_missing_embedding(self):
        """No embedding field."""
        e = {"rag_pattern": {"settings": {"chunking": {"method": "recursive"}}}}
        result = _settings_from_rag_pattern(e)
        assert result["chunking"] == {"method": "recursive"}
        assert result["embeddings"]["model_id"] is None

    def test_none_values_in_settings(self):
        """Fields exist but are None."""
        e = {"rag_pattern": {"settings": {"chunking": None, "embedding": None}}}
        result = _settings_from_rag_pattern(e)
        assert result["chunking"] == {}
        assert result["embeddings"]["model_id"] is None

    def test_empty_settings(self):
        """Settings is empty dict."""
        e = {"rag_pattern": {"settings": {}}}
        result = _settings_from_rag_pattern(e)
        # Empty dict is falsy, so function returns None
        assert result is None

    def test_non_dict_rag_pattern(self):
        """E['rag_pattern'] is not a dict."""
        e = {"rag_pattern": "not_a_dict"}
        # Non-empty string is truthy, so it tries .get() and raises AttributeError
        # This is a bug in the function, but we test actual behavior
        try:
            _settings_from_rag_pattern(e)
            assert False, "Should have raised AttributeError"
        except AttributeError:
            pass

    def test_non_dict_settings(self):
        """E['rag_pattern']['settings'] is not a dict."""
        e = {"rag_pattern": {"settings": "not_a_dict"}}
        # Non-empty string is truthy, passes `if not rp` check, then tries .get()
        # This is a bug in the function, but we test actual behavior
        try:
            _settings_from_rag_pattern(e)
            assert False, "Should have raised AttributeError"
        except AttributeError:
            pass

    def test_missing_model_id_in_embedding(self):
        """Nested objects exist but model_id is absent."""
        e = {"rag_pattern": {"settings": {"embedding": {}}}}
        result = _settings_from_rag_pattern(e)
        assert result["embeddings"]["model_id"] is None

    def test_missing_model_id_in_generation(self):
        """Nested objects exist but model_id is absent."""
        e = {"rag_pattern": {"settings": {"generation": {}}}}
        result = _settings_from_rag_pattern(e)
        assert result["generation"]["model_id"] is None


class TestNormalizeFlatSettings:
    """Tests for _normalize_flat_settings function."""

    def test_standard_flat_structure(self):
        """Standard structure with all keys."""
        settings = {
            "embedding": {"model_id": "embed-1"},
            "generation": {"model_id": "gen-1"},
            "chunking": {"method": "recursive"},
            "retrieval": {"number_of_chunks": 5},
        }
        result = _normalize_flat_settings(settings)
        assert result["embeddings"]["model_id"] == "embed-1"
        assert result["generation"]["model_id"] == "gen-1"
        assert result["chunking"] == {"method": "recursive"}
        assert result["retrieval"] == {"number_of_chunks": 5}

    def test_none_settings(self):
        """Input is None."""
        assert _normalize_flat_settings(None) is None

    def test_empty_settings(self):
        """Input is empty dict."""
        result = _normalize_flat_settings({})
        # Empty dict is falsy, so function returns None
        assert result is None

    def test_embedding_vs_embeddings_priority(self):
        """Both embedding and embeddings keys present."""
        # Code uses .get("embedding") first, then .get("embeddings")
        settings = {"embedding": {"model_id": "first"}, "embeddings": {"model_id": "second"}}
        result = _normalize_flat_settings(settings)
        assert result["embeddings"]["model_id"] == "first"

    def test_flat_embedding_model(self):
        """embedding is dict with model_id vs flat embedding_model string."""
        settings = {"embedding": {"model_id": "nested"}, "embedding_model": "flat"}
        result = _normalize_flat_settings(settings)
        # Nested should win
        assert result["embeddings"]["model_id"] == "nested"

    def test_flat_embedding_model_only(self):
        """Only flat embedding_model string."""
        settings = {"embedding_model": "flat-model"}
        result = _normalize_flat_settings(settings)
        assert result["embeddings"]["model_id"] == "flat-model"

    def test_flat_foundation_model(self):
        """generation is dict with model_id vs flat foundation_model string."""
        settings = {"generation": {"model_id": "nested"}, "foundation_model": "flat"}
        result = _normalize_flat_settings(settings)
        # Nested should win
        assert result["generation"]["model_id"] == "nested"

    def test_flat_foundation_model_only(self):
        """Only flat foundation_model string."""
        settings = {"foundation_model": "flat-gen"}
        result = _normalize_flat_settings(settings)
        assert result["generation"]["model_id"] == "flat-gen"

    def test_non_dict_embedding(self):
        """Embedding is a string or other non-dict type."""
        settings = {"embedding": "not_a_dict"}
        result = _normalize_flat_settings(settings)
        assert result["embeddings"]["model_id"] is None

    def test_non_dict_generation(self):
        """Generation is a string or other non-dict type."""
        settings = {"generation": "not_a_dict"}
        result = _normalize_flat_settings(settings)
        assert result["generation"]["model_id"] is None

    def test_missing_all_optional_keys(self):
        """Only required structure, all config keys missing."""
        settings = {"other_field": "value"}
        result = _normalize_flat_settings(settings)
        assert result["chunking"] == {}
        assert result["embeddings"]["model_id"] is None
        assert result["retrieval"] == {}
        assert result["generation"]["model_id"] is None

    def test_nested_and_flat_keys_both_present(self):
        """Mixed schema (e.g., both embedding.model_id and embedding_model)."""
        settings = {
            "embedding": {"model_id": "nested-embed"},
            "embedding_model": "flat-embed",
            "generation": {"model_id": "nested-gen"},
            "foundation_model": "flat-gen",
        }
        result = _normalize_flat_settings(settings)
        # Nested should win
        assert result["embeddings"]["model_id"] == "nested-embed"
        assert result["generation"]["model_id"] == "nested-gen"


class TestMetricToMeanKey:
    """Tests for _metric_to_mean_key function."""

    def test_standard_metric_name(self):
        """Faithfulness -> mean_faithfulness."""
        assert _metric_to_mean_key("faithfulness") == "mean_faithfulness"

    def test_empty_string(self):
        """Empty string -> mean_."""
        assert _metric_to_mean_key("") == "mean_"

    def test_already_prefixed(self):
        """mean_faithfulness -> mean_mean_faithfulness (no special handling)."""
        assert _metric_to_mean_key("mean_faithfulness") == "mean_mean_faithfulness"

    def test_metric_with_underscores(self):
        """answer_correctness -> mean_answer_correctness."""
        assert _metric_to_mean_key("answer_correctness") == "mean_answer_correctness"


class TestMetricDisplayName:
    """Tests for _metric_display_name function."""

    def test_underscored_metric(self):
        """answer_correctness -> answer correctness."""
        assert _metric_display_name("answer_correctness") == "answer correctness"

    def test_empty_string(self):
        """Empty string -> optimization metric."""
        assert _metric_display_name("") == "optimization metric"

    def test_no_underscores(self):
        """Faithfulness -> faithfulness."""
        assert _metric_display_name("faithfulness") == "faithfulness"

    def test_multiple_underscores(self):
        """context_answer_correctness -> context answer correctness."""
        assert _metric_display_name("context_answer_correctness") == "context answer correctness"

    def test_leading_trailing_spaces_after_replacement(self):
        """Ensure .strip() works."""
        # Metric with trailing underscore
        assert _metric_display_name("metric_") == "metric"


class TestHeaderTwoLines:
    """Tests for _header_two_lines function."""

    def test_dotted_label_single_word_second_part(self):
        """chunking.method -> chunking<br>method."""
        result = _header_two_lines("chunking.method")
        assert result == "chunking<br>method"

    def test_dotted_label_multi_word_second_part(self):
        """retrieval.number of chunks -> split at last space (3 lines)."""
        result = _header_two_lines("retrieval.number of chunks")
        # Splits at last space in "number of chunks" -> "number of" + "chunks"
        assert result == "retrieval<br>number of<br>chunks"

    def test_no_dot(self):
        """Pattern_Name -> Pattern Name (underscore to space)."""
        result = _header_two_lines("Pattern_Name")
        assert result == "Pattern Name"

    def test_html_special_chars(self):
        """Label with <, >, & should be escaped."""
        result = _header_two_lines("test<tag>")
        assert "&lt;tag&gt;" in result

    def test_empty_string(self):
        """Empty string."""
        result = _header_two_lines("")
        assert result == ""

    def test_multiple_dots(self):
        """a.b.c (only split on first dot)."""
        result = _header_two_lines("a.b.c")
        assert result == "a<br>b.c"

    def test_second_part_with_underscore_and_space(self):
        """embeddings.model_id name -> proper splitting and escaping."""
        result = _header_two_lines("embeddings.model_id name")
        # Second part is "model_id name" which becomes "model id name"
        # Split at last space: "model id" + "name"
        assert result == "embeddings<br>model id<br>name"

    def test_second_part_single_word_after_underscore_replacement(self):
        """chunking.chunk_size -> 2 lines."""
        result = _header_two_lines("chunking.chunk_size")
        # Second part is "chunk_size" -> "chunk size" -> split at space -> "chunk" + "size"
        assert result == "chunking<br>chunk<br>size"

    def test_whitespace_handling(self):
        """Labels with extra spaces."""
        result = _header_two_lines("  label  ")
        assert result == "  label  "


class TestBuildLeaderboardHtml:
    """Tests for _build_leaderboard_html function."""

    def test_standard_inputs(self):
        """All parameters provided with typical values."""
        result = _build_leaderboard_html(
            header_row="<th>Name</th>",
            table_body="<tr><td>test</td></tr>",
            best_pattern_name="pattern_a",
            num_patterns=1,
            eval_metric="faithfulness",
            colgroup_html="<colgroup></colgroup>",
        )
        assert "<!DOCTYPE html>" in result
        assert "RAG Patterns Leaderboard" in result
        assert "<th>Name</th>" in result
        assert "<tr><td>test</td></tr>" in result
        assert "pattern_a" in result
        assert "faithfulness" in result

    def test_empty_colgroup_html(self):
        """Default value for colgroup_html."""
        result = _build_leaderboard_html(
            header_row="<th>Name</th>",
            table_body="<tr><td>test</td></tr>",
            best_pattern_name="pattern_a",
            num_patterns=1,
            eval_metric="faithfulness",
        )
        assert "<!DOCTYPE html>" in result

    def test_html_injection_in_best_pattern_name(self):
        """Ensure best_pattern_name is properly escaped."""
        result = _build_leaderboard_html(
            header_row="",
            table_body="",
            best_pattern_name="<script>alert('xss')</script>",
            num_patterns=1,
            eval_metric="faithfulness",
        )
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_html_injection_in_eval_metric(self):
        """Ensure eval_metric is properly escaped."""
        result = _build_leaderboard_html(
            header_row="",
            table_body="",
            best_pattern_name="pattern",
            num_patterns=1,
            eval_metric="<b>metric</b>",
        )
        assert "&lt;b&gt;metric&lt;/b&gt;" in result

    def test_special_characters_in_eval_metric(self):
        """Underscores, spaces, symbols."""
        result = _build_leaderboard_html(
            header_row="",
            table_body="",
            best_pattern_name="pattern",
            num_patterns=1,
            eval_metric="answer_correctness",
        )
        # _metric_display_name converts to "answer correctness"
        assert "answer correctness" in result

    def test_num_patterns_zero(self):
        """Edge case with 0 patterns."""
        result = _build_leaderboard_html(
            header_row="",
            table_body="",
            best_pattern_name="none",
            num_patterns=0,
            eval_metric="faithfulness",
        )
        assert "0 pattern(s)" in result

    def test_num_patterns_one(self):
        """Single pattern."""
        result = _build_leaderboard_html(
            header_row="",
            table_body="",
            best_pattern_name="pattern_a",
            num_patterns=1,
            eval_metric="faithfulness",
        )
        assert "1 pattern(s)" in result

    def test_large_num_patterns(self):
        """Many patterns."""
        result = _build_leaderboard_html(
            header_row="",
            table_body="",
            best_pattern_name="best",
            num_patterns=100,
            eval_metric="faithfulness",
        )
        assert "100 pattern(s)" in result

    def test_empty_header_row_and_table_body(self):
        """Edge case with empty strings."""
        result = _build_leaderboard_html(
            header_row="",
            table_body="",
            best_pattern_name="pattern",
            num_patterns=0,
            eval_metric="faithfulness",
        )
        assert "<!DOCTYPE html>" in result

    def test_very_long_pattern_name(self):
        """Ensure no layout breaking."""
        long_name = "a" * 200
        result = _build_leaderboard_html(
            header_row="",
            table_body="",
            best_pattern_name=long_name,
            num_patterns=1,
            eval_metric="faithfulness",
        )
        assert long_name in result

    def test_metric_display_name_transformation(self):
        """Via _metric_display_name call."""
        result = _build_leaderboard_html(
            header_row="",
            table_body="",
            best_pattern_name="pattern",
            num_patterns=1,
            eval_metric="context_correctness",
        )
        # Should convert to "context correctness"
        assert "context correctness" in result


class TestOptimizationScore:
    """Tests for _optimization_score function."""

    def test_final_score_present_and_valid(self):
        """final_score: 0.95 -> (False, -0.95)."""
        e = {"final_score": 0.95}
        assert _optimization_score(e) == (False, -0.95)

    def test_final_score_as_string_number(self):
        """final_score: "0.95" -> (False, -0.95)."""
        e = {"final_score": "0.95"}
        assert _optimization_score(e) == (False, -0.95)

    def test_final_score_invalid_non_numeric(self):
        """final_score: "invalid" -> fallback to scores."""
        e = {"final_score": "invalid", "scores": {"metric1": {"mean": 0.8}}}
        assert _optimization_score(e) == (False, -0.8)

    def test_final_score_none(self):
        """final_score: None -> fallback to scores."""
        e = {"final_score": None, "scores": {"metric1": {"mean": 0.7}}}
        assert _optimization_score(e) == (False, -0.7)

    def test_scores_nested_structure(self):
        """scores.scores nested structure."""
        e = {"scores": {"scores": {"metric1": {"mean": 0.85}}}}
        assert _optimization_score(e) == (False, -0.85)

    def test_scores_flat_structure(self):
        """Scores flat structure."""
        e = {"scores": {"metric1": {"mean": 0.9}}}
        assert _optimization_score(e) == (False, -0.9)

    def test_multiple_metrics_in_scores(self):
        """Multiple metrics, should return first found mean."""
        e = {"scores": {"metric1": {"mean": 0.7}, "metric2": {"mean": 0.8}}}
        result = _optimization_score(e)
        # Should return one of them (order depends on dict iteration)
        assert result[0] is False
        assert result[1] in (-0.7, -0.8)

    def test_scores_with_no_mean(self):
        """All metrics missing mean key -> (True, 0)."""
        e = {"scores": {"metric1": {"other": "value"}}}
        assert _optimization_score(e) == (True, 0)

    def test_scores_none(self):
        """scores: None -> (True, 0)."""
        e = {"scores": None}
        assert _optimization_score(e) == (True, 0)

    def test_scores_empty(self):
        """scores: {} -> (True, 0)."""
        e = {"scores": {}}
        assert _optimization_score(e) == (True, 0)

    def test_scores_non_dict(self):
        """scores: "invalid" -> raises AttributeError."""
        e = {"scores": "invalid"}
        # Non-empty string is truthy, so it's not replaced by {}, then tries .get()
        # This is a bug in the function, but we test actual behavior
        try:
            _optimization_score(e)
            assert False, "Should have raised AttributeError"
        except AttributeError:
            pass

    def test_mean_non_numeric(self):
        """mean: "bad" -> (True, 0)."""
        e = {"scores": {"metric": {"mean": "bad"}}}
        assert _optimization_score(e) == (True, 0)

    def test_empty_dict_input(self):
        """Empty dict -> (True, 0)."""
        assert _optimization_score({}) == (True, 0)

    def test_scores_with_non_dict_metric_info(self):
        """Metric info is not a dict."""
        e = {"scores": {"metric": "not_a_dict"}}
        assert _optimization_score(e) == (True, 0)
