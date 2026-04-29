"""Shared fixtures for all tests in this directory."""

from unittest.mock import Mock

import pytest


@pytest.fixture
def sample_output_data():
    """Complete output_data structure for testing."""
    return {
        "name": "test_pattern",
        "settings": {
            "generation": {
                "model_id": "gpt-4",
                "system_message_text": "You are helpful",
                "user_message_text": "Answer: {question}",
                "context_template_text": "{document}",
            },
            "embedding": {
                "model_id": "text-embedding-ada-002",
                "embedding_params": {"embedding_dimension": 768},
                "distance_metric": "cosine",
            },
            "vector_store": {
                "datasource_type": "chroma",
                "collection_name": "test_collection",
            },
            "retrieval": {
                "method": "simple",
                "number_of_chunks": 5,
            },
            "chunking": {
                "method": "recursive",
                "chunk_size": 512,
                "chunk_overlap": 50,
            },
        },
    }


@pytest.fixture
def minimal_output_data():
    """Minimal output_data with missing fields."""
    return {"name": "minimal_pattern", "settings": {}}


@pytest.fixture
def mock_evaluation_result():
    """Mock evaluation result object with complete data."""
    result = Mock()
    result.pattern_name = "test_pattern"
    result.execution_time = 123.45
    result.collection = "test_collection"
    result.indexing_params = {
        "chunking": {"method": "recursive", "chunk_size": 512, "chunk_overlap": 100},
        "embedding": {"model_id": "embed-model", "distance_metric": "cosine"},
        "vector_store": {"datasource_type": "chroma"},
    }
    result.rag_params = {
        "retrieval": {"method": "simple", "number_of_chunks": 5},
        "generation": {
            "model_id": "gen-model",
            "context_template_text": "{document}",
            "user_message_text": "Question: {question}",
            "system_message_text": "You are helpful",
        },
    }
    return result


@pytest.fixture
def mock_eval_data():
    """Mock evaluation data for _evaluation_result_fallback tests."""
    data = Mock()
    data.question = "What is AI?"
    data.ground_truths = ["Artificial Intelligence"]
    data.answer = "AI is Artificial Intelligence"
    data.contexts = ["AI stands for Artificial Intelligence"]
    data.context_ids = ["doc1"]
    data.question_id = "q1"
    return data


@pytest.fixture
def sample_notebook_dict():
    """Sample notebook dictionary structure."""
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "# Test Notebook",
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": "print('hello')",
                "execution_count": None,
                "outputs": [],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
