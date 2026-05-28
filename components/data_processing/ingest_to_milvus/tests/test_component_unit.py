"""Unit tests for the ingest_to_milvus component."""

import tempfile
from pathlib import Path
from unittest import mock

from kfp import compiler
from kfp_components.components.data_processing.ingest_to_milvus import ingest_to_milvus

# ---------------------------------------------------------------------------
# KFP compile / signature tests
# ---------------------------------------------------------------------------


def test_component_compiles():
    """Compiler().compile() succeeds and produces a non-empty YAML."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        compiler.Compiler().compile(ingest_to_milvus, f.name)
        assert Path(f.name).stat().st_size > 0


def test_component_signature():
    """All 14 parameters present with expected names."""
    spec = ingest_to_milvus.component_spec
    input_names = set(spec.inputs)
    expected = {
        "s3_endpoint",
        "s3_bucket",
        "milvus_host",
        "s3_prefix",
        "embedding_endpoint",
        "embedding_model",
        "embedding_dim",
        "milvus_port",
        "milvus_db",
        "milvus_token",
        "collection_name",
        "drop_existing",
        "embed_batch_size",
        "milvus_batch_size",
    }
    assert expected == input_names


# ---------------------------------------------------------------------------
# Helpers for logic tests
# ---------------------------------------------------------------------------

JSONL_LINE = '{"source_file": "test.pdf", "chunk_index": 0, "text": "hello world"}\n'


def _make_s3_mock(jsonl_content=JSONL_LINE, keys=None):
    """Return a mocked boto3 S3 client.

    Args:
        jsonl_content: JSONL text returned by get_object.
        keys: List of S3 keys the paginator returns. Defaults to one .jsonl key.
    """
    if keys is None:
        keys = ["chunks/doc1.jsonl"]

    s3_client = mock.MagicMock()

    # Paginator
    pages = [{"Contents": [{"Key": k} for k in keys]}] if keys else []
    paginator = mock.MagicMock()
    paginator.paginate.return_value = pages
    s3_client.get_paginator.return_value = paginator

    # get_object
    body = mock.MagicMock()
    body.read.return_value = jsonl_content.encode("utf-8")
    s3_client.get_object.return_value = {"Body": body}

    return s3_client


def _make_milvus_mock(has_collection=False, existing_dim=None):
    """Return a mocked MilvusClient instance.

    Args:
        has_collection: Value returned by has_collection().
        existing_dim: If set, describe_collection returns this dim for the embedding field.
    """
    client = mock.MagicMock()
    client.has_collection.return_value = has_collection

    if existing_dim is not None:
        client.describe_collection.return_value = {
            "fields": [
                {"name": "embedding", "params": {"dim": existing_dim}},
            ]
        }
    else:
        client.describe_collection.return_value = {"fields": []}

    # insert returns a dict; we don't inspect it.
    client.insert.return_value = {"insert_count": 1}
    # prepare_index_params returns a mock that supports .add_index()
    client.prepare_index_params.return_value = mock.MagicMock()

    return client


def _run_component(
    s3_mock=None,
    milvus_mock=None,
    embedding_endpoint="http://embed:8080",
    embedding_model="test-model",
    embedding_dim=4,
    milvus_token="",
    drop_existing=True,
    milvus_batch_size=256,
    embed_batch_size=64,
    collection_name="test_col",
    extra_env=None,
):
    """Invoke ingest_to_milvus.python_func with all external deps mocked.

    Returns:
        (result, mocks_dict) where mocks_dict contains the mock objects for assertions.
    """
    if s3_mock is None:
        s3_mock = _make_s3_mock()
    if milvus_mock is None:
        milvus_mock = _make_milvus_mock(has_collection=False)

    env = {"S3_ACCESS_KEY": "key", "S3_SECRET_KEY": "secret"}
    if extra_env:
        env.update(extra_env)

    mock_boto3 = mock.MagicMock()
    mock_boto3.client.return_value = s3_mock

    mock_milvus_client_cls = mock.MagicMock(return_value=milvus_mock)
    mock_collection_schema = mock.MagicMock()
    mock_field_schema = mock.MagicMock()
    mock_data_type = mock.MagicMock()

    mock_req_lib = mock.MagicMock()
    embed_response = mock.MagicMock()
    embed_response.json.return_value = {
        "data": [{"index": 0, "embedding": [0.1] * embedding_dim}],
    }
    embed_response.raise_for_status = mock.MagicMock()
    mock_req_lib.post.return_value = embed_response

    mock_st_module = mock.MagicMock()
    mock_local_model = mock.MagicMock()
    mock_local_model.encode.return_value = mock.MagicMock(tolist=lambda: [[0.1] * embedding_dim])
    mock_st_module.SentenceTransformer.return_value = mock_local_model

    patches = {
        "boto3": mock_boto3,
        "requests": mock_req_lib,
        "pymilvus": mock.MagicMock(
            MilvusClient=mock_milvus_client_cls,
            CollectionSchema=mock_collection_schema,
            FieldSchema=mock_field_schema,
            DataType=mock_data_type,
        ),
        "sentence_transformers": mock_st_module,
    }

    with mock.patch.dict("os.environ", env, clear=False), mock.patch.dict("sys.modules", patches):
        result = ingest_to_milvus.python_func(
            s3_endpoint="http://s3:9000",
            s3_bucket="test-bucket",
            milvus_host="milvus",
            s3_prefix="chunks",
            embedding_endpoint=embedding_endpoint,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            milvus_port=19530,
            milvus_db="default",
            milvus_token=milvus_token,
            collection_name=collection_name,
            drop_existing=drop_existing,
            embed_batch_size=embed_batch_size,
            milvus_batch_size=milvus_batch_size,
        )

    return result, {
        "boto3": mock_boto3,
        "s3": s3_mock,
        "milvus_client_cls": mock_milvus_client_cls,
        "milvus": milvus_mock,
        "req_lib": mock_req_lib,
        "st_module": mock_st_module,
        "local_model": mock_local_model,
        "collection_schema": mock_collection_schema,
        "field_schema": mock_field_schema,
        "data_type": mock_data_type,
    }


# ---------------------------------------------------------------------------
# Logic tests
# ---------------------------------------------------------------------------


def test_creates_collection_when_none_exists():
    """When has_collection returns False, create_collection and create_index are called."""
    milvus = _make_milvus_mock(has_collection=False)
    result, mocks = _run_component(milvus_mock=milvus)

    milvus.create_collection.assert_called_once()
    milvus.create_index.assert_called_once()
    assert "test_col" in result


def test_drops_existing_when_drop_existing_true():
    """When has_collection returns True and drop_existing=True, drop_collection is called."""
    milvus = _make_milvus_mock(has_collection=True)
    result, mocks = _run_component(milvus_mock=milvus, drop_existing=True)

    milvus.drop_collection.assert_called_once_with("test_col")
    milvus.create_collection.assert_called_once()


def test_appends_when_drop_existing_false():
    """When has_collection returns True and drop_existing=False, no drop and no create."""
    milvus = _make_milvus_mock(has_collection=True, existing_dim=4)
    result, mocks = _run_component(milvus_mock=milvus, drop_existing=False, embedding_dim=4)

    milvus.drop_collection.assert_not_called()
    milvus.create_collection.assert_not_called()


def test_dimension_mismatch_raises():
    """When existing dim != embedding_dim and drop_existing=False, ValueError is raised."""
    milvus = _make_milvus_mock(has_collection=True, existing_dim=128)
    import pytest

    with pytest.raises(ValueError, match="dim=128"):
        _run_component(milvus_mock=milvus, drop_existing=False, embedding_dim=4)


def test_embedding_via_endpoint():
    """When embedding_endpoint is set, requests.post is called."""
    result, mocks = _run_component(embedding_endpoint="http://embed:8080")

    mocks["req_lib"].post.assert_called()
    mocks["st_module"].SentenceTransformer.assert_not_called()


def test_embedding_via_local_model():
    """When embedding_endpoint is empty, SentenceTransformer is used."""
    result, mocks = _run_component(embedding_endpoint="")

    mocks["st_module"].SentenceTransformer.assert_called_once()
    mocks["local_model"].encode.assert_called()
    mocks["req_lib"].post.assert_not_called()


def test_milvus_token_passed():
    """When milvus_token is non-empty, MilvusClient receives token kwarg."""
    _, mocks = _run_component(milvus_token="mytoken")

    call_kwargs = mocks["milvus_client_cls"].call_args[1]
    assert call_kwargs["token"] == "mytoken"


def test_milvus_token_not_passed_when_empty():
    """When milvus_token is empty, MilvusClient is called WITHOUT token kwarg."""
    _, mocks = _run_component(milvus_token="")

    call_kwargs = mocks["milvus_client_cls"].call_args[1]
    assert "token" not in call_kwargs


def test_empty_s3_raises():
    """When paginator returns no .jsonl files, FileNotFoundError is raised."""
    s3 = _make_s3_mock(keys=[])
    import pytest

    with pytest.raises(FileNotFoundError, match="No chunks found"):
        _run_component(s3_mock=s3)


def test_returns_collection_and_count():
    """Component returns '{collection_name}:{N}'."""
    result, _ = _run_component(collection_name="my_docs")

    assert result == "my_docs:1"
