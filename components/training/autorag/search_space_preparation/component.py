from typing import List, Optional

from kfp import dsl
from kfp.compiler import Compiler
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
)
def search_space_preparation(
    test_data: dsl.Input[dsl.Artifact],
    extracted_text: dsl.Input[dsl.Artifact],
    search_space_prep_report: dsl.Output[dsl.Artifact],
    embedding_models: Optional[List] = None,
    generation_models: Optional[List] = None,
    metric: str = None,
):
    """Runs an AutoRAG experiment's first phase which includes:

        - AutoRAG search space creation given the user's constraints,
        - embedding and foundation models number limitation and initial selection,

    Generates a .yml-formatted report including results of this experiment's phase.
    For its exact content please refer to the `search_space_prep_report_schema.yml` file.

    Args:
        test_data: A path to a .json file containing questions and expected answers that can be retrieved
            from input documents. Necessary baseline for calculating quality metrics of RAG pipeline.

        extracted_text: A path to either a single file or a folder of files. The document(s) will be sampled
            and used during the models selection process.

        search_space_prep_report: kfp-enforced argument specifying an output artifact.
            Provided by kfp backend automatically.

        embedding_models: List of embedding model identifiers to try out in the experiment process.
            This list, if too long, will undergo models preselection (limiting).

        generation_models: List of generation model identifiers to try out in the experiment process.
            This list, if too long, will undergo models preselection (limiting).

        metric: Quality metric to evaluate the intermediate RAG patterns.
    """
    # ChromaDB (via ai4rag) requires sqlite3 >= 3.35; RHEL9 base image has older sqlite.
    # Patch stdlib sqlite3 with pysqlite3-binary before any ai4rag import.
    import sys

    try:
        import pysqlite3

        sys.modules["sqlite3"] = pysqlite3
    except ImportError:
        pass

    import logging
    import os
    import ssl
    from dataclasses import fields, is_dataclass
    from pathlib import Path

    import httpx
    import pandas as pd
    import yaml as yml
    from ai4rag.core.experiment.benchmark_data import BenchmarkData
    from ai4rag.core.experiment.mps import ModelsPreSelector
    from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
    from ai4rag.rag.foundation_models.base_model import BaseFoundationModel
    from ai4rag.search_space.prepare.prepare_search_space import (
        prepare_search_space_with_ogx,
    )
    from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
    from langchain_core.documents import Document
    from ogx_client import APIConnectionError as OGXAPIConnectionError
    from ogx_client import OgxClient

    _ssl_logger = logging.getLogger(__name__)

    def _is_ssl_error(exc: BaseException) -> bool:
        """Check whether an exception (or its cause/context chain) is an SSL verification failure."""
        seen = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            msg = str(current).upper()
            if "CERTIFICATE_VERIFY_FAILED" in msg or "SSL" in msg:
                return True
            current = current.__cause__ or current.__context__
        return False

    def _create_ogx_client(**kwargs) -> OgxClient:
        """Create OgxClient, falling back to SSL-unverified if self-signed cert detected."""
        client = OgxClient(**kwargs)
        try:
            client.models.list()
        except (ssl.SSLCertVerificationError, httpx.ConnectError, OGXAPIConnectionError) as exc:
            if _is_ssl_error(exc):
                _ssl_logger.warning("SSL verification failed for OgxClient — retrying with verify=False. ")
                client = OgxClient(
                    **kwargs,
                    http_client=httpx.Client(verify=False),
                )
            else:
                raise
        return client

    TOP_N_GENERATION_MODELS = 3
    TOP_K_EMBEDDING_MODELS = 2
    METRIC = "faithfulness"
    SAMPLE_SIZE = 5
    SEED = 17

    supported_metrics = ["faithfulness", "answer_correctness", "context_correctness"]

    if embedding_models:
        if not isinstance(embedding_models, list):
            raise TypeError("embedding_models must be a list.")
        for i, m in enumerate(embedding_models):
            if not m:
                raise TypeError(f"embedding_models[{i}] must be a non-empty string.")

    if generation_models:
        if not isinstance(generation_models, list):
            raise TypeError("generation_models must be a list.")
        for i, m in enumerate(generation_models):
            if not m:
                raise TypeError(f"generation_models[{i}] must be a non-empty string.")

    if metric and metric not in supported_metrics:
        raise ValueError(f"Metric {metric} is not supported. Supported metrics are {supported_metrics}.")

    def load_as_langchain_doc(path: str | Path) -> list[Document]:
        """Given path to a text-based file or a folder thereof load everything to memory.

        Args:
            path: str | Path
                A local path to either a text file or a folder of text files.

        Returns":

        list[Document]
            A list of langchain `Document` objects.
        """
        if isinstance(path, str):
            path = Path(path)

        documents = []
        if path.is_dir():
            for doc_path in path.iterdir():
                with doc_path.open("r", encoding="utf-8") as doc:
                    documents.append(
                        Document(
                            page_content=doc.read(),
                            metadata={"document_id": doc_path.stem},
                        )
                    )

        elif path.is_file():
            with path.open("r", encoding="utf-8") as doc:
                documents.append(Document(page_content=doc.read(), metadata={"document_id": path.stem}))

        return documents

    def prepare_ai4rag_search_space() -> AI4RAGSearchSpace:
        """Prepares search space for AI4RAG experiment.

        Returns:
            AI4RAGSearchSpace
                Search space for AI4RAG experiment.
        """
        payload = {}
        if generation_models:
            payload["foundation_models"] = [{"model_id": gm} for gm in generation_models]
        if embedding_models:
            payload["embedding_models"] = [{"model_id": gm} for gm in embedding_models]

        return prepare_search_space_with_ogx(payload, client=client)

    def represent_model_instance(dumper, model: BaseFoundationModel | BaseEmbeddingModel) -> yml.Node:
        """Helper method instructing the yml.Dumper on how to serialize the *Model instances"""
        if isinstance(model, BaseEmbeddingModel):
            type_ = "embedding"
        elif isinstance(model, BaseFoundationModel):
            type_ = "generation"

        params = model.params
        if is_dataclass(params):  # OGX* model classes hold params as dataclass instances
            params = {
                field.name: getattr(model.params, field.name)
                for field in fields(model.params)
                if getattr(model.params, field.name)
            }
        elif hasattr(params, "model_dump"):  # Pydantic v2 models
            params = params.model_dump(exclude_unset=True)
        elif hasattr(params, "dict"):  # Pydantic v1 models
            params = params.dict(exclude_unset=True)

        return dumper.represent_mapping("!Model", {model.model_id: params or {}, "type_": type_})

    yml.add_multi_representer(BaseFoundationModel, represent_model_instance, Dumper=yml.SafeDumper)
    yml.add_multi_representer(BaseEmbeddingModel, represent_model_instance, Dumper=yml.SafeDumper)

    ogx_client_base_url = os.environ.get("OGX_CLIENT_BASE_URL", None)
    ogx_client_api_key = os.environ.get("OGX_CLIENT_API_KEY", None)

    if not ogx_client_base_url or not ogx_client_api_key:
        raise ValueError("OGX_CLIENT_BASE_URL and OGX_CLIENT_API_KEY environment variables must be set.")

    client = _create_ogx_client(base_url=ogx_client_base_url, api_key=ogx_client_api_key)

    search_space = prepare_ai4rag_search_space()

    benchmark_data = BenchmarkData(pd.read_json(Path(test_data.path)))
    documents = load_as_langchain_doc(extracted_text.path)

    if (
        len(search_space["foundation_model"].values) > TOP_N_GENERATION_MODELS
        or len(search_space["embedding_model"].values) > TOP_K_EMBEDDING_MODELS
    ):
        mps = ModelsPreSelector(
            benchmark_data=benchmark_data.get_random_sample(n_records=SAMPLE_SIZE, random_seed=SEED),
            documents=documents,
            foundation_models=search_space._search_space["foundation_model"].values,
            embedding_models=search_space._search_space["embedding_model"].values,
            metric=metric if metric else METRIC,
        )
        mps.evaluate_patterns()
        selected_models = mps.select_models(
            n_embedding_models=TOP_K_EMBEDDING_MODELS,
            n_foundation_models=TOP_N_GENERATION_MODELS,
        )
        selected_models_names = {
            "foundation_model": selected_models["foundation_models"],
            "embedding_model": selected_models["embedding_models"],
        }

    else:
        selected_models_names = {
            "foundation_model": search_space["foundation_model"].values,
            "embedding_model": search_space["embedding_model"].values,
        }

    verbose_search_space_repr = {
        k: v.all_values()
        for k, v in search_space._search_space.items()
        if k not in ("foundation_model", "embedding_model")
    }
    verbose_search_space_repr |= selected_models_names

    with open(search_space_prep_report.path, "w") as report_file:
        yml.safe_dump(verbose_search_space_repr, report_file)


if __name__ == "__main__":
    Compiler().compile(
        search_space_preparation,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
