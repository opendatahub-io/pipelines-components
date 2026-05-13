from typing import List, Optional

from kfp import dsl
from kfp.compiler import Compiler
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
    install_kfp_package=False,
)
def search_space_preparation(
    test_data: dsl.Input[dsl.Artifact],
    extracted_text: dsl.Input[dsl.Artifact],
    search_space_prep_report: dsl.Output[dsl.Artifact],
    chat_model_url: Optional[str] = None,
    chat_model_token: Optional[str] = None,
    embedding_model_url: Optional[str] = None,
    embedding_model_token: Optional[str] = None,
    embeddings_models: Optional[List] = None,
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

        chat_model_url: Base URL for the chat/generation model API.

        chat_model_token: API token for the chat model endpoint.

        embedding_model_url: Base URL for the embedding model API.

        embedding_model_token: API token for the embedding model endpoint.

        search_space_prep_report: kfp-enforced argument specifying an output artifact.
            Provided by kfp backend automatically.

        embeddings_models: List of embedding model identifiers to try out in the experiment process.
            This list, if too long, will undergo models preselection (limiting).

        generation_models: List of generation model identifiers to try out in the experiment process.
            This list, if too long, will undergo models preselection (limiting).

        metric: Quality metric to evaluate the intermediate RAG patterns.
    """
    import logging
    import os
    import ssl
    from collections import namedtuple
    from dataclasses import fields, is_dataclass
    from pathlib import Path

    import httpx
    import pandas as pd
    import yaml as yml
    from ai4rag.core.experiment.benchmark_data import BenchmarkData
    from ai4rag.core.experiment.mps import ModelsPreSelector
    from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
    from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel
    from ai4rag.rag.foundation_models.base_model import BaseFoundationModel
    from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
    from ai4rag.search_space.prepare.prepare_search_space import (
        prepare_search_space_with_llama_stack,
    )
    from ai4rag.search_space.src.parameter import Parameter
    from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
    from langchain_core.documents import Document
    from llama_stack_client import APIConnectionError as LSAPIConnectionError
    from llama_stack_client import LlamaStackClient
    from openai import APIConnectionError as OAIAPIConnectionError
    from openai import OpenAI

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

    def _create_openai_client(api_key: str, base_url: str) -> OpenAI:
        """Create OpenAI client, falling back to SSL-unverified if self-signed cert detected."""
        client = OpenAI(api_key=api_key, base_url=base_url)
        try:
            client.models.list()
        except (ssl.SSLCertVerificationError, httpx.ConnectError, OAIAPIConnectionError) as exc:
            if _is_ssl_error(exc):
                _ssl_logger.warning(
                    "SSL verification failed for %s — retrying with verify=False. ",
                    base_url,
                )
                client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    http_client=httpx.Client(verify=False),
                )
            else:
                raise
        return client

    def _create_llama_stack_client(**kwargs) -> LlamaStackClient:
        """Create LlamaStackClient, falling back to SSL-unverified if self-signed cert detected."""
        client = LlamaStackClient(**kwargs)
        try:
            client.models.list()
        except (ssl.SSLCertVerificationError, httpx.ConnectError, LSAPIConnectionError) as exc:
            if _is_ssl_error(exc):
                _ssl_logger.warning("SSL verification failed for LlamaStackClient — retrying with verify=False. ")
                client = LlamaStackClient(
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

    if embeddings_models:
        if not isinstance(embeddings_models, list):
            raise TypeError("embeddings_models must be a list.")
        for i, m in enumerate(embeddings_models):
            if not m:
                raise TypeError(f"embeddings_models[{i}] must be a non-empty string.")

    if generation_models:
        if not isinstance(generation_models, list):
            raise TypeError("generation_models must be a list.")
        for i, m in enumerate(generation_models):
            if not m:
                raise TypeError(f"generation_models[{i}] must be a non-empty string.")

    if metric and metric not in supported_metrics:
        raise ValueError(f"Metric {metric} is not supported. Supported metrics are {supported_metrics}.")

    if embedding_model_url and chat_model_url:
        # Specification of OpenAI API compatibility
        embedding_model_url += "/v1"
        chat_model_url += "/v1"

    def _get_model_metadata_from(url: str, token: str) -> dict:
        """Retrieve specified model's metadata from the model's deployment URL.

        Currently the following keys are returned:
            - id: model's identifier
            - max_model_len: size of the context window

        Args:
            url: str
                Model deployment url.

            token: str
                Authorization token.

        Returns:
            Model's metadata as key-value mapping.

        Raises:
            ValueError
                If model metadata could not be retrieved (for whatever reason).
        """
        api_client = _create_openai_client(api_key=token, base_url=url)
        models = api_client.models.list()
        response = {"id": "", "max_model_len": 0}
        required_md = {"id"}

        if models.data:
            response["id"] = getattr(models.data[0], "id", "")
            response["max_model_len"] = getattr(models.data[0], "max_model_len", 0)

        for k in response.keys() & required_md:
            if not response[k]:
                raise ValueError(
                    f"Could not retrieve all the required model metadata from the provided url: ({url}). "
                    f"Please verify all the required data ({required_md}) is defined on "
                    "the deployment and can be obtained programmatically."
                )

        return response

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
        if in_memory_vector_store_scenario:
            gen_model_md = _get_model_metadata_from(chat_model_url, chat_model_token)
            em_model_md = _get_model_metadata_from(embedding_model_url, embedding_model_token)
            em_model_params = {"context_length": em_model_md["max_model_len"]} if em_model_md["max_model_len"] else {}
            params = [
                Parameter(
                    "foundation_model",
                    "C",
                    values=[
                        OpenAIFoundationModel(
                            client=client.generation_model,
                            model_id=gen_model_md["id"],
                            params={
                                "max_completion_tokens": 2048,
                                "temperature": 0.2,
                            },
                        )
                    ],
                ),
                Parameter(
                    "embedding_model",
                    "C",
                    values=[
                        OpenAIEmbeddingModel(
                            client=client.embedding_model,
                            model_id=em_model_md["id"],
                            params=em_model_params,
                        )
                    ],
                ),
            ]
            return AI4RAGSearchSpace(params=params, vector_store_type="chroma")
        else:
            payload = {}
            if generation_models:
                payload["foundation_models"] = [{"model_id": gm} for gm in generation_models]
            if embeddings_models:
                payload["embedding_models"] = [{"model_id": gm} for gm in embeddings_models]

            return prepare_search_space_with_llama_stack(payload, client=client.llama_stack)

    def represent_model_instance(dumper, model: BaseFoundationModel | BaseEmbeddingModel) -> yml.Node:
        """Helper method instructing the yml.Dumper on how to serialize the *Model instances"""
        if isinstance(model, BaseEmbeddingModel):
            type_ = "embedding"
        elif isinstance(model, BaseFoundationModel):
            type_ = "generation"

        params = model.params
        if is_dataclass(params):  # LS* model classes hold params as dataclass instances
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

    llama_stack_client_base_url = os.environ.get("LLAMA_STACK_CLIENT_BASE_URL", None)
    llama_stack_client_api_key = os.environ.get("LLAMA_STACK_CLIENT_API_KEY", None)

    in_memory_vector_store_scenario = False
    Client = namedtuple(
        "Client",
        ["llama_stack", "generation_model", "embedding_model"],
        defaults=[None, None, None],
    )

    if llama_stack_client_base_url and llama_stack_client_api_key:
        client = Client(llama_stack=_create_llama_stack_client())
    else:
        if not all(
            (
                chat_model_url,
                chat_model_token,
                embedding_model_url,
                embedding_model_token,
            )
        ):
            raise ValueError(
                "All of (`chat_model_url`, `chat_model_token`, `embedding_model_url`, `embedding_model_token`) "
                "have to be defined when running AutoRAG experiment on an in-memory vector store."
            )
        client = Client(
            generation_model=_create_openai_client(api_key=chat_model_token, base_url=chat_model_url),
            embedding_model=_create_openai_client(api_key=embedding_model_token, base_url=embedding_model_url),
        )
        in_memory_vector_store_scenario = True

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
