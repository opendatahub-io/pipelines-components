from pathlib import Path
from typing import List, Optional

from kfp import dsl
from kfp.compiler import Compiler
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]

_AUTORAG_SHARED = Path(__file__).parents[1] / "shared"


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
    embedded_artifact_path=str(_AUTORAG_SHARED / "component_status.py"),
    install_kfp_package=False,
)
def search_space_preparation(
    test_data: dsl.Input[dsl.Artifact],
    extracted_text: dsl.Input[dsl.Artifact],
    search_space_prep_report: dsl.Output[dsl.Artifact],
    component_status: dsl.Output[dsl.Artifact] = None,
    embedded_artifact: dsl.EmbeddedInput[dsl.Dataset] = None,
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
        component_status: Output artifact containing stage-level progress tracking.
        embedded_artifact: Embedded ``autorag.shared`` helpers injected by KFP at runtime.

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
    LANGUAGE_DETECTION_SAMPLE_SIZE = 10

    # English-only names: KFP's MySQL backend rejects multi-byte UTF-8 in
    # PipelineRuntimeManifest, so no non-ASCII literals in component source.
    LANGUAGE_MAP: dict[str, str] = {
        "ja": "Japanese",
        "ko": "Korean",
        "zh-cn": "Chinese",
        "zh-tw": "Chinese",
        "en": "English",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
        "pt": "Portuguese",
        "it": "Italian",
        "ru": "Russian",
        "ar": "Arabic",
        "hi": "Hindi",
        "th": "Thai",
        "vi": "Vietnamese",
        "pl": "Polish",
        "nl": "Dutch",
        "sv": "Swedish",
        "cs": "Czech",
        "tr": "Turkish",
    }

    def _detect_language_via_llm(
        questions: list[str],
        llm_client: OgxClient,
        allowed_generation_models: list[str] | None = None,
    ) -> dict[str, str] | None:
        """Detect the dominant language by asking an LLM.

        Sends a few sample questions to a generation model and asks for the ISO 639-1 code.
        Prefers models from allowed_generation_models when provided.
        Returns a dict with code/name, or None for English.
        """
        sample_text = "\n".join(f"- {q}" for q in questions[:5])
        valid_codes = ", ".join(sorted(LANGUAGE_MAP.keys()))

        try:
            models_response = llm_client.models.list()
            models_list = models_response.data if hasattr(models_response, "data") else list(models_response)
            registered_ids = {(m.identifier if hasattr(m, "identifier") else str(m.id)) for m in models_list}

            model_id = None
            if allowed_generation_models:
                for gm in allowed_generation_models:
                    if gm in registered_ids:
                        model_id = gm
                        break
            if not model_id:
                for m in models_list:
                    if hasattr(m, "model_type") and getattr(m, "model_type", "") == "llm":
                        model_id = m.identifier if hasattr(m, "identifier") else str(m.id)
                        break
            if not model_id:
                _ssl_logger.warning("No models available for LLM language detection.")
                return None

            response = llm_client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a language detection assistant. "
                            "Given text samples, respond with ONLY the ISO 639-1 language code "
                            f"(one of: {valid_codes}). "
                            "Nothing else — just the code."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"What language are these questions written in?\n{sample_text}",
                    },
                ],
                max_completion_tokens=10,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip().lower().replace('"', "").replace("'", "")
            detected_code = raw.split()[0] if raw else None

            if not detected_code:
                return None
            if detected_code == "en":
                return {"code": "en", "name": "English"}

            name = LANGUAGE_MAP.get(detected_code)
            if not name:
                _ssl_logger.warning("LLM returned unsupported language code: %s", detected_code)
                return None
            _ssl_logger.info("Language detected via LLM: %s (%s)", detected_code, name)
            return {"code": detected_code, "name": name}

        except Exception as exc:
            _ssl_logger.warning("LLM language detection failed: %s", exc)
            return None

    def _detect_benchmark_language(df: pd.DataFrame, llm_client: OgxClient) -> dict[str, str] | None:
        """Detect the dominant language from benchmark questions.

        Uses LLM via OGX for accurate detection across all languages.
        Returns a dict with code/name, or None for English / on failure.
        """
        questions = df["question"].dropna().astype(str).tolist()
        if not questions:
            return None

        sample = questions[:LANGUAGE_DETECTION_SAMPLE_SIZE]
        return _detect_language_via_llm(sample, llm_client, allowed_generation_models=generation_models)

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
                    doc_id = doc_path.stem if doc_path.suffix == ".md" else doc_path.name
                    documents.append(
                        Document(
                            page_content=doc.read(),
                            metadata={"document_id": doc_id},
                        )
                    )

        elif path.is_file():
            doc_id = path.stem if path.suffix == ".md" else path.name
            with path.open("r", encoding="utf-8") as doc:
                documents.append(Document(page_content=doc.read(), metadata={"document_id": doc_id}))

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

    import sys

    sys.path.insert(0, str(Path(embedded_artifact.path)))
    try:
        from component_status import component_status_tracker
    finally:
        sys.path.pop(0)

    status = component_status_tracker(component_status, "search_space_preparation")
    with status:
        with status.stage("validate_inputs"):
            if not ogx_client_base_url or not ogx_client_api_key:
                raise ValueError("OGX_CLIENT_BASE_URL and OGX_CLIENT_API_KEY environment variables must be set.")

            client = _create_ogx_client(base_url=ogx_client_base_url, api_key=ogx_client_api_key)

        with status.stage("prepare_search_space"):
            search_space = prepare_ai4rag_search_space()

            benchmark_df = pd.read_json(Path(test_data.path))
            detected_language = _detect_benchmark_language(benchmark_df, llm_client=client)
            benchmark_data = BenchmarkData(benchmark_df)
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

            if detected_language:
                verbose_search_space_repr["detected_language"] = detected_language

        with status.stage("write_report"):
            with open(search_space_prep_report.path, "w") as report_file:
                yml.safe_dump(verbose_search_space_repr, report_file)


if __name__ == "__main__":
    Compiler().compile(
        search_space_preparation,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
