from pathlib import Path
from typing import Optional

from kfp import dsl
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
    embedded_artifact_path=str((Path(__file__).parent / "notebook_templates")),
)
def rag_templates_optimization(
    extracted_text: dsl.InputPath(dsl.Artifact),
    test_data: dsl.InputPath(dsl.Artifact),
    search_space_prep_report: dsl.InputPath(dsl.Artifact),
    rag_patterns: dsl.Output[dsl.Artifact],
    embedded_artifact: dsl.EmbeddedInput[dsl.Dataset],
    test_data_key: Optional[str],
    chat_model_url: Optional[str] = None,
    chat_model_token: Optional[str] = None,
    embedding_model_url: Optional[str] = None,
    embedding_model_token: Optional[str] = None,
    llama_stack_vector_io_provider_id: Optional[str] = None,
    optimization_settings: Optional[dict] = None,
    input_data_key: Optional[str] = "",
):
    """RAG Templates Optimization component.

    Carries out the iterative RAG optimization process.

    Args:
        extracted_text: A path pointing to a folder containg extracted texts from input documents.

        test_data: A path pointing to test data used for evaluating RAG pattern quality.

        search_space_prep_report: A path pointing to a .yml file containig short
            report on the experiment's first phase (search space preparation).

        rag_patterns: kfp-enforced argument specifying an output artifact. Provided by kfp backend automatically.

        embedded_artifact: kfp-enforced argument to allow access of base64 encoded dir with notebook templates.

        test_data_key: Path to the benchmark JSON file in object storage used by generated notebooks.

        chat_model_url: Inference endpoint URL for the chat/generation model (OpenAI-compatible).
            Required for in-memory scenario.

        chat_model_token: Optional API token for the chat model endpoint. Omit if deployment has no auth.

        embedding_model_url: Inference endpoint URL for the embedding model. Required for in-memory scenario.

        embedding_model_token: Optional API token for the embedding model endpoint. Omit if no auth.

        llama_stack_vector_io_provider_id: Vector I/O provider identifier as registered in llama-stack.

        optimization_settings: Additional settings customising the experiment.

        input_data_key: A path to documents dir within a bucket used as an input to AI4RAG experiment.

    Returns:
        rag_patterns: Folder containing all generated RAG patterns (each subdir: pattern.json,
            indexing_notebook.ipynb, inference_notebook.ipynb).
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
    from collections import namedtuple
    from json import dump as json_dump
    from json import load as json_load
    from pathlib import Path
    from string import Formatter
    from typing import Any, Literal, Self

    import httpx
    import pandas as pd
    import yaml as yml
    from ai4rag.core.experiment.experiment import AI4RAGExperiment
    from ai4rag.core.experiment.results import ExperimentResults
    from ai4rag.core.hpo.gam_opt import GAMOptSettings
    from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
    from ai4rag.rag.embedding.llama_stack import LSEmbeddingModel
    from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel
    from ai4rag.rag.foundation_models.base_model import BaseFoundationModel
    from ai4rag.rag.foundation_models.llama_stack import LSFoundationModel
    from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
    from ai4rag.search_space.src.parameter import Parameter
    from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
    from ai4rag.utils.event_handler.event_handler import BaseEventHandler, LogLevel
    from langchain_core.documents import Document
    from llama_stack_client import APIConnectionError as LSAPIConnectionError
    from llama_stack_client import LlamaStackClient
    from openai import APIConnectionError as OAIAPIConnectionError
    from openai import OpenAI

    DEFAULT_MAX_NUMBER_OF_RAG_PATTERNS = 8
    MAX_NUMBER_OF_RAG_PATTERNS_ALLOWED_RANGE = (4, 20)
    METRIC = "faithfulness"
    SUPPORTED_OPTIMIZATION_METRICS = frozenset({"faithfulness", "answer_correctness", "context_correctness"})
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

    # Mutable containers so the nested _create_openai_client can update the flags
    # without requiring `nonlocal` or returning the flag.
    _chat_ssl_verify = [True]
    _embedding_ssl_verify = [True]

    def _create_openai_client(api_key: str, base_url: str, _ssl_verify: list) -> OpenAI:
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
                _ssl_verify[0] = False
            else:
                raise
        return client

    # Mutable container so the nested _create_llama_stack_client can update the flag
    # without requiring `nonlocal` or returning the flag.
    _ls_ssl_verify = [True]

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
                _ls_ssl_verify[0] = False
            else:
                raise
        return client

    if not isinstance(test_data_key, str) or not test_data_key.strip() or not test_data_key.lower().endswith(".json"):
        raise ValueError("test_data_path must point to a JSON file")

    if optimization_settings is not None:
        if not isinstance(optimization_settings, dict):
            raise TypeError("optimization_settings must be a dictionary.")
        max_rag_patterns = optimization_settings.get("max_number_of_rag_patterns", DEFAULT_MAX_NUMBER_OF_RAG_PATTERNS)
        if isinstance(max_rag_patterns, str):
            try:
                max_rag_patterns = int(max_rag_patterns.strip())
            except ValueError as exc:
                raise ValueError(
                    "optimization_settings.max_number_of_rag_patterns must be a valid integer "
                    f"(e.g. from the pipeline UI); got {max_rag_patterns!r}."
                ) from exc
        if not isinstance(max_rag_patterns, int):
            raise TypeError("optimization_settings.max_number_of_rag_patterns must be an integer.")

        _ssl_logger.info("max_number_of_rag_patterns %s", max_rag_patterns)
        if not (
            MAX_NUMBER_OF_RAG_PATTERNS_ALLOWED_RANGE[0]
            <= max_rag_patterns
            <= MAX_NUMBER_OF_RAG_PATTERNS_ALLOWED_RANGE[1]
        ):
            raise ValueError(
                f"optimization_settings.max_number_of_rag_patterns must be in a range"
                f"{MAX_NUMBER_OF_RAG_PATTERNS_ALLOWED_RANGE[0]} to "
                f"{MAX_NUMBER_OF_RAG_PATTERNS_ALLOWED_RANGE[1]}."
            )

    class NotebookCell:
        """Represents a single cell in a Jupyter notebook.

        Parameters
        ----------
        cell_type : Literal["code", "markdown"]
            The type of cell.
        source : str | list[str]
            The cell content. Can be a string or list of strings.
        metadata : dict, optional
            Cell metadata.
        """

        def __init__(
            self,
            cell_type: Literal["code", "markdown"],
            source: str | list[str],
            metadata: dict | None = None,
        ):
            self.cell_type = cell_type
            self.metadata = metadata or {}

            self.source = source

            if cell_type == "code":
                self.execution_count = None
                self.outputs = []

        def to_dict(self) -> dict:
            """Convert cell to notebook JSON format.

            Returns:
                dict: Cell in notebook format.
            """
            cell_dict = {
                "cell_type": self.cell_type,
                "metadata": self.metadata,
                "source": self.source,
            }

            if self.cell_type == "code":
                cell_dict["execution_count"] = self.execution_count
                cell_dict["outputs"] = self.outputs

            return cell_dict

        def format_source(
            self,
            placeholders_mapping: dict,
        ) -> Self:
            """Formats cell source based on provided placeholders_mapping.

            Returns:
                Self: Instance of NotebookCell.
            """
            if isinstance(self.source, list):
                new_source = []
                for line in self.source:
                    line_mapping = {}
                    for _, field_name, _, _ in Formatter().parse(line):
                        if field_name is None:
                            continue
                        line_mapping[field_name] = placeholders_mapping.get(field_name, "")

                    new_source.append(line.format(**line_mapping))
                    self.source = new_source

                return self

            self.source = self.source.format(**placeholders_mapping)

            return self

    class Notebook:
        """Builder class for creating and manipulating Jupyter notebooks.

        This class provides a fluent API for programmatically building notebooks
        by adding code and markdown cells, formatting content, and saving to disk.

        Parameters
        ----------
        kernel_name : str, default="python3"
            Kernel name for the notebook.
        kernel_display_name : str, default="Python 3"
            Display name for the kernel.
        language : str, default="python"
            Programming language.
        language_version : str, default="3.11.0"
            Language version.
        cells : list[NotebookCell] | None, default=None
            Notebook cells to build the notebook from.

        Examples:
        --------
        >>> nb = Notebook(
            cells=[
                NotebookCell(
                    cell_type="markdown",
                    source="### Hello world!",
                )
            ])
        >>> nb.save("output.ipynb")
        """

        def __init__(
            self,
            kernel_name: str = "python3",
            kernel_display_name: str = "Python 3",
            language: str = "python",
            language_version: str = "3.13.11",
            cells: list[NotebookCell] | None = None,
        ):
            self.cells: list[NotebookCell] = cells if cells else []
            self.metadata = {
                "kernelspec": {
                    "display_name": kernel_display_name,
                    "language": language,
                    "name": kernel_name,
                },
                "language_info": {"name": language, "version": language_version},
            }
            self.nbformat = 4
            self.nbformat_minor = 4

        def to_dict(self) -> dict:
            """Convert notebook to dictionary format.

            Returns:
                dict: Notebook in JSON format.
            """
            return {
                "cells": [cell.to_dict() for cell in self.cells],
                "metadata": self.metadata,
                "nbformat": self.nbformat,
                "nbformat_minor": self.nbformat_minor,
            }

        def save(self, path: str | Path, indent: int = 2) -> "Notebook":
            """Save notebook to a file.

            Parameters
            ----------
            path : str | Path
                Output file path.
            indent : int, default=2
                JSON indentation level.

            Returns:
                Notebook: Self for method chaining.

            Examples:
            --------
            >>> nb = Notebook()
            >>> nb.save("output.ipynb")
            """
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with path.open("w+") as f:
                json_dump(self.to_dict(), f, indent=indent)

            return self

        @classmethod
        def load(
            cls,
            notebook_name: Literal[
                "ls_indexing_template.ipynb",
                "ls_inference_template.ipynb",
                "chroma_template.ipynb",
            ],
        ) -> "Notebook":
            """Load a Jupyter notebook from a file.

            Parameters
            ----------
            path : str | Path
                Input file path to the .ipynb file.

            Returns:
            -------
            Notebook
                A new Notebook instance populated with the loaded cells and metadata.

            Examples:
            --------
            >>> nb = Notebook.load("existing_notebook.ipynb")
            """
            with open(Path(embedded_artifact.path) / notebook_name, "r") as f:
                nb_dict = json_load(f)

            loaded_cells = []
            for cell_data in nb_dict.get("cells", []):
                cell = NotebookCell(
                    cell_type=cell_data.get("cell_type", "code"),
                    source=cell_data.get("source", ""),
                    metadata=cell_data.get("metadata", {}),
                )

                # Restore code-specific attributes not handled in __init__
                if cell.cell_type == "code":
                    cell.execution_count = cell_data.get("execution_count")
                    cell.outputs = cell_data.get("outputs", [])

                loaded_cells.append(cell)

            # Safely extract metadata to initialize the Notebook properly
            metadata = nb_dict.get("metadata", {})
            kernelspec = metadata.get("kernelspec", {})
            language_info = metadata.get("language_info", {})

            # Instantiate the new Notebook with the parsed cells
            notebook = cls(
                kernel_name=kernelspec.get("name", "python3"),
                kernel_display_name=kernelspec.get("display_name", "Python 3"),
                language=language_info.get("name", "python"),
                language_version=language_info.get("version", "3.13.11"),
                cells=loaded_cells,
            )

            # Preserve the exact original metadata and notebook formatting versions
            notebook.metadata = metadata
            notebook.nbformat = nb_dict.get("nbformat", 4)
            notebook.nbformat_minor = nb_dict.get("nbformat_minor", 4)

            return notebook

    def create_placeholder_mapping(
        output_data: dict[str, Any],
        test_data_key: str = "",
        input_data_key: str = "",
        chat_model_url: str = "",
        embedding_model_url: str = "",
        ls_ssl_verify: bool = True,
        chat_ssl_verify: bool = True,
        embedding_ssl_verify: bool = True,
    ) -> dict[str, Any]:
        """Create a mapping from placeholder names to their values from output.json.

        This function extracts values from the output.json structure and creates
        a flat dictionary suitable for use with NotebookCell.format_source().

        Expected output.json structure:
        {
            "config": {
                "pattern_name": "...",
                "autorag_version": "...",
                "llama_stack": {
                    "foundation_model": {...},
                    "embedding_model": {...},
                    "vector_store": {...},
                    "retriever": {...},
                    "chunker": {...}
                },
                "data": {...}
            }
        }

        Args:
            output_data: The parsed pattern.json data
            test_data_key: Test data key.
            input_data_key: Input data key.
            chat_model_url: Chat model url.
            embedding_model_url: Embedding model url.
            ls_ssl_verify: Whether LlamaStack SSL verification succeeded.
            chat_ssl_verify: Whether chat model (OpenAI) SSL verification succeeded.
            embedding_ssl_verify: Whether embedding model (OpenAI) SSL verification succeeded.

        Returns:
            Dictionary mapping placeholder names to their values.
        """
        mapping = {}

        mapping["PATTERN_NAME"] = output_data.get("name", "")
        settings = output_data.get("settings", {})
        fm = settings.get("generation", {})
        mapping["FM_MODEL_ID"] = fm.get("model_id", "")
        mapping["SYSTEM_MESSAGE"] = fm.get("system_message_text", "")
        mapping["USER_MESSAGE"] = fm.get("user_message_text", "")
        mapping["CONTEXT_TEXT"] = fm.get("context_template_text", "")

        em = settings.get("embedding", {})
        mapping["EMBEDDING_MODEL_ID"] = em.get("model_id", "")
        mapping["EMBEDDING_PARAMS"] = em.get("embedding_params", {"embedding_dimension": 768})
        mapping["DISTANCE_METRIC"] = em.get("distance_metric", "")

        vs = settings.get("vector_store", {})
        mapping["PROVIDER_ID"] = vs.get("datasource_type", "")
        mapping["COLLECTION_NAME"] = vs.get("collection_name", "")

        ret = settings.get("retrieval", {})
        mapping["RETRIEVAL_METHOD"] = ret.get("method", "")
        mapping["NUMBER_OF_CHUNKS"] = ret.get("number_of_chunks", 5)

        ch = settings.get("chunking", {})
        mapping["CHUNKING_METHOD"] = ch.get("method", "")
        mapping["CHUNK_SIZE"] = ch.get("chunk_size", 512)
        mapping["CHUNK_OVERLAP"] = ch.get("chunk_overlap", 50)

        mapping["TEST_DATA_KEY"] = test_data_key
        mapping["INPUT_DATA_KEY"] = input_data_key

        mapping["CHAT_MODEL_URL"] = chat_model_url
        mapping["EMBEDDING_MODEL_URL"] = embedding_model_url
        mapping["LS_SSL_VERIFY"] = ls_ssl_verify
        mapping["CHAT_SSL_VERIFY"] = chat_ssl_verify
        mapping["EMBEDDING_SSL_VERIFY"] = embedding_ssl_verify

        return mapping

    def generate_notebook_from_templates(
        notebook_template: Literal[
            "ls_inference",
            "ls_indexing",
            "chroma",
        ],
        output_data: dict[str, Any],
        output_notebook_path: Path,
        test_data_key: str = "",
        input_data_key: str = "",
        ls_ssl_verify: bool = True,
        chat_ssl_verify: bool = True,
        embedding_ssl_verify: bool = True,
    ) -> None:
        """Generate a filled notebook from templates and output.json.

        Args:
            notebook_template: One of the allowed template names.
            output_data: The parsed output.json data.
            output_notebook_path: Path where to save the generated notebook.
            test_data_key: Path to test data file within bucket used as input to AI4RAG.
            input_data_key: Path to documents dir within bucket used as input to AI4RAG.
            ls_ssl_verify: Whether LlamaStack SSL verification succeeded; False causes the
                generated notebook to skip the SSL probe and connect with verify=False directly.
            chat_ssl_verify: Whether chat model (OpenAI) SSL verification succeeded.
            embedding_ssl_verify: Whether embedding model (OpenAI) SSL verification succeeded.

        Returns:
            None. The notebook is written to output_notebook_path.
        """
        placeholder_mapping = create_placeholder_mapping(
            output_data,
            test_data_key=test_data_key,
            input_data_key=input_data_key,
            chat_model_url=chat_model_url,
            embedding_model_url=embedding_model_url,
            ls_ssl_verify=ls_ssl_verify,
            chat_ssl_verify=chat_ssl_verify,
            embedding_ssl_verify=embedding_ssl_verify,
        )
        notebook = Notebook.load(notebook_name=f"{notebook_template}_template.ipynb")
        filled_cells = []
        for cell in notebook.cells:
            filled_cell = cell.format_source(placeholder_mapping)
            filled_cells.append(filled_cell)

        notebook = Notebook(cells=filled_cells)

        notebook.save(Path(output_notebook_path))

    if embedding_model_url and chat_model_url:
        # Specification of OpenAI API compatibility
        embedding_model_url += "/v1"
        chat_model_url += "/v1"

    class TmpEventHandler(BaseEventHandler):
        """Exists temporarily only for the purpose of satisying type hinting checks"""

        def on_status_change(self, level: LogLevel, message: str, step: str | None = None) -> None:
            pass

        def on_pattern_creation(self, payload: dict, evaluation_results: list, **kwargs) -> None:
            pass

    def load_as_langchain_doc(path: str | Path) -> list[Document]:
        """Load a text file or folder into a list of langchain Document objects.

        Args:
            path: A local path to either a text file or a folder of text files.

        Returns:
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
        for name, value in (
            ("chat_model_url", chat_model_url),
            ("chat_model_token", chat_model_token),
            ("embedding_model_url", embedding_model_url),
            ("embedding_model_token", embedding_model_token),
        ):
            if not value:
                raise TypeError(f"{name} must be a non-empty string.")
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
                "have to be defined when running AutoRAG experiment using an in-memory vector store."
            )
        client = Client(
            generation_model=_create_openai_client(
                api_key=chat_model_token, base_url=chat_model_url, _ssl_verify=_chat_ssl_verify
            ),
            embedding_model=_create_openai_client(
                api_key=embedding_model_token, base_url=embedding_model_url, _ssl_verify=_embedding_ssl_verify
            ),
        )
        in_memory_vector_store_scenario = True

    def construct_model_instance(loader, node: yml.MappingNode) -> BaseEmbeddingModel | BaseFoundationModel:
        """Instructs yml.Loader on how to construct "!Model" tag."""
        mapping = loader.construct_mapping(node, deep=True)

        match mapping:
            case {"type_": "embedding", **id_to_params}:
                model_id, params = id_to_params.popitem()
                if in_memory_vector_store_scenario:
                    return OpenAIEmbeddingModel(client=client.embedding_model, model_id=model_id, params=params)
                else:
                    return LSEmbeddingModel(client=client.llama_stack, model_id=model_id, params=params)

            case {"type_": "generation", **id_to_params}:
                model_id, params = id_to_params.popitem()
                if in_memory_vector_store_scenario:
                    return OpenAIFoundationModel(client=client.generation_model, model_id=model_id, params=params)
                else:
                    return LSFoundationModel(client=client.llama_stack, model_id=model_id, params=params)
            case _:
                raise ValueError(f"Cannot load the yml-serialized !Model tag: {mapping}")

    yml.add_constructor("!Model", construct_model_instance, Loader=yml.SafeLoader)

    optimization_settings = optimization_settings if optimization_settings else {}
    if not (optimization_metric := optimization_settings.get("metric", None)):
        optimization_metric = METRIC
    if optimization_metric not in SUPPORTED_OPTIMIZATION_METRICS:
        raise ValueError(
            "optimization_metric must be one of %s; got %r"
            % (sorted(SUPPORTED_OPTIMIZATION_METRICS), optimization_metric)
        )

    documents = load_as_langchain_doc(extracted_text)

    # reload the search space
    with open(search_space_prep_report, "r") as f:
        search_space = yml.safe_load(f)

    search_space = AI4RAGSearchSpace(
        params=[Parameter(param, "C", values=values) for param, values in search_space.items()]
    )

    event_handler = TmpEventHandler()
    max_rag_patterns = optimization_settings.get("max_number_of_rag_patterns", DEFAULT_MAX_NUMBER_OF_RAG_PATTERNS)
    if isinstance(max_rag_patterns, str):
        try:
            max_rag_patterns = int(max_rag_patterns.strip())
        except ValueError as exc:
            raise ValueError(
                "optimization_settings.max_number_of_rag_patterns must be a valid integer "
                f"(e.g. from the pipeline UI); got {max_rag_patterns!r}."
            ) from exc
    optimizer_settings = GAMOptSettings(max_evals=max_rag_patterns)

    benchmark_data = pd.read_json(Path(test_data))

    if not llama_stack_vector_io_provider_id or not llama_stack_vector_io_provider_id.strip():
        if in_memory_vector_store_scenario:
            llama_stack_vector_io_provider_id = "chroma"
        else:
            raise ValueError(
                "llama_stack_vector_io_provider_id must be provided when using llama-stack vector database."
            )

    # ai4rag expects vector_store_type with an "ls_" prefix for llama-stack providers.
    # Users provide the raw llama-stack provider_id (e.g. "milvus"); the prefix is added here.
    # If the user already included "ls_", don't double-prefix.
    if in_memory_vector_store_scenario:
        vector_store_type = llama_stack_vector_io_provider_id
    elif llama_stack_vector_io_provider_id.startswith("ls_"):
        vector_store_type = llama_stack_vector_io_provider_id
    else:
        vector_store_type = f"ls_{llama_stack_vector_io_provider_id}"

    rag_exp = AI4RAGExperiment(
        client=None if in_memory_vector_store_scenario else client.llama_stack,
        event_handler=event_handler,
        optimizer_settings=optimizer_settings,
        search_space=search_space,
        benchmark_data=benchmark_data,
        vector_store_type=vector_store_type,
        documents=documents,
        optimization_metric=optimization_metric,
        # TODO some necessary kwargs (if any at all)
    )

    # retrieve documents && run optimisation loop
    rag_exp.search()

    def _evaluation_result_fallback(eval_data_list, evaluation_result):
        """Build evaluation_results.json-style list when question_scores missing or incomplete."""
        out = []
        for ev in eval_data_list:
            answer_contexts = []
            if getattr(ev, "contexts", None) and getattr(ev, "context_ids", None):
                answer_contexts = [{"text": t, "document_id": doc_id} for t, doc_id in zip(ev.contexts, ev.context_ids)]
            scores = {}
            q_scores = (evaluation_result.scores or {}).get("question_scores") or {}
            for key in q_scores:
                if isinstance(q_scores[key], dict) and getattr(ev, "question_id", None) in q_scores[key]:
                    scores[key] = q_scores[key][ev.question_id]
            out.append(
                {
                    "question": getattr(ev, "question", ""),
                    "correct_answers": getattr(ev, "ground_truths", None),
                    "answer": getattr(ev, "answer", ""),
                    "answer_contexts": answer_contexts,
                    "scores": scores,
                }
            )
        return out

    rag_patterns_dir = Path(rag_patterns.path)
    evaluation_data_list = getattr(rag_exp.results, "evaluation_data", [])

    def _build_pattern_json(evaluation_result, iteration: int, max_combinations: int) -> dict:
        """Build pattern.json with flat schema (name, iteration, settings, scores, final_score)."""
        idx = evaluation_result.indexing_params or {}
        rp = evaluation_result.rag_params or {}
        chunking = idx.get("chunking") or {}
        # ai4rag puts embedding in indexing_params.embedding, not rag_params
        embedding_from_idx = idx.get("embedding") or idx.get("embeddings") or {}
        embeddings = rp.get("embeddings") or rp.get("embedding") or embedding_from_idx
        retrieval = rp.get("retrieval") or {}

        # ai4rag retrieval: search_mode is "hybrid" | "vector"; ranker_* used when search_mode is hybrid
        def _ret(key: str, default=None):
            return retrieval.get(key) if isinstance(retrieval, dict) else default

        def _rp(key: str, default=None):
            return rp.get(key) if isinstance(rp, dict) else default

        retrieval_method = _ret("method") or _ret("retrieval_method") or _rp("retrieval_method") or "simple"
        number_of_chunks = _ret("number_of_chunks") or _rp("number_of_chunks") or 5
        search_mode = _ret("search_mode") or _rp("search_mode")
        ranker_strategy = _ret("ranker_strategy") or _rp("ranker_strategy")
        ranker_k = _ret("ranker_k") if _ret("ranker_k") is not None else _rp("ranker_k")
        ranker_alpha = _ret("ranker_alpha") if _ret("ranker_alpha") is not None else _rp("ranker_alpha")
        generation = rp.get("generation") or {}
        # embedding model_id: from indexing_params.embedding (ai4rag), or rag_params, or flat embedding_model
        embedding_model_id = None
        if isinstance(embedding_from_idx, dict) and embedding_from_idx.get("model_id"):
            embedding_model_id = embedding_from_idx.get("model_id")
        if not embedding_model_id and isinstance(embeddings, dict):
            embedding_model_id = embeddings.get("model_id")
        if not embedding_model_id and isinstance(rp.get("embedding_model"), str):
            embedding_model_id = rp.get("embedding_model")
        if not embedding_model_id and hasattr(rp.get("embedding_model"), "model_id"):
            embedding_model_id = getattr(rp.get("embedding_model"), "model_id", None)
        # generation model_id: from rag_params.generation (ai4rag) or flat foundation_model
        generation_model_id = generation.get("model_id") if isinstance(generation, dict) else None
        if not generation_model_id and isinstance(rp.get("foundation_model"), str):
            generation_model_id = rp.get("foundation_model")
        if not generation_model_id and hasattr(rp.get("foundation_model"), "model_id"):
            generation_model_id = getattr(rp.get("foundation_model"), "model_id", None)
        return {
            "name": getattr(evaluation_result, "pattern_name", ""),
            "iteration": iteration,
            "max_combinations": max_combinations,
            "duration_seconds": getattr(evaluation_result, "execution_time", 0) or 0,
            "settings": {
                "vector_store": {
                    "datasource_type": idx.get("vector_store", {}).get("datasource_type")
                    or rp.get("vector_store", {}).get("datasource_type")
                    or vector_store_type,
                    "collection_name": getattr(evaluation_result, "collection", "") or "",
                },
                "chunking": {
                    "method": chunking.get("method", "recursive"),
                    "chunk_size": chunking.get("chunk_size", 2048),
                    "chunk_overlap": chunking.get("chunk_overlap", 256),
                },
                "embedding": {
                    "model_id": embedding_model_id or "",
                    "distance_metric": (
                        embeddings.get("distance_metric", "cosine") if isinstance(embeddings, dict) else "cosine"
                    ),
                    "embedding_params": embeddings.get("embedding_params", {"embedding_dimension": 768}),
                },
                "retrieval": {
                    "method": retrieval_method,
                    "number_of_chunks": number_of_chunks,
                    **({"search_mode": search_mode} if search_mode is not None else {}),
                    **({"ranker_strategy": ranker_strategy} if ranker_strategy is not None else {}),
                    **({"ranker_k": ranker_k} if ranker_k is not None else {}),
                    **({"ranker_alpha": ranker_alpha} if ranker_alpha is not None else {}),
                },
                "generation": {
                    "model_id": generation_model_id or "",
                    "context_template_text": generation.get("context_template_text", "{document}"),
                    "user_message_text": generation.get(
                        "user_message_text",
                        (
                            "\n\nContext:\n{reference_documents}:\n\nQuestion: {question}. \nAgain, please answer "
                            "the question based on the context provided only. If the context is not related to "
                            "the question, just say you cannot answer. Respond exclusively in the language of "
                            "the question."
                        ),
                    ),
                    "system_message_text": generation.get(
                        "system_message_text",
                        (
                            "Please answer the question I provide in the Question section below, based solely "
                            "on the information I provide in the Context section. If unanswerable, say so."
                        ),
                    ),
                },
            },
        }

    evaluations_list = list(rag_exp.results.evaluations)
    max_combinations = getattr(rag_exp.results, "max_combinations", len(evaluations_list)) or 24

    rag_patterns.metadata["name"] = "rag_patterns_artifact"
    rag_patterns.metadata["uri"] = rag_patterns.uri
    rag_patterns.metadata["metadata"] = {"patterns": []}
    for i, eval in enumerate(evaluations_list):
        patt_dir = rag_patterns_dir / eval.pattern_name
        patt_dir.mkdir(parents=True, exist_ok=True)

        pattern_data = _build_pattern_json(eval, iteration=i, max_combinations=max_combinations)
        if llama_stack_vector_io_provider_id == "chroma":
            generate_notebook_from_templates(
                "chroma",
                pattern_data,
                Path(patt_dir, "indexing_and_inference.ipynb"),
                input_data_key=input_data_key,
                test_data_key=test_data_key,
                chat_ssl_verify=_chat_ssl_verify[0],
                embedding_ssl_verify=_embedding_ssl_verify[0],
            )
        else:
            generate_notebook_from_templates(
                "ls_indexing",
                pattern_data,
                Path(patt_dir, "indexing.ipynb"),
                input_data_key=input_data_key,
                ls_ssl_verify=_ls_ssl_verify[0],
            )

            generate_notebook_from_templates(
                "ls_inference",
                pattern_data,
                Path(patt_dir, "inference.ipynb"),
                test_data_key=test_data_key,
                ls_ssl_verify=_ls_ssl_verify[0],
            )

        # Flat schema: scores = per-metric aggregates (mean, ci_low, ci_high); final_score
        pattern_data["scores"] = (getattr(eval, "scores", None) or {}).get("scores") or {}
        pattern_data["final_score"] = getattr(eval, "final_score", None)
        rag_patterns.metadata["metadata"]["patterns"].append(pattern_data)
        with (patt_dir / "pattern.json").open("w+", encoding="utf-8") as pattern_details:
            json_dump(pattern_data, pattern_details, indent=2)

        eval_data = evaluation_data_list[i] if i < len(evaluation_data_list) else []
        try:
            q_scores = (eval.scores or {}).get("question_scores") or {}
            if q_scores and all(isinstance(q_scores.get(k), dict) for k in q_scores):
                evaluation_result_list = ExperimentResults.create_evaluation_results_json(eval_data, eval)
            else:
                evaluation_result_list = _evaluation_result_fallback(eval_data, eval)
        except (KeyError, TypeError):
            evaluation_result_list = _evaluation_result_fallback(eval_data, eval)
        with (patt_dir / "evaluation_results.json").open("w+", encoding="utf-8") as f:
            json_dump(evaluation_result_list, f, indent=2)

    # TODO autorag_run_artifact


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        rag_templates_optimization,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
