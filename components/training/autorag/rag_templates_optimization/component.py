from typing import Optional

from kfp import dsl
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
    embedded_artifact_path=("components/training/autorag/shared"),
    install_kfp_package=False,
)
def rag_templates_optimization(
    extracted_text: dsl.InputPath(dsl.Artifact),
    test_data: dsl.InputPath(dsl.Artifact),
    search_space_prep_report: dsl.InputPath(dsl.Artifact),
    rag_patterns: dsl.Output[dsl.Artifact],
    embedded_artifact: dsl.EmbeddedInput[dsl.Dataset],
    test_data_key: Optional[str],
    vector_io_provider_id: str,
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

        vector_io_provider_id: Vector I/O provider identifier as registered in OGX.

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
    from json import dump as json_dump
    from json import load as json_load
    from pathlib import Path
    from re import search
    from string import Formatter, Template
    from typing import Any, Literal, Self

    import httpx
    import pandas as pd
    import yaml as yml
    from ai4rag.core.experiment.experiment import AI4RAGExperiment
    from ai4rag.core.hpo.gam_opt import GAMOptSettings
    from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
    from ai4rag.rag.embedding.ogx import OGXEmbeddingModel
    from ai4rag.rag.foundation_models.base_model import BaseFoundationModel
    from ai4rag.rag.foundation_models.ogx import OGXFoundationModel
    from ai4rag.search_space.src.parameter import Parameter
    from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
    from ai4rag.utils.event_handler import KFPEventHandler
    from langchain_core.documents import Document
    from ogx_client import OgxClient

    DEFAULT_MAX_NUMBER_OF_RAG_PATTERNS = 8
    MAX_NUMBER_OF_RAG_PATTERNS_ALLOWED_RANGE = (4, 20)
    METRIC = "faithfulness"
    SUPPORTED_OPTIMIZATION_METRICS = frozenset({"faithfulness", "answer_correctness", "context_correctness"})
    _ssl_logger = logging.getLogger(__name__)

    def _create_ogx_client(base_url, api_key) -> OgxClient:
        """Creates OgxClient.

        For the time being (temporarily), if self-signed certificate is detected in the certificates chain, then
        OGXClient is created with `verify=False` option making it (insecurely) NOT validate the server-side certificate.

        Args:
            base_url: URL pointing to OGX server.
            api_key: API Key to initialise the OGXClient instance with.
        """
        try:
            httpx.get(base_url)
        except httpx.ConnectError as e:
            if search(r"\bself.*signed.*certificate\b", str(e)):
                _ssl_logger.info("OGX server presents a self-signed certificate")
                if httpx.get(base_url, verify=False).status_code != 200:
                    _ssl_logger.error(
                        "Cannot establish connection with the OGX server even without "
                        "verification of the self-signed certificate.",
                        exc_info=True,
                    )
                    raise e
                _ssl_logger.warning("Initialising OGXClient without server-side certificate verification.")
                return OgxClient(base_url=base_url, api_key=api_key, http_client=httpx.Client(verify=False))

        return OgxClient(base_url=base_url, api_key=api_key)

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
                "ogx_indexing_template.ipynb",
                "ogx_inference_template.ipynb",
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
            with open(Path(embedded_artifact.path) / "notebook_templates" / notebook_name, "r") as f:
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
    ) -> dict[str, Any]:
        """Create a mapping from placeholder names to their values from output.json.

        This function extracts values from the output.json structure and creates
        a flat dictionary suitable for use with NotebookCell.format_source().

        Expected output.json structure:
        {
            "config": {
                "pattern_name": "...",
                "autorag_version": "...",
                "ogx": {
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
        # Hybrid search parameters (optional, None if not present)
        mapping["SEARCH_MODE"] = ret.get("search_mode")
        mapping["RANKER_STRATEGY"] = ret.get("ranker_strategy")
        mapping["RANKER_K"] = ret.get("ranker_k")
        mapping["RANKER_ALPHA"] = ret.get("ranker_alpha")

        ch = settings.get("chunking", {})
        mapping["CHUNKING_METHOD"] = ch.get("method", "")
        mapping["CHUNK_SIZE"] = ch.get("chunk_size", 512)
        mapping["CHUNK_OVERLAP"] = ch.get("chunk_overlap", 50)

        mapping["TEST_DATA_KEY"] = test_data_key
        mapping["INPUT_DATA_KEY"] = input_data_key

        return mapping

    def generate_notebook_from_templates(
        notebook_template: Literal[
            "ogx_inference",
            "ogx_indexing",
        ],
        output_data: dict[str, Any],
        output_notebook_path: Path,
        test_data_key: str = "",
        input_data_key: str = "",
    ) -> None:
        """Generate a filled notebook from templates and output.json.

        Args:
            notebook_template: One of the allowed template names.
            output_data: The parsed output.json data.
            output_notebook_path: Path where to save the generated notebook.
            test_data_key: Path to test data file within bucket used as input to AI4RAG.
            input_data_key: Path to documents dir within bucket used as input to AI4RAG.

        Returns:
            None. The notebook is written to output_notebook_path.
        """
        placeholder_mapping = create_placeholder_mapping(
            output_data,
            test_data_key=test_data_key,
            input_data_key=input_data_key,
        )
        notebook = Notebook.load(notebook_name=f"{notebook_template}_template.ipynb")
        filled_cells = []
        for cell in notebook.cells:
            filled_cell = cell.format_source(placeholder_mapping)
            filled_cells.append(filled_cell)

        notebook = Notebook(cells=filled_cells)

        notebook.save(Path(output_notebook_path))

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

    ogx_client_base_url = (os.environ.get("OGX_CLIENT_BASE_URL") or "").strip()
    ogx_client_api_key = (os.environ.get("OGX_CLIENT_API_KEY") or "").strip()

    if not ogx_client_base_url or not ogx_client_api_key:
        raise ValueError(
            "OGX_CLIENT_BASE_URL and OGX_CLIENT_API_KEY environment variables must be set to non-empty values."
        )

    client = _create_ogx_client(ogx_client_base_url, ogx_client_api_key)

    def construct_model_instance(loader, node: yml.MappingNode) -> BaseEmbeddingModel | BaseFoundationModel:
        """Instructs yml.Loader on how to construct "!Model" tag."""
        mapping = loader.construct_mapping(node, deep=True)

        match mapping:
            case {"type_": "embedding", **id_to_params}:
                model_id, params = id_to_params.popitem()
                return OGXEmbeddingModel(client=client, model_id=model_id, params=params)

            case {"type_": "generation", **id_to_params}:
                model_id, params = id_to_params.popitem()
                return OGXFoundationModel(client=client, model_id=model_id, params=params)
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

    if not isinstance(vector_io_provider_id, str) or not vector_io_provider_id.strip():
        raise ValueError("vector_io_provider_id must be a non-empty string.")
    vector_io_provider_id = vector_io_provider_id.strip()

    rag_exp = AI4RAGExperiment(
        client=client,
        event_handler=KFPEventHandler(),
        optimizer_settings=optimizer_settings,
        search_space=search_space,
        benchmark_data=benchmark_data,
        vector_store_type="ogx",
        documents=documents,
        optimization_metric=optimization_metric,
        ogx_vector_io_provider_id=vector_io_provider_id,
        # TODO some necessary kwargs (if any at all)
    )

    # retrieve documents && run optimisation loop
    rag_exp.search()

    rag_patterns_dir = Path(rag_patterns.path)

    rag_patterns.metadata["name"] = "rag_patterns_artifact"
    rag_patterns.metadata["uri"] = rag_patterns.uri
    rag_patterns.metadata["metadata"] = {"patterns": []}

    for pattern_data in rag_exp.event_handler.patterns:
        patt_dir = rag_patterns_dir / pattern_data["payload"]["pattern_name"]
        patt_dir.mkdir(parents=True, exist_ok=True)

        generate_notebook_from_templates(
            "ogx_indexing",
            pattern_data["payload"],
            Path(patt_dir, "indexing.ipynb"),
            input_data_key=input_data_key,
        )

        generate_notebook_from_templates(
            "ogx_inference",
            pattern_data["payload"],
            Path(patt_dir, "inference.ipynb"),
            test_data_key=test_data_key,
        )

        rag_patterns.metadata["metadata"]["patterns"].append(pattern_data["payload"])

        with (patt_dir / "pattern.json").open("w+", encoding="utf-8") as f:
            json_dump(pattern_data["payload"], f, indent=2)

        template_context = {
            "response_template": pattern_data["payload"]["settings"]["responses_template"],
        }
        with (Path(embedded_artifact.path) / "script_templates" / "create_model_response.py.templ").open(
            "r", encoding="utf-8"
        ) as f:
            model_responses_templ = Template(f.read())
            with (patt_dir / "create_model_response.py").open("w+", encoding="utf-8") as ff:
                ff.write(model_responses_templ.substitute(template_context))

        with (patt_dir / "evaluation_results.json").open("w+", encoding="utf-8") as f:
            json_dump(pattern_data["evaluation_results"], f, indent=2)

    # TODO autorag_run_artifact


temp_embedded_artifacts_dir.cleanup()

if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        rag_templates_optimization,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
