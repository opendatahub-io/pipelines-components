"""Mock implementations of ai4rag classes for unit testing.

This module provides mock classes and a factory function to create a fake ai4rag
module hierarchy that can be injected into sys.modules, allowing tests to run
without the actual ai4rag library installed.
"""

import types
from unittest.mock import MagicMock


class BaseEmbeddingModel:
    """Mock base class for ai4rag embedding models.

    This class is used for isinstance checks in tests and YAML serialization.
    """

    pass


class BaseFoundationModel:
    """Mock base class for ai4rag foundation models.

    This class is used for isinstance checks in tests and YAML serialization.
    """

    pass


class BaseEventHandler:
    """Mock base class for ai4rag event handlers.

    This class supports inheritance in tests.
    """

    pass


class LogLevel:
    """Mock LogLevel enum for ai4rag event handlers.

    Provides the same attributes as the real LogLevel enum.
    """

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


def make_ai4rag_modules() -> dict[str, types.ModuleType]:
    """Create a fake ai4rag module hierarchy.

    Returns a dictionary of module names to module objects that can be injected
    into sys.modules to mock the ai4rag library.

    Returns:
        Dictionary mapping module names to mock module objects.
    """
    modules = {}

    # Create root module
    ai4rag = types.ModuleType("ai4rag")
    modules["ai4rag"] = ai4rag

    # Create ai4rag.core
    core = types.ModuleType("ai4rag.core")
    ai4rag.core = core
    modules["ai4rag.core"] = core

    # Create ai4rag.core.experiment
    experiment = types.ModuleType("ai4rag.core.experiment")
    core.experiment = experiment
    modules["ai4rag.core.experiment"] = experiment

    # Create ai4rag.core.experiment.benchmark_data
    benchmark_data = types.ModuleType("ai4rag.core.experiment.benchmark_data")
    benchmark_data.BenchmarkData = MagicMock(name="BenchmarkData")
    experiment.benchmark_data = benchmark_data
    modules["ai4rag.core.experiment.benchmark_data"] = benchmark_data

    # Create ai4rag.core.experiment.experiment
    experiment_module = types.ModuleType("ai4rag.core.experiment.experiment")
    experiment_module.AI4RAGExperiment = MagicMock(name="AI4RAGExperiment")
    experiment.experiment = experiment_module
    modules["ai4rag.core.experiment.experiment"] = experiment_module

    # Create ai4rag.core.experiment.mps
    mps = types.ModuleType("ai4rag.core.experiment.mps")
    mps.ModelsPreSelector = MagicMock(name="ModelsPreSelector")
    experiment.mps = mps
    modules["ai4rag.core.experiment.mps"] = mps

    # Create ai4rag.core.experiment.results
    results = types.ModuleType("ai4rag.core.experiment.results")
    results.ExperimentResults = MagicMock(name="ExperimentResults")
    experiment.results = results
    modules["ai4rag.core.experiment.results"] = results

    # Create ai4rag.core.hpo
    hpo = types.ModuleType("ai4rag.core.hpo")
    core.hpo = hpo
    modules["ai4rag.core.hpo"] = hpo

    # Create ai4rag.core.hpo.gam_opt
    gam_opt = types.ModuleType("ai4rag.core.hpo.gam_opt")
    gam_opt.GAMOptSettings = MagicMock(name="GAMOptSettings")
    hpo.gam_opt = gam_opt
    modules["ai4rag.core.hpo.gam_opt"] = gam_opt

    # Create ai4rag.rag
    rag = types.ModuleType("ai4rag.rag")
    ai4rag.rag = rag
    modules["ai4rag.rag"] = rag

    # Create ai4rag.rag.chunking
    chunking = types.ModuleType("ai4rag.rag.chunking")
    chunking.LangChainChunker = MagicMock(name="LangChainChunker")
    rag.chunking = chunking
    modules["ai4rag.rag.chunking"] = chunking

    # Create ai4rag.rag.embedding
    embedding = types.ModuleType("ai4rag.rag.embedding")
    rag.embedding = embedding
    modules["ai4rag.rag.embedding"] = embedding

    # Create ai4rag.rag.embedding.base_model
    embedding_base_model = types.ModuleType("ai4rag.rag.embedding.base_model")
    embedding_base_model.BaseEmbeddingModel = BaseEmbeddingModel
    embedding.base_model = embedding_base_model
    modules["ai4rag.rag.embedding.base_model"] = embedding_base_model

    # Create ai4rag.rag.embedding.openai_model
    openai_embedding = types.ModuleType("ai4rag.rag.embedding.openai_model")
    openai_embedding.OpenAIEmbeddingModel = MagicMock(name="OpenAIEmbeddingModel")
    embedding.openai_model = openai_embedding
    modules["ai4rag.rag.embedding.openai_model"] = openai_embedding

    # Create ai4rag.rag.embedding.llama_stack
    llama_stack_embedding = types.ModuleType("ai4rag.rag.embedding.llama_stack")
    llama_stack_embedding.LSEmbeddingModel = MagicMock(name="LSEmbeddingModel")
    llama_stack_embedding.LSEmbeddingParams = MagicMock(name="LSEmbeddingParams")
    embedding.llama_stack = llama_stack_embedding
    modules["ai4rag.rag.embedding.llama_stack"] = llama_stack_embedding

    # Create ai4rag.rag.foundation_models
    foundation_models = types.ModuleType("ai4rag.rag.foundation_models")
    rag.foundation_models = foundation_models
    modules["ai4rag.rag.foundation_models"] = foundation_models

    # Create ai4rag.rag.foundation_models.base_model
    foundation_base_model = types.ModuleType("ai4rag.rag.foundation_models.base_model")
    foundation_base_model.BaseFoundationModel = BaseFoundationModel
    foundation_models.base_model = foundation_base_model
    modules["ai4rag.rag.foundation_models.base_model"] = foundation_base_model

    # Create ai4rag.rag.foundation_models.openai_model
    openai_foundation = types.ModuleType("ai4rag.rag.foundation_models.openai_model")
    openai_foundation.OpenAIFoundationModel = MagicMock(name="OpenAIFoundationModel")
    foundation_models.openai_model = openai_foundation
    modules["ai4rag.rag.foundation_models.openai_model"] = openai_foundation

    # Create ai4rag.rag.foundation_models.llama_stack
    llama_stack_foundation = types.ModuleType("ai4rag.rag.foundation_models.llama_stack")
    llama_stack_foundation.LSFoundationModel = MagicMock(name="LSFoundationModel")
    foundation_models.llama_stack = llama_stack_foundation
    modules["ai4rag.rag.foundation_models.llama_stack"] = llama_stack_foundation

    # Create ai4rag.rag.vector_store
    vector_store = types.ModuleType("ai4rag.rag.vector_store")
    rag.vector_store = vector_store
    modules["ai4rag.rag.vector_store"] = vector_store

    # Create ai4rag.rag.vector_store.llama_stack
    llama_stack_vector_store = types.ModuleType("ai4rag.rag.vector_store.llama_stack")
    llama_stack_vector_store.LSVectorStore = MagicMock(name="LSVectorStore")
    vector_store.llama_stack = llama_stack_vector_store
    modules["ai4rag.rag.vector_store.llama_stack"] = llama_stack_vector_store

    # Create ai4rag.search_space
    search_space = types.ModuleType("ai4rag.search_space")
    ai4rag.search_space = search_space
    modules["ai4rag.search_space"] = search_space

    # Create ai4rag.search_space.prepare
    prepare = types.ModuleType("ai4rag.search_space.prepare")
    search_space.prepare = prepare
    modules["ai4rag.search_space.prepare"] = prepare

    # Create ai4rag.search_space.prepare.prepare_search_space
    prepare_search_space_module = types.ModuleType("ai4rag.search_space.prepare.prepare_search_space")
    prepare_search_space_module.prepare_search_space_with_llama_stack = MagicMock(
        name="prepare_search_space_with_llama_stack"
    )
    prepare.prepare_search_space = prepare_search_space_module
    modules["ai4rag.search_space.prepare.prepare_search_space"] = prepare_search_space_module

    # Create ai4rag.search_space.src
    src = types.ModuleType("ai4rag.search_space.src")
    search_space.src = src
    modules["ai4rag.search_space.src"] = src

    # Create ai4rag.search_space.src.parameter
    parameter = types.ModuleType("ai4rag.search_space.src.parameter")
    parameter.Parameter = MagicMock(name="Parameter")
    src.parameter = parameter
    modules["ai4rag.search_space.src.parameter"] = parameter

    # Create ai4rag.search_space.src.search_space
    search_space_module = types.ModuleType("ai4rag.search_space.src.search_space")
    search_space_module.AI4RAGSearchSpace = MagicMock(name="AI4RAGSearchSpace")
    src.search_space = search_space_module
    modules["ai4rag.search_space.src.search_space"] = search_space_module

    # Create ai4rag.utils
    utils = types.ModuleType("ai4rag.utils")
    ai4rag.utils = utils
    modules["ai4rag.utils"] = utils

    # Create ai4rag.utils.event_handler
    event_handler_pkg = types.ModuleType("ai4rag.utils.event_handler")
    utils.event_handler = event_handler_pkg
    modules["ai4rag.utils.event_handler"] = event_handler_pkg

    # Create ai4rag.utils.event_handler.event_handler
    event_handler_module = types.ModuleType("ai4rag.utils.event_handler.event_handler")
    event_handler_module.BaseEventHandler = BaseEventHandler
    event_handler_module.LogLevel = LogLevel
    event_handler_pkg.event_handler = event_handler_module
    modules["ai4rag.utils.event_handler.event_handler"] = event_handler_module

    return modules
