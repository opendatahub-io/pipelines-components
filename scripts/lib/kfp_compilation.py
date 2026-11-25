"""KFP module loading and compilation utilities."""

import ast
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

COMPONENT_DECORATORS = {"component", "container_component", "notebook_component"}
PIPELINE_DECORATORS = {"pipeline"}


def load_module_from_path(module_path: str, module_name: str) -> ModuleType:
    """Dynamically load a Python module from a file path.

    Args:
        module_path: File path to the Python module.
        module_name: Name to assign to the loaded module.

    Returns:
        The loaded module object.

    Raises:
        ImportError: If the module cannot be loaded.
    """
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _is_platform_spec(doc: dict[str, Any]) -> bool:
    """True if doc looks like a KFP platform spec (has top-level 'platforms' dict)."""
    return isinstance(doc.get("platforms"), dict)


def _is_pipeline_spec(doc: dict[str, Any]) -> bool:
    """True if doc looks like a KFP pipeline/component spec (has deploymentSpec or root)."""
    return "deploymentSpec" in doc or "root" in doc


def _load_compiled_yaml(path: str) -> dict[str, Any]:
    """Load compiled YAML from path; return single doc or two-doc wrapper.

    Uses safe_load_all and filters to dicts. One doc -> return it; two docs ->
    classify by content: doc with top-level 'platforms' -> platform_spec, doc with
    'deploymentSpec' or 'root' -> pipeline_spec. If we cannot classify both
    unambiguously, raise ValueError (no fallback).
    """
    with open(path) as f:
        docs = [d for d in yaml.safe_load_all(f) if isinstance(d, dict)]
    if not docs:
        raise ValueError(
            f"Compiled YAML at {path} has no dict document. Expected at least one pipeline/component spec."
        )
    if len(docs) == 1:
        return docs[0]
    # Two docs: identify by content only.
    pipeline_spec = next((d for d in docs if _is_pipeline_spec(d) and not _is_platform_spec(d)), None)
    platform_spec = next((d for d in docs if _is_platform_spec(d)), None)
    if pipeline_spec is not None and platform_spec is not None and pipeline_spec is not platform_spec:
        return {"pipeline_spec": pipeline_spec, "platform_spec": platform_spec}
    raise ValueError(
        f"Compiled YAML at {path} has two documents but could not classify them: "
        "expected one doc with 'deploymentSpec' or 'root' (pipeline spec) and one with "
        "'platforms' (platform spec). Refusing to guess."
    )


def compile_and_get_yaml(func: Any, output_path: str) -> dict[str, Any]:
    """Compile a component or pipeline function and return the parsed YAML.

    Uses safe_load_all. Two docs (pipeline + platform spec) are returned separately;
    key layout differs so callers handle each via get_base_images_from_compile_result.

    Args:
        func: The KFP component or pipeline function to compile.
        output_path: Path to write the compiled YAML.

    Returns:
        Single dict (one doc) or {"pipeline_spec": ..., "platform_spec": ...} (two docs).

    Raises:
        ValueError: If the compiled YAML contains no dict document.
        Exception: If compilation fails.
    """
    compiler_mod = importlib.import_module("kfp.compiler")
    compiler_class = getattr(compiler_mod, "Compiler")
    compiler_class().compile(func, output_path)
    return _load_compiled_yaml(output_path)


def find_decorated_functions_runtime(module: Any, decorator_type: str) -> list[tuple[str, Any]]:
    """Find all functions decorated with @dsl.component or @dsl.pipeline at runtime.

    Args:
        module: The loaded Python module.
        decorator_type: Either 'component' or 'pipeline'.

    Returns:
        List of tuples (function_name, function_object).
    """
    functions = []
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        attr = getattr(module, attr_name, None)
        if attr is None or not callable(attr):
            continue
        is_component = hasattr(attr, "component_spec") or (
            getattr(attr, "__wrapped__", None) is not None and hasattr(getattr(attr, "__wrapped__"), "component_spec")
        )
        is_pipeline = hasattr(attr, "pipeline_spec") or getattr(attr, "_pipeline_func", None) is not None
        is_match = (decorator_type == "component" and is_component) or (decorator_type == "pipeline" and is_pipeline)
        if is_match:
            functions.append((attr_name, attr))
    return functions


def find_decorated_function_names_ast(file_path: Path) -> dict[str, list[str]]:
    """Find functions decorated with KFP decorators in a Python file.

    Args:
        file_path: Path to the Python file to analyze.

    Returns:
        A dict mapping decorator type to list of function names:
        {"components": [...], "pipelines": [...]}
        Returns empty dict {} if the file cannot be parsed.
    """
    try:
        content = file_path.read_text()
        tree = ast.parse(content)
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"  Warning: Could not parse {file_path}: {e}")
        return {}

    result: dict[str, list[str]] = {"components": [], "pipelines": []}

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                decorator_name = extract_decorator_name(decorator)
                if decorator_name is not None and decorator_name in COMPONENT_DECORATORS:
                    result["components"].append(node.name)
                    break
                if decorator_name is not None and decorator_name in PIPELINE_DECORATORS:
                    result["pipelines"].append(node.name)
                    break

    return result


def extract_decorator_name(decorator: ast.expr) -> str | None:
    """Extract the name from a decorator AST node.

    Handles @name, @module.name, and @name() call forms.

    Args:
        decorator: AST node representing the decorator.

    Returns:
        The decorator name, or None if it cannot be determined.
    """
    if isinstance(decorator, ast.Name):
        return decorator.id
    if isinstance(decorator, ast.Attribute):
        return decorator.attr
    if isinstance(decorator, ast.Call):
        if isinstance(decorator.func, ast.Name):
            return decorator.func.id
        if isinstance(decorator.func, ast.Attribute):
            return decorator.func.attr
    return None
