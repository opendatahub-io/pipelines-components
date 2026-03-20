"""AST helpers to read static ``description`` from ``@dsl.pipeline`` in ``pipeline.py`` files."""

import ast
from pathlib import Path

from scripts.lib.kfp_compilation import extract_decorator_name


def _parse_python_file(file_path: Path) -> ast.AST:
    """Parse a Python file into an AST."""
    with open(file_path, encoding="utf-8") as f:
        source = f.read()
    return ast.parse(source)


def _concatenated_string_literal(node: ast.expr) -> str | None:
    """Resolve static string expressions used in decorator kwargs (literals and implicit concat).

    Handles ``ast.Constant`` strings and ``ast.BinOp`` with ``+`` joining other static strings.
    Returns None if the expression uses f-strings, variables, or other dynamic constructs.

    Args:
        node: AST expression node.

    Returns:
        The concatenated string, or None if not a fully static string literal.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _concatenated_string_literal(node.left)
        right = _concatenated_string_literal(node.right)
        if left is not None and right is not None:
            return left + right
        return None
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                return None
        return "".join(parts)
    return None


def _pipeline_decorator_node(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.expr | None:
    """Return the first @pipeline decorator node on the function, if any."""
    for decorator in func_node.decorator_list:
        if extract_decorator_name(decorator) == "pipeline":
            return decorator
    return None


def _description_from_pipeline_decorator(decorator: ast.expr) -> str | None:
    """Extract ``description`` from ``@dsl.pipeline(...)`` when it is a static string."""
    if not isinstance(decorator, ast.Call):
        return None
    for keyword in decorator.keywords:
        if keyword.arg != "description" or keyword.value is None:
            continue
        resolved = _concatenated_string_literal(keyword.value)
        if resolved is not None:
            return resolved.strip()
    return None


def _first_line_of_docstring(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """Return the first non-empty line of the function docstring, if any."""
    doc = ast.get_docstring(func_node, clean=True)
    if not doc:
        return None
    first = doc.strip().split("\n", maxsplit=1)[0].strip()
    return first or None


def extract_pipeline_description_from_file(
    file_path: Path,
    *,
    function_name: str | None = None,
) -> str | None:
    """Extract pipeline description from ``@dsl.pipeline(description=...)`` or the function docstring.

    Looks at top-level functions only. If ``function_name`` is set, prefers that function when it
    is decorated with ``@dsl.pipeline``; otherwise uses the first ``@dsl.pipeline`` function in the
    file.

    Precedence for the returned string:

    1. The ``description`` keyword in ``@dsl.pipeline(...)`` when it resolves to static string
       literals (including implicit concatenation).
    2. The first line of the function's docstring.

    Args:
        file_path: Path to ``pipeline.py``.
        function_name: Optional function name to match (e.g. metadata ``name``).

    Returns:
        Description text, or None if nothing could be extracted.
    """
    try:
        tree = _parse_python_file(file_path)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None

    top_functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = [
        n for n in getattr(tree, "body", []) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    def is_pipeline_fn(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        return _pipeline_decorator_node(fn) is not None

    candidates = [fn for fn in top_functions if is_pipeline_fn(fn)]
    if not candidates:
        return None

    chosen: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    if function_name:
        chosen = next((fn for fn in candidates if fn.name == function_name), None)
    if chosen is None:
        chosen = candidates[0]

    dec = _pipeline_decorator_node(chosen)
    if dec is None:
        return _first_line_of_docstring(chosen)

    from_decorator = _description_from_pipeline_decorator(dec)
    if from_decorator:
        return from_decorator
    return _first_line_of_docstring(chosen)
