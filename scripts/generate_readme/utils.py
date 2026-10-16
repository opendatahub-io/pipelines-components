"""Utility functions for README generation."""

import re
import textwrap

from scripts.generate_readme.constants import MAX_LINE_LENGTH, SPECIAL_CASE_WORDS


def wrap_text(text: str, width: int = MAX_LINE_LENGTH) -> str:
    """Wrap text to specified width while preserving paragraph breaks.

    Args:
        text: The text to wrap.
        width: Maximum line width.

    Returns:
        Wrapped text with preserved paragraph structure.
    """
    if not text:
        return text

    # Split into paragraphs (separated by blank lines)
    paragraphs = text.split("\n\n")
    wrapped_paragraphs = []

    for paragraph in paragraphs:
        # Remove existing line breaks within paragraph
        paragraph = " ".join(paragraph.split())
        # Wrap to width
        wrapped = textwrap.fill(paragraph, width=width, break_long_words=False, break_on_hyphens=False)
        wrapped_paragraphs.append(wrapped)

    return "\n\n".join(wrapped_paragraphs)


def format_title(title: str) -> str:
    """Format a title from snake_case, kebab-case, or camelCase to Title Case.

    Args:
        title: The title to format.

    Returns:
        Formatted title in Title Case with spaces.
    """
    # First, handle camelCase by inserting spaces before capitals
    title = re.sub(r"([a-z])([A-Z])", r"\1 \2", title)

    # Replace underscores and hyphens with spaces
    title = title.replace("_", " ").replace("-", " ")

    # Split into words and capitalize each, preserving known special-case words
    words = title.split()
    formatted_words = [SPECIAL_CASE_WORDS.get(word.lower(), word.capitalize()) for word in words]

    return " ".join(formatted_words)
