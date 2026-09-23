"""Constants and shared configuration for the generate_readme package."""

# Custom content marker for preserving user-added content
CUSTOM_CONTENT_MARKER = "<!-- custom-content -->"

# README Templates
CATEGORY_README_TEMPLATE = "CATEGORY_README.md.j2"
SUBCATEGORY_README_TEMPLATE = "SUBCATEGORY_README.md.j2"
README_TEMPLATE = "README.md.j2"

# Exit codes
EXIT_SUCCESS = 0  # All files in sync (check mode) or successfully updated (fix mode)
EXIT_DIFF_DETECTED = 1  # Diffs detected in check mode
EXIT_ERROR = 2  # Actual error (e.g., missing function, failed to write file)

# Markdown formatting constraints (per .markdownlint.json)
MAX_LINE_LENGTH = 300  # Maximum line length for Markdown content

# Words with non-standard capitalization that should be preserved as-is
# instead of being title-cased. Keys must be lowercase.
SPECIAL_CASE_WORDS = {
    "kfp": "KFP",
    "api": "API",
    "url": "URL",
    "id": "ID",
    "ui": "UI",
    "ci": "CI",
    "cd": "CD",
    "automl": "AutoML",
    "autorag": "AutoRAG",
}
