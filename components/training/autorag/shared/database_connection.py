"""Database connection contract shared by AutoRAG pipeline compilers."""

from .runtime_embed.database_connection import SECRET_KEY_TO_ENV, normalize_connection

__all__ = ["SECRET_KEY_TO_ENV", "normalize_connection"]
