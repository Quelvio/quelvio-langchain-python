"""Single source of truth for the package version.

Kept in a dedicated module so it can be imported without triggering the
heavier imports in ``__init__`` (User-Agent construction, etc.).
"""

__version__ = "0.1.0"
