"""Importing this package registers every adapter module found next to base.py."""

import importlib
import pkgutil

from .base import REGISTRY, Source, get_source, make_row, register, take

for module in pkgutil.iter_modules(__path__):
    if module.name != "base":
        importlib.import_module(f"{__name__}.{module.name}")

__all__ = ["REGISTRY", "Source", "get_source", "make_row", "register", "take"]
