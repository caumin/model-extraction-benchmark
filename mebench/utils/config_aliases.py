"""Helpers for backward-compatible config key aliases."""

from __future__ import annotations

import warnings
from typing import Mapping, Any


def resolve_iterations(
    config: Mapping[str, Any],
    *,
    default: int,
    context: str,
    allow_num_rounds: bool = False,
) -> int:
    """Resolve canonical loop-count key with compatibility aliases.

    Canonical key is ``iterations``.
    Legacy aliases: ``num_rounds`` (optional) and ``rounds``.
    """

    if config.get("iterations") is not None:
        return int(config["iterations"])

    if allow_num_rounds and config.get("num_rounds") is not None:
        warnings.warn(
            f"{context}: 'num_rounds' is deprecated; use 'iterations' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return int(config["num_rounds"])

    if config.get("rounds") is not None:
        warnings.warn(
            f"{context}: 'rounds' is deprecated; use 'iterations' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return int(config["rounds"])

    return int(default)


def resolve_query_budget(config: Mapping[str, Any], *, default: int, context: str) -> int:
    """Resolve canonical query budget key with compatibility aliases.

    Canonical key is ``query_budget``.
    Legacy aliases: ``querybudget`` and ``budget``.
    """

    if config.get("query_budget") is not None:
        return int(config["query_budget"])

    if config.get("querybudget") is not None:
        warnings.warn(
            f"{context}: 'querybudget' is deprecated; use 'query_budget' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return int(config["querybudget"])

    if config.get("budget") is not None:
        warnings.warn(
            f"{context}: attack-level 'budget' is deprecated; use 'query_budget' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return int(config["budget"])

    return int(default)


def resolve_nominal_query_budget(config: Mapping[str, Any], *, default: int, context: str) -> int:
    """Resolve nominal query budget key with compatibility alias."""

    if config.get("nominal_query_budget") is not None:
        return int(config["nominal_query_budget"])

    if config.get("nominal_querybudget") is not None:
        warnings.warn(
            f"{context}: 'nominal_querybudget' is deprecated; use 'nominal_query_budget' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return int(config["nominal_querybudget"])

    return int(default)
