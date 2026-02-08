import os
from typing import Any, Mapping, Optional


def resolve_num_workers(config: Optional[Mapping[str, Any]] = None, *, default: int = 0) -> int:
    """Resolve DataLoader num_workers.

    Precedence:
    1) Explicit config['num_workers'] if present and not None
    2) env MEBENCH_NUM_WORKERS
    3) default
    """
    if config is not None and "num_workers" in config and config.get("num_workers") is not None:
        try:
            return int(config.get("num_workers"))
        except (TypeError, ValueError):
            return int(default)

    raw = os.environ.get("MEBENCH_NUM_WORKERS")
    if raw is None:
        return int(default)
    raw = str(raw).strip()
    if raw == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def resolve_prefetch_factor(*, default: int = 2) -> int:
    raw = os.environ.get("MEBENCH_PREFETCH_FACTOR")
    if raw is None:
        return int(default)
    raw = str(raw).strip()
    if raw == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def pool_loader_kwargs(device: str, config: Optional[Mapping[str, Any]] = None) -> dict:
    """Common DataLoader kwargs for pool scanning on GPU runs.

    This is intended for ImageFolder/Subset-backed loaders that are safe with
    multi-worker loading on Windows (i.e., top-level dataset types).
    """
    num_workers = resolve_num_workers(config, default=0)
    pin_memory = str(device).startswith("cuda")
    kwargs: dict = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if num_workers and int(num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(resolve_prefetch_factor(default=2))
    return kwargs
