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


def resolve_persistent_workers(
    config: Optional[Mapping[str, Any]] = None,
    *,
    default: bool = False,
) -> bool:
    if config is not None and "persistent_workers" in config:
        return bool(config.get("persistent_workers"))

    raw = os.environ.get("MEBENCH_PERSISTENT_WORKERS")
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


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
    if num_workers and int(num_workers) > 0 and resolve_persistent_workers(config, default=False):
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(resolve_prefetch_factor(default=2))
    return kwargs


def resolve_pool_num_workers(
    attack_config: Optional[Mapping[str, Any]] = None,
    dataset_config: Optional[Mapping[str, Any]] = None,
    *,
    default: Optional[int] = None,
) -> Optional[int]:
    """Resolve DataLoader workers for pool scanning/scoring.

    Precedence:
    1) attack_config['pool_num_workers']
    2) attack_config['num_workers']
    3) dataset_config['num_workers']
    4) default

    Note: If this returns None, callers should omit explicit override and let
    `pool_loader_kwargs()` fall back to env/default behavior.
    """

    if attack_config is not None:
        raw = attack_config.get("pool_num_workers")
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                return default

        raw = attack_config.get("num_workers")
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                return default

    if dataset_config is not None:
        raw = dataset_config.get("num_workers")
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                return default

    return default


def resolve_train_num_workers(
    substitute_config: Optional[Mapping[str, Any]] = None,
    attack_config: Optional[Mapping[str, Any]] = None,
    *,
    default: Optional[int] = None,
) -> Optional[int]:
    """Resolve DataLoader workers for substitute training loaders.

    Precedence:
    1) substitute_config['train_num_workers']
    2) substitute_config['num_workers']
    3) attack_config['train_num_workers'] (legacy)
    4) attack_config['num_workers']
    5) default
    """

    if substitute_config is not None:
        raw = substitute_config.get("train_num_workers")
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                return default

        raw = substitute_config.get("num_workers")
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                return default

    if attack_config is not None:
        raw = attack_config.get("train_num_workers")
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                return default

        raw = attack_config.get("num_workers")
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                return default

    return default


def resolve_val_num_workers(
    substitute_config: Optional[Mapping[str, Any]] = None,
    attack_config: Optional[Mapping[str, Any]] = None,
    *,
    default: Optional[int] = None,
) -> Optional[int]:
    """Resolve DataLoader workers for substitute validation loaders.

    Precedence:
    1) substitute_config['val_num_workers']
    2) substitute_config['num_workers']
    3) attack_config['val_num_workers'] (legacy)
    4) attack_config['num_workers']
    5) default
    """

    if substitute_config is not None:
        raw = substitute_config.get("val_num_workers")
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                return default

        raw = substitute_config.get("num_workers")
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                return default

    if attack_config is not None:
        raw = attack_config.get("val_num_workers")
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                return default

        raw = attack_config.get("num_workers")
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                return default

    return default


def load_pool_to_memory(
    config: Mapping[str, Any],
    device: str,
    max_samples: int = 100_000,
    desc: str = "Loading pool to RAM",
) -> Any:
    """Load entire pool dataset into a single tensor on CPU by default.

    Args:
        config: Dataset config dict
        device: Runtime device (e.g. 'cuda:0' or 'cpu').
                Cached tensors stay on CPU unless config explicitly opts into
                device-resident caching via ``cache_device``.
        max_samples: Maximum number of samples to load.
        desc: Progress bar description.

    Returns:
        torch.Tensor: Shape (N, C, H, W)
    """
    from mebench.data.loaders import create_dataloader
    from tqdm import tqdm
    import torch

    # Create a loader specifically for efficient sequential reading
    # Use larger batch size for throughput
    loader = create_dataloader(
        dict(config),
        batch_size=512,
        shuffle=False,
    )
    
    # We can't rely on len(loader) if it's dynamic/infinite, but here it's fine.
    # Just collect until max_samples.
    batches = []
    total = 0
    
    for x, _ in tqdm(loader, desc=desc, leave=False):
        batches.append(x)
        total += x.size(0)
        if total >= max_samples:
            break
            
    if not batches:
        return torch.empty(0)
        
    full_tensor = torch.cat(batches, dim=0)
    if full_tensor.size(0) > max_samples:
        full_tensor = full_tensor[:max_samples]
        
    cache_device = str(config.get("cache_device", "cpu")).strip().lower()
    if cache_device in {"runtime", "device", "same_as_runtime", "auto"}:
        cache_target = str(device)
    else:
        cache_target = cache_device

    if cache_target.startswith("cuda"):
        try:
            return full_tensor.to(cache_target)
        except RuntimeError:
            # Fallback to CPU if VRAM not enough
            return full_tensor.cpu()
            
    return full_tensor
