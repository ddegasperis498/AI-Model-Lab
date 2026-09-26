from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover - backend may be inspected without torch import succeeding
    torch = None


def sanitize_for_json(value: Any) -> Any:
    """Return a JSON-safe representation.

    NaN and +/-Infinity are represented as ``None``. Tensor/scalar values are
    converted to normal Python values. The function is deliberately recursive
    because telemetry payloads contain nested lists/dicts.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if torch is not None and torch.is_tensor(value):
        if value.numel() == 1:
            return sanitize_for_json(value.detach().cpu().item())
        return sanitize_for_json(value.detach().cpu().tolist())
    if hasattr(value, 'item') and callable(value.item):
        try:
            return sanitize_for_json(value.item())
        except Exception:
            pass
    if is_dataclass(value):
        return sanitize_for_json(asdict(value))
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(v) for v in value]
    return value
