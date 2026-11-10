from __future__ import annotations

import time
from typing import Any, Callable, Dict, List
import numpy as np
import torch


def measure_verifier_latency(
    verifier: Any,
    pairs: List[tuple[str, str]],
    num_warmup: int = 2,
) -> Dict[str, Any]:
    """Measures single-answer (batch size = 1) latency metrics excluding warmup."""
    latencies: List[float] = []

    # Warmup runs
    for i in range(min(num_warmup, len(pairs))):
        ctx, ans = pairs[i]
        verifier.verify(ctx, ans)

    for ctx, ans in pairs:
        t0 = time.perf_counter()
        verifier.verify(ctx, ans)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

    lat_arr = np.array(latencies)
    return {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "p50_ms": float(np.percentile(lat_arr, 50)) if len(lat_arr) > 0 else 0.0,
        "p95_ms": float(np.percentile(lat_arr, 95)) if len(lat_arr) > 0 else 0.0,
        "mean_ms": float(np.mean(lat_arr)) if len(lat_arr) > 0 else 0.0,
        "std_ms": float(np.std(lat_arr)) if len(lat_arr) > 0 else 0.0,
    }