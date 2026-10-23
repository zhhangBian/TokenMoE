"""Lightweight observation summaries without dataset/parquet imports."""

import numpy as np


def distribution(values):
    values = [float(x) for x in values if x is not None]
    return {
        "count": len(values),
        "p50": float(np.percentile(values, 50)) if values else None,
        "p95": float(np.percentile(values, 95)) if values else None,
        "reason": None if values else "No observed values with complete endpoints",
    }
