"""How fast does the service answer? Typical and worst-case response times.

Run it with ``python run.py latency --url https://YOUR-SERVICE.run.app`` (or without --url
for the service on your laptop). It:

1. sends ONE request first and times it on its own: if the service had no running instance
   (Cloud Run scales to zero after about 15 minutes without traffic), that is a **cold
   start**: a new container starts and loads the model before it can answer;
2. then sends ``--n`` requests one after another for real applicants and times each;
3. summarizes them (your ``summarize``) and writes reports/latency.json.

p50 is the typical answer time (half the requests are faster). p95 is what the slowest 1 in
20 requests waits: users notice the slow ones, so p95 matters more than the average.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

STORE = Path("models/feature_store.parquet")
REPORT = Path("reports/latency.json")


# TODO(student)
def summarize(times_ms: list[float]) -> dict[str, float]:
    """Summary of response times in milliseconds.

    Returns a dict with: ``n`` (how many), ``p50_ms`` and ``p95_ms`` (the 50th and 95th
    percentiles), ``max_ms``, and ``mean_ms``, all as plain floats (``n`` as an int).

    Hints:
        - ``np.percentile(times_ms, [50, 95])``.
        - An empty list is a mistake by the caller: raise a ValueError.
    """
    if not times_ms:
        raise ValueError("no response times to summarize")
    p50, p95 = np.percentile(times_ms, [50, 95])
    return {"n": len(times_ms), "p50_ms": float(p50), "p95_ms": float(p95),
            "max_ms": float(max(times_ms)), "mean_ms": float(np.mean(times_ms))}


# PROVIDED
def timed_post(client: httpx.Client, url: str, payload: dict) -> tuple[float, int]:
    """Send one /predict request; return (milliseconds, HTTP status)."""
    start = time.perf_counter()
    response = client.post(f"{url}/predict", json=payload)
    return (time.perf_counter() - start) * 1000, response.status_code


# PROVIDED
def measure(url: str, n: int, seed: int = 0) -> dict:
    """The cold-start request, then ``n`` timed requests for random real applicants."""
    if not STORE.exists():
        raise SystemExit(f"{STORE} not found: run python run.py export first")
    ids = pd.read_parquet(STORE, columns=["SK_ID_CURR"])["SK_ID_CURR"]
    chosen = ids.sample(n + 1, replace=True, random_state=seed).tolist()
    with httpx.Client(timeout=60) as client:
        first_ms, first_status = timed_post(client, url, {"SK_ID_CURR": int(chosen[0])})
        results = [timed_post(client, url, {"SK_ID_CURR": int(i)}) for i in chosen[1:]]
    times = [ms for ms, status in results if status == 200]
    errors = sum(status != 200 for _, status in results) + (first_status != 200)
    return {"url": url, "first_request_ms": round(first_ms, 1), "errors": errors,
            **summarize(times), "created": datetime.now(UTC).isoformat(timespec="seconds")}


# PROVIDED
def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the service's response times.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="the service's base URL")
    parser.add_argument("--n", type=int, default=100, help="timed requests after the first")
    args = parser.parse_args()
    url = args.url.rstrip("/")
    record = measure(url, args.n)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{url}\n  first request (a cold start if no instance was running): "
          f"{record['first_request_ms']:.0f} ms")
    print(f"  {record['n']} requests: p50 {record['p50_ms']:.0f} ms, p95 {record['p95_ms']:.0f} "
          f"ms, max {record['max_ms']:.0f} ms, errors {record['errors']}")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
