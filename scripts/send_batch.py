"""Send one batch of production applicants through your service, as the loan screen would.

# PROVIDED: this whole script is finished.

Run it with ``python run.py send --batch 1 --url https://YOUR-SERVICE.run.app`` (or without
--url for the service on your laptop). It reads data/production/batchN.jsonl (written by
``python run.py pool``), sends every request to /predict, a few at a time, and writes a
short receipt to reports/sent_batchN.json: how many were sent, how many answered 200, and how
long it took. The predictions themselves are in your service's log, not in the receipt.

Send each batch ONCE. Sending a batch twice logs every applicant twice (the monitoring keeps
only the latest answer per applicant and batch, but your receipt would no longer match).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx

PRODUCTION_DIR = Path("data/production")
REPORTS = Path("reports")


# PROVIDED
def read_batch(batch: int) -> list[dict]:
    path = PRODUCTION_DIR / f"batch{batch}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found: run python run.py pool first")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# PROVIDED
def send(url: str, requests: list[dict], workers: int) -> tuple[Counter, float]:
    """POST every request; return the count of each HTTP status and the seconds taken."""
    statuses: Counter = Counter()
    with httpx.Client(timeout=60) as client:
        def post(payload: dict) -> int:
            for attempt in range(3):  # a cold start or a brief overload: try again
                try:
                    status = client.post(f"{url}/predict", json=payload).status_code
                except httpx.HTTPError:
                    status = 0
                if status not in (0, 429, 500, 502, 503, 504):
                    return status
                time.sleep(1 + attempt)
            return status

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, status in enumerate(pool.map(post, requests), start=1):
                statuses[status] += 1
                if i % 200 == 0:
                    print(f"  {i:,} / {len(requests):,} sent", flush=True)
    return statuses, time.perf_counter() - start


# PROVIDED
def main() -> None:
    parser = argparse.ArgumentParser(description="Send a production batch to the service.")
    parser.add_argument("--batch", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="the service's base URL")
    parser.add_argument("--workers", type=int, default=8, help="requests in flight at once")
    args = parser.parse_args()
    url = args.url.rstrip("/")
    requests = read_batch(args.batch)

    print(f"Sending batch {args.batch} ({len(requests):,} applicants) to {url} ...")
    statuses, seconds = send(url, requests, args.workers)
    receipt = {"batch": args.batch, "url": url, "sent": len(requests),
               "ok": statuses.get(200, 0),
               "statuses": {str(k): v for k, v in sorted(statuses.items())},
               "seconds": round(seconds, 1),
               "sent_at": datetime.now(UTC).isoformat(timespec="seconds")}
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"sent_batch{args.batch}.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"{receipt['ok']:,} of {receipt['sent']:,} answered 200 in {seconds:.0f} s "
          f"(statuses {receipt['statuses']}). Wrote {out}")
    if receipt["ok"] < receipt["sent"]:
        print("Some requests failed: read the service's log (Cloud Run -> Logs) before going on.")


if __name__ == "__main__":
    main()
