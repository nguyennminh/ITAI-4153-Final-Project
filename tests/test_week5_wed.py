"""Week 5 Wednesday tests: the latency summary, and what the Dockerfile must contain.

# DO NOT EDIT: do not change, skip, or loosen these tests to make them pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.measure_latency import summarize

ROOT = Path(__file__).resolve().parents[1]


def dockerfile() -> list[str]:
    """The Dockerfile's instructions, without comments or blank lines."""
    lines = (ROOT / "Dockerfile").read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


# ---------------------------------------------------------------- summarize

def test_summarize_hand_example() -> None:
    times = [float(t) for t in range(1, 101)]  # 1, 2, ..., 100 ms
    summary = summarize(times)
    assert summary["n"] == 100
    assert summary["p50_ms"] == pytest.approx(50.5)
    assert summary["p95_ms"] == pytest.approx(95.05)
    assert summary["max_ms"] == 100.0
    assert summary["mean_ms"] == pytest.approx(50.5)


def test_summarize_returns_plain_numbers() -> None:
    summary = summarize([10.0, 20.0, 30.0])
    assert isinstance(summary["n"], int)
    assert all(type(summary[k]) is float for k in ("p50_ms", "p95_ms", "max_ms", "mean_ms"))


def test_summarize_refuses_nothing() -> None:
    with pytest.raises(ValueError):
        summarize([])


# ---------------------------------------------------------------- the Dockerfile

def test_dockerfile_installs_the_pinned_requirements() -> None:
    lines = dockerfile()
    assert any(line.startswith("COPY") and "requirements-api.txt" in line for line in lines)
    assert any(line.startswith("RUN") and "-r requirements-api.txt" in line for line in lines)
    install = next(i for i, line in enumerate(lines) if "-r requirements-api.txt" in line)
    code = next(i for i, line in enumerate(lines)
                if line.startswith("COPY") and "api/" in line)
    assert install < code, "install the requirements before copying the code (layer cache)"


def test_dockerfile_runs_as_a_user_without_root_rights() -> None:
    users = [line.split()[1] for line in dockerfile() if line.startswith("USER")]
    assert users, "add a USER instruction"
    assert users[-1] not in ("root", "0"), "the last USER must not be root"


def test_dockerfile_starts_uvicorn_on_port() -> None:
    cmd = [line for line in dockerfile() if line.startswith(("CMD", "ENTRYPOINT"))]
    assert cmd, "add a CMD that starts the service"
    assert "uvicorn" in cmd[-1] and "api.main:app" in cmd[-1]
    assert "0.0.0.0" in cmd[-1], "listen on 0.0.0.0, or nothing outside the container reaches it"
    assert "PORT" in cmd[-1], "use the PORT that Cloud Run sets"


def test_requirements_api_are_pinned() -> None:
    path = ROOT / "requirements-api.txt"
    if not path.exists():
        pytest.skip("requirements-api.txt not written yet (python run.py export)")
    packages = [line for line in path.read_text().splitlines()
                if line.strip() and not line.startswith("#")]
    assert packages and all("==" in line for line in packages), "every package needs =="
