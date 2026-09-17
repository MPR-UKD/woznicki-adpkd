"""HTTP API wrapping the fit.sh segmentation pipeline.

Lets an external caller (e.g. a nomiso plugin) submit a segmentation job,
poll its status, and retrieve the results.json produced by
postprocess_masks.py, without needing to exec into the container or use
the CLI directly. The CLI (fit.sh) keeps working unchanged; this module
only wraps it.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

API_VERSION = "1.0.0"

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data/adpkd")).resolve()
INPUT_ROOT = (DATA_ROOT / "inputs").resolve()
OUTPUT_ROOT = (DATA_ROOT / "outputs").resolve()
SOURCE_DIR = Path(os.environ.get("SOURCE_DIR", "/workspace/source")).resolve()
RESULTS_FOLDER = Path(
    os.environ.get("RESULTS_FOLDER", str(SOURCE_DIR / "trained_models"))
).resolve()


class SubmitJobRequest(BaseModel):
    """Request body for POST /jobs."""

    input_path: str
    small: bool = False
    cpu: bool = False


class Job(BaseModel):
    """Tracked state of a single segmentation job."""

    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: str
    output_path: str
    error: str | None = None


_executor = ThreadPoolExecutor(max_workers=1)  # serialize: one GPU, one job at a time
_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()

app = FastAPI(title="adpkd-net", version=API_VERSION)


@app.get("/health")
def health() -> dict:
    """Report whether the service is up and model weights are present."""
    if not RESULTS_FOLDER.is_dir():
        raise HTTPException(
            status_code=503,
            detail=f"model weights not found at {RESULTS_FOLDER}",
        )
    return {"status": "ok"}


@app.get("/version")
def version() -> dict:
    """Report the API version for compatibility checks by callers."""
    return {"api_version": API_VERSION}


@app.post("/jobs", status_code=202)
def submit_job(req: SubmitJobRequest) -> Job:
    """Queue a segmentation job for an input file already on the shared volume."""
    input_path = (INPUT_ROOT / req.input_path).resolve()
    if not input_path.is_relative_to(INPUT_ROOT):
        raise HTTPException(
            status_code=400,
            detail="input_path must resolve within the shared input directory",
        )
    if not input_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"input file not found: {req.input_path}"
        )

    job_id = uuid.uuid4().hex
    output_dir = OUTPUT_ROOT / job_id
    job = Job(
        job_id=job_id,
        status="queued",
        created_at=datetime.now(timezone.utc).isoformat(),
        output_path=str(output_dir.relative_to(DATA_ROOT)),
    )

    with _jobs_lock:
        _jobs[job_id] = job

    _executor.submit(_run_job, job_id, input_path, output_dir, req.small, req.cpu)
    return job


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> Job:
    """Report the status of a previously submitted job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/jobs/{job_id}/result")
def get_job_result(job_id: str) -> dict:
    """Return the parsed results.json and output file paths for a succeeded job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status != "succeeded":
        raise HTTPException(
            status_code=409, detail=f"job is {job.status}, not succeeded"
        )

    output_dir = DATA_ROOT / job.output_path
    results_file = output_dir / "results.json"
    if not results_file.is_file():
        raise HTTPException(
            status_code=500, detail="results.json missing for a succeeded job"
        )

    results = json.loads(results_file.read_text())
    output_files = [
        str(p.relative_to(DATA_ROOT)) for p in sorted(output_dir.glob("*.nii.gz"))
    ]
    return {"results": results, "output_files": output_files}


def _run_job(
    job_id: str, input_path: Path, output_dir: Path, small: bool, cpu: bool
) -> None:
    with _jobs_lock:
        _jobs[job_id].status = "running"

    cmd = [
        "bash",
        str(SOURCE_DIR / "fit.sh"),
        "-i",
        str(input_path),
        "-o",
        str(output_dir),
    ]
    if small:
        cmd.append("--small")
    if cpu:
        cmd.append("--cpu")

    result = subprocess.run(cmd, cwd=str(SOURCE_DIR), capture_output=True, text=True)

    with _jobs_lock:
        job = _jobs[job_id]
        if result.returncode == 0:
            job.status = "succeeded"
        else:
            job.status = "failed"
            job.error = (result.stderr or result.stdout)[-4000:]
