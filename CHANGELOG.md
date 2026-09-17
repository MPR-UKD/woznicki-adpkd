# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Covers this branch (`http-api`) relative to the parent repo's `main` branch.

## [Unreleased]

### Added

- `source/api_service.py`: a FastAPI HTTP API wrapping `fit.sh`, so another service (e.g. a nomiso plugin) can submit segmentation jobs over HTTP instead of exec'ing into the container:
  - `POST /jobs` — queue a job for an input file already on the shared volume (`input_path`, `small`, `cpu`).
  - `GET /jobs/{job_id}` — job status: `queued` / `running` / `succeeded` / `failed`.
  - `GET /jobs/{job_id}/result` — the parsed `results.json` plus output file paths, once succeeded.
  - `GET /health` — fast liveness probe (model weights present at `RESULTS_FOLDER`), used by the container `HEALTHCHECK`.
  - `GET /version` — API version string for compatibility checks.
  - `GET /status` — GPU availability (name/device) and, per task (`Task002_Kidney` = axial, `Task003_coronal` = coronal), whether a complete trained model is present: the `2d` and `3d_fullres` `model_final_checkpoint` (`.model` + `.model.pkl`) checkpoints the default ensemble path needs, plus the ensemble `postprocessing.json`.
  - Jobs run serialized — one GPU, one job at a time — via a single-worker background executor; additional submissions queue rather than running concurrently.
- `docker-compose.yml`: an `adpkd` service running the API, with `data` (job input/output files) and `models` (trained model weights) named volumes, an NVIDIA GPU reservation, and a healthcheck against `/health`. No host port is published — reachable only on the compose network (e.g. `http://adpkd:9000`).
- `.dockerignore`: excludes `source/trained_models/` and `test_data/` from the build context. Model weights are supplied via the `models` volume at runtime instead of being baked into the image; sample test volumes aren't needed by the service and were only ever used for a manual smoke check.
- `.gitattributes`: enforces LF line endings (`* text=auto eol=lf`, `*.sh text eol=lf`) so shell scripts don't pick up CRLF from a Windows checkout/edit and break `bash` inside the Linux container.
- `fastapi`/`uvicorn[standard]` dependencies, for the HTTP API.
- `docs/api.md`: the HTTP API reference (starting the service, its volumes, every endpoint, the `result`/`status` response shapes).
- README: the one-time `models` volume-seeding step that replaces the old "download into `source/` before building" flow, and a link to `docs/api.md` for the API reference.

### Changed

- Dockerfile:
  - No longer copies `uv.lock` (it was never read — `uv pip install -r pyproject.toml` is uv's pip-compatible interface, which doesn't consult a lockfile) or `test_data/` (now excluded via `.dockerignore` instead).
  - Adds `EXPOSE 9000` and a `HEALTHCHECK` for the new API.
  - Creates `/data` owned by uid 1000 at build time. Relevant when this image and nomiso's share one Docker volume (see the root-level `docker-compose.yml` in the parent monorepo): nomiso's own container runs as uid 1000 (`appuser`), while this one runs as root and has no `/data` path of its own — without this, whichever container happens to populate that shared volume first would leave it root-owned, blocking nomiso from writing to it.

[Unreleased]: https://github.com/MPR-UKD/woznicki-adpkd/compare/main...http-api
