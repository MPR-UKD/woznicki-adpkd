# HTTP API

> **TL;DR** — `source/api_service.py` wraps `fit.sh` in a FastAPI app so another service (e.g. a nomiso plugin) can submit segmentation jobs over HTTP instead of exec'ing into the container. Covers: starting the service and its volumes, every endpoint, and the exact `result`/`status` response shapes.

---

## Starting the service

GPU inference is exclusive to one job at a time — the service queues additional submissions rather than running them concurrently.

```bash
docker compose up --build
```

This starts the `adpkd` service on the compose network at `http://adpkd:9000` (no host port is published — see `docker-compose.yml`), with two named volumes:

| Volume | Mounted at | Purpose |
|---|---|---|
| `data` | `/data/adpkd` | Shared input/output files for jobs (`inputs/`, `outputs/<job_id>/`) |
| `models` | `/workspace/source/trained_models` | Trained model weights, seeded once — see the main `README.md` |

## Endpoints

| Method & path | Description |
|---|---|
| `GET /health` | `200` if the service is up and model weights are present at `RESULTS_FOLDER`; `503` otherwise. Fast liveness probe — used by the container `HEALTHCHECK` |
| `GET /version` | `{"api_version": "1.0.0"}`, for callers to check compatibility |
| `GET /status` | Heavier, detailed readiness — GPU availability and per-task trained-model completeness (see below). For a UI to display, not for orchestration to poll |
| `POST /jobs` | Queue a job. Body: `{"input_path": "<path relative to data/inputs>", "small": false, "cpu": false}`. Returns `202` with the job |
| `GET /jobs/{job_id}` | Job status: `queued` \| `running` \| `succeeded` \| `failed`, with `error` set on failure |
| `GET /jobs/{job_id}/result` | Once `succeeded`: the parsed `results.json` plus output file paths (relative to `data/`) |

`input_path` in `POST /jobs` must point to a file already under the `data` volume's `inputs/` directory — the API exchanges file paths, not file contents, since NIfTI volumes are too large to upload in a request body.

## Result shape

`GET /jobs/{job_id}/result` returns the same structure `postprocess_masks.py` writes to `results.json`:

```json
{
  "results": {
    "both_kidneys": {"volume": "1234.56 ml"},
    "left_kidney": {"volume": "617.28 ml"},
    "right_kidney": {"volume": "617.28 ml"},
    "liver": "not measured"
  },
  "output_files": ["outputs/<job_id>/seg.nii.gz", "outputs/<job_id>/both_kidneys.nii.gz", "..."]
}
```

An organ that wasn't measured (e.g. no liver label in the segmentation) reports the string `"not measured"` instead of a `volume` object.

## Status shape

`GET /status` reports GPU availability and, per task (`Task002_Kidney` = axial, `Task003_coronal` = coronal), whether a complete trained model is present — the `2d` and `3d_fullres` checkpoints (`model_final_checkpoint.model` + `.model.pkl`) `fit.sh`'s default ensemble path needs, plus the ensemble `postprocessing.json`:

```json
{
  "gpu": {"available": true, "name": "NVIDIA A100", "device": "cuda:0"},
  "tasks": {
    "Task002_Kidney": {"ready": true, "missing": []},
    "Task003_coronal": {"ready": false, "missing": ["3d_fullres checkpoint"]}
  },
  "models_ready": false
}
```

`models_ready` is `true` only when every task in `tasks` reports `ready: true`.

## Plugin integration

> [!NOTE]
> The `libs/nomiso-plugin-adpkd` package is the nomiso-side `PluginBase` implementation that calls these endpoints — see `libs/nomiso/docs/plugins.md` for the general plugin contract it follows.
