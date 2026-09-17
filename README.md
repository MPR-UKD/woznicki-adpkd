### Download docker image

```bash
docker pull piotrekwoznicki/adpkd-net:v0.1
```

More instructions on the inference can be found at [Docker Hub](https://hub.docker.com/repository/docker/piotrekwoznicki/adpkd-net).

### Before building the docker image, download the trained models

Download the _trained_models_ folder from [GDrive](https://drive.google.com/drive/folders/1D2glVKAKcAdQmmqct964RZoxHCpyDqgc?usp=sharing).

Model weights are **not** baked into the image (`.dockerignore` excludes `source/trained_models/`). Seed the `models` volume once instead:

```bash
docker volume create adpkd-net_models
docker run --rm \
    -v adpkd-net_models:/target \
    -v /path/to/downloaded/trained_models:/source:ro \
    alpine cp -a /source/. /target/
```

Run this again whenever the trained models are updated.

### Run interactively (CLI)

Requires a GPU with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed (Docker Engine 19.03+). If you don't have a GPU, skip this and run inference with `--cpu` (see below).

#### 1. Launch the container

```bash
bash launch_container.sh [path_to_dataset - will be mounted at /workspace/data]
```

#### 2. Run inference

```bash
cd /workspace/source
bash fit.sh \
    -i / --InputVol [abs_path_to_nifti_image] \
    -o / --OutputDir [where_to_save_segmentation_and_JSON] \
    --small [whether to use single model - OPTIONAL (requires shorter calculation time)] \
    --cpu [run without a GPU - OPTIONAL (much slower: roughly 10-30x GPU runtime)]
```

#### 3. Run inference for test

Sample test volumes aren't baked into the image (`.dockerignore` excludes `test_data/`, keeping the image smaller). Put your own axial/coronal test NIfTI volumes in the host directory you pass to `launch_container.sh` (mounted at `/workspace/data`) and reference them from there:

```bash
cd /workspace/source
bash fit.sh -i /workspace/data/T2_ax.nii.gz -o /workspace/data/test_results_ax
bash fit.sh -i /workspace/data/T2_cor.nii.gz -o /workspace/data/test_results_cor
```

### Run as a service (HTTP API)

`source/api_service.py` wraps `fit.sh` in a FastAPI app so another service (e.g. a nomiso plugin) can submit segmentation jobs over HTTP instead of exec'ing into the container. GPU inference is exclusive to one job at a time — the service queues additional submissions rather than running them concurrently.

```bash
docker compose up --build
```

This starts the `adpkd` service on the compose network at `http://adpkd:9000` (no host port is published — see `docker-compose.yml`), with two named volumes:

| Volume | Mounted at | Purpose |
|---|---|---|
| `data` | `/data/adpkd` | Shared input/output files for jobs (`inputs/`, `outputs/<job_id>/`) |
| `models` | `/workspace/source/trained_models` | Trained model weights, seeded once per the step above |

#### Endpoints

| Method & path | Description |
|---|---|
| `GET /health` | `200` if the service is up and model weights are present at `RESULTS_FOLDER`; `503` otherwise |
| `GET /version` | `{"api_version": "1.0.0"}`, for callers to check compatibility |
| `POST /jobs` | Queue a job. Body: `{"input_path": "<path relative to data/inputs>", "small": false, "cpu": false}`. Returns `202` with the job |
| `GET /jobs/{job_id}` | Job status: `queued` \| `running` \| `succeeded` \| `failed`, with `error` set on failure |
| `GET /jobs/{job_id}/result` | Once `succeeded`: the parsed `results.json` plus output file paths (relative to `data/`) |

`input_path` in `POST /jobs` must point to a file already under the `data` volume's `inputs/` directory — the API exchanges file paths, not file contents, since NIfTI volumes are too large to upload in a request body.

#### Result shape

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

> [!NOTE]
> Integrating this service into a nomiso plugin (the `PluginBase` page that calls it) is not part of this package. See `libs/nomiso/docs/plugins.md` for that contract; a plugin would call these endpoints over the compose network and use nomiso's `DATA_DIR` conventions for the shared volume.
