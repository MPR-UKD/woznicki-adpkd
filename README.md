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

`source/api_service.py` wraps `fit.sh` in a FastAPI app so another service can submit segmentation jobs over HTTP instead of exec'ing into the container.

```bash
docker compose up --build
```

See [`docs/api.md`](docs/api.md) for the service's volumes, every endpoint, and the exact `result`/`status` response shapes.
