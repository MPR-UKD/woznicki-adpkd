### Download docker image

```
$ docker pull piotrekwoznicki/adpkd-net:v0.1
```

More instructions on the inference can be found at [Docker Hub](https://hub.docker.com/repository/docker/piotrekwoznicki/adpkd-net).

### Run

#### 1. Launch the container:

Requires a GPU with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed (Docker Engine 19.03+). If you don't have a GPU, skip this and run inference with `--cpu` (see below).

```
$ bash launch_container.sh [path_to_dataset - will be mounted at /workspace/data]
```

#### 2. Run inference:

```
$ cd /workspace/source
$ bash fit.sh  \
    -i / --InputVol [abs_path_to_nifti_image]
    -o / --OutputDir [where_to_save_segmentation_and_JSON]
    --small [whether to use single model - OPTIONAL (requires shorter calculation time)]
    --cpu [run without a GPU - OPTIONAL (much slower: roughly 10-30x GPU runtime)]
```

#### 3. Run inference for test:

```
$ cd /workspace/source
$ bash fit.sh  \
    -i /workspace/test_data/T2_ax.nii.gz
    -o /workspace/data/test_results_ax

$ bash fit.sh  \
    -i /workspace/test_data/T2_cor.nii.gz
    -o /workspace/data/test_results_cor
```

### Before building the docker image, download the folder _trained_models_ from [GDrive](https://drive.google.com/drive/folders/1D2glVKAKcAdQmmqct964RZoxHCpyDqgc?usp=sharing) and place it in the _source_ directory

