  #!/bin/bash

if ! [ $1 ]; then
    echo "> Missing path to data directory!"
    exit
fi
DATA_PATH=$1

# Requires the NVIDIA Container Toolkit (successor to nvidia-docker2) and Docker Engine 19.03+
docker run \
  --gpus all \
  -v "$DATA_PATH":/workspace/data \
  -it piotrekwoznicki/adpkd-net:v0.1
