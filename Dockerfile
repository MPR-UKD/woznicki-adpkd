FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv also manages Python interpreters itself, downloading standalone builds
# rather than relying on apt PPAs (e.g. deadsnakes), which keeps the image
# build independent of Launchpad's network availability.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy the source, in case this is run as a remote job
COPY ./pyproject.toml ./uv.lock /workspace/
COPY ./source /workspace/source
COPY ./test_data /workspace/test_data

WORKDIR /workspace

RUN uv venv --python 3.12 /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Required python packages for inference. The vendored nnUNet package is
# installed separately with --no-deps: its own setup.py pulls in the
# deprecated "sklearn" PyPI shim, which fails to build under uv/pip's
# modern resolver (see pyproject.toml for details).
RUN uv pip install --python /opt/venv/bin/python -r pyproject.toml \
    && uv pip install --python /opt/venv/bin/python --no-deps -e source/nnUNet/
