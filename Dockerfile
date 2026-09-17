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
COPY ./pyproject.toml /workspace/
COPY ./source /workspace/source

WORKDIR /workspace

RUN uv venv --python 3.12 /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Required python packages for inference. The vendored nnUNet package is
# installed separately with --no-deps: its own setup.py pulls in the
# deprecated "sklearn" PyPI shim, which fails to build under uv/pip's
# modern resolver (see pyproject.toml for details).
RUN uv pip install --python /opt/venv/bin/python -r pyproject.toml \
    && uv pip install --python /opt/venv/bin/python --no-deps -e source/nnUNet/

# HTTP API (source/api_service.py) — trained_models and the shared data
# directory are volumes, not baked into the image; see .dockerignore.
#
# When composed with nomiso (docker-compose.yml at the repo root), this
# container and nomiso's share one Docker volume at /data. Whichever
# container starts first is the one whose image populates that volume on
# first mount — this container still runs as root regardless, but nomiso
# runs as uid 1000 (appuser) and needs write access under /data even when
# this image is the one that ends up populating it.
RUN mkdir -p /data && chown -R 1000:1000 /data

EXPOSE 9000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -f http://127.0.0.1:9000/health || exit 1
