ARG UV_VERSION=0.8.14
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MAS_VENV_DIR=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl ffmpeg rclone \
    && rm -rf /var/lib/apt/lists/* \
    && uv python install 3.11 \
    && uv venv --python 3.11 /opt/venv
WORKDIR /opt/ma-sub
COPY requirements.lock ./
RUN uv pip install --no-cache --python /opt/venv/bin/python -r requirements.lock
COPY . ./
RUN chmod +x mas runpod/*.sh
CMD ["sleep", "infinity"]
