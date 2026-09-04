FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3.11 python3-pip ffmpeg git curl && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/ma-sub
COPY . .
RUN python3.11 -m pip install --no-cache-dir -r requirements.lock
ENTRYPOINT ["./mas"]
