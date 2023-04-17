FROM tiangolo/uvicorn-gunicorn:python3.9-slim

RUN mkdir /Shark-spotter
COPY . /Shark-spotter
WORKDIR /Shark-spotter

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub && apt-key add 7fa2af80.pub

ENV CUDA_VERSION 11.4.2

RUN echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list \
    && echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_VERSION \
        cuda-compat-11-4 \
        cuda-nvcc-$CUDA_VERSION \
        cuda-command-line-tools-$CUDA_VERSION \
        libcudnn8=8.2.4.15-1+cuda11.4 \
        libcudnn8-dev=8.2.4.15-1+cuda11.4 \
        libnccl2=2.10.3-1+cuda11.4 \
        libnccl-dev=2.10.3-1+cuda11.4 \
    && apt-mark hold libcudnn8 \
    && apt-mark hold libcudnn8-dev \
    && apt-mark hold libnccl2 \
    && apt-mark hold libnccl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn main:app --host=0.0.0.0 --port=${PORT:-8000}
