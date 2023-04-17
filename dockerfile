# Start with a base Ubuntu image
FROM ubuntu:18.04

RUN mkdir /Shark-spotter
COPY . /Shark-spotter
WORKDIR /Shark-spotter

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        gnupg \
        curl \
        build-essential \
        python3-dev \
        python3-pip

# Install TensorFlow and other Python packages
RUN pip3 install -r requirements.txt

# Install TensorRT
RUN echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" >> /etc/apt/sources.list.d/nvidia-ml.list && \
    curl -sL https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    apt-get update && \
    apt-get install -y --no-install-recommends tensorrt

# Set the LD_LIBRARY_PATH environment variable
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/TensorRT-6.0.1.5/lib:$LD_LIBRARY_PATH

CMD uvicorn main:app --host=0.0.0.0 --port=${PORT:-8000}