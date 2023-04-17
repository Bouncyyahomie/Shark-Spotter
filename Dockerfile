FROM tiangolo/uvicorn-gunicorn:python3.9-slim

RUN mkdir /Shark-spotter
COPY . /Shark-spotter
WORKDIR /Shark-spotter

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add -
RUN echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" >> /etc/apt/sources.list.d/cuda.list
RUN echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" >> /etc/apt/sources.list.d/cuda.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-command-line-tools \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN apt-get install gcc ffmpeg libsm6 libxext6 openssh-client -y
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn main:app --host=0.0.0.0 --port=${PORT:-8000}
