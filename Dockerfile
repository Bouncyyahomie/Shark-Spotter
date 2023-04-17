FROM tiangolo/uvicorn-gunicorn:python3.9-slim
RUN mkdir /Shark-spotter
COPY . /Shark-spotter
WORKDIR /Shark-spotter
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        ffmpeg \
        libsm6 \
        libxext6 \
        openssh-client \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
# Set up GPU configuration (if applicable)
# Uncomment the following lines if using GPU acceleration
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-command-line-tools-11-1 \
    && rm -rf /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

CMD uvicorn main:app  --host=0.0.0.0 --port=${PORT:-8000}