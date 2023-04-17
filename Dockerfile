FROM tiangolo/uvicorn-gunicorn:python3.9-slim

# Install necessary packages
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

# Copy application code to Docker container
RUN mkdir /Shark-spotter
WORKDIR /Shark-spotter
COPY . /Shark-spotter/

# Install application dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Set up GPU configuration (if applicable)
# Uncomment the following lines if using GPU acceleration
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-command-line-tools-11-4 \
    && rm -rf /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Expose the port
EXPOSE 8000

# Set the command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
