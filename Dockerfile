FROM tiangolo/uvicorn-gunicorn:python3.9-slim
RUN mkdir /Shark-spotter
COPY . /Shark-spotter
WORKDIR /Shark-spotter
RUN apt-get update
RUN apt-get install gcc ffmpeg libsm6 libxext6 openssh-client -y
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn main:app  --host=0.0.0.0 --port=${PORT:-8000}