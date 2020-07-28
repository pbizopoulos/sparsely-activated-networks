FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime
RUN apt-get update && apt-get upgrade -y && apt-get install -y git
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .
