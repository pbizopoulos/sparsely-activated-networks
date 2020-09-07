FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
RUN apt-get update && apt-get upgrade -y && apt-get install -y git
COPY requirements.txt .
RUN python -m pip install --upgrade pip && python -m pip install --no-cache-dir -r requirements.txt
COPY . .
