FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime
RUN apt-get update && apt-get install -y git
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip && python -m pip install --no-cache-dir -r requirements.txt
COPY . .
