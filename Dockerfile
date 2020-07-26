FROM pytorch/pytorch
RUN apt-get update && apt-get upgrade -y && apt-get install -y git
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .
