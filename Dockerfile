FROM pytorch/pytorch
ENV TZ=Europe/Athens
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install -y --no-install-recommends texlive-full
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
