FROM texlive/texlive
WORKDIR /usr/src/app
RUN apt-get update && apt-get install -y python3-pip
RUN python3 -m pip install --break-system-packages rebiber
