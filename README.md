![test-reproducible-build](https://github.com/pbizopoulos/sparsely-activated-networks/workflows/test-reproducible-build/badge.svg)
[![arXiv](http://img.shields.io/badge/cs.LG-arXiv%3A1907.06592-B31B1B.svg)](https://arxiv.org/abs/1907.06592)
[![citation](http://img.shields.io/badge/citation-0091FF.svg)](https://scholar.google.com/scholar?q=Sparsely%20Activated%20Networks.%20arXiv%202020)
[![template](http://img.shields.io/badge/template-EEE0B1.svg)](https://github.com/pbizopoulos/reproducible-builds-for-computational-research-papers-template)

# Sparsely Activated Networks
This repository contains the code that generates the results of the paper **Sparsely Activated Networks** appeared in TNNLS.

## Requirements
- UNIX utilities (cmp, cp, echo, rm, touch)
- [docker](https://docs.docker.com/get-docker/)
- make
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit) (required only when using CUDA)

## Instructions
1. `git clone https://github.com/pbizopoulos/sparsely-activated-networks`
2. `cd sparsely-activated-networks`
3. `sudo systemctl start docker`
4. make options
    * `make`             # Generate pdf.
    * `make ARGS=--full` # Generate full pdf.
    * `make clean`       # Remove cache, results directories and tex auxiliary files.
