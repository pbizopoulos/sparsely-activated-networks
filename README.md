[![arXiv](http://img.shields.io/badge/cs.LG-arXiv%3A1907.06592-B31B1B.svg)](https://arxiv.org/abs/1907.06592)
[![citation](http://img.shields.io/badge/citation-0091FF.svg)](https://scholar.google.com/scholar?q=Sparsely%20Activated%20Networks.%20arXiv%202020)
[![template](http://img.shields.io/badge/template-EEE0B1.svg)](https://github.com/pbizopoulos/latex-technical-documents-with-docker-and-make-template)
[![test-local-reproducibility](https://github.com/pbizopoulos/sparsely-activated-networks/workflows/test-local-reproducibility/badge.svg)](https://github.com/pbizopoulos/sparsely-activated-networks/actions?query=workflow%3Atest-local-reproducibility)

# Sparsely Activated Networks
This repository contains the code that generates **Sparsely Activated Networks** appeared in TNNLS.

## Requirements
- [POSIX-oriented operating system](https://en.wikipedia.org/wiki/POSIX#POSIX-oriented_operating_systems)
- [Docker](https://docs.docker.com/get-docker/)
- [Make](https://www.gnu.org/software/make/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit) (required only when using CUDA)

## Instructions
1. `git clone https://github.com/pbizopoulos/sparsely-activated-networks`
2. `cd sparsely-activated-networks`
3. `sudo systemctl start docker`
4. make options
    * `make`             # Generate the fast/draft version document.
    * `make ARG=--full`  # Generate the slow/final version document.
    * `make clean`       # Remove the tmp directory.
