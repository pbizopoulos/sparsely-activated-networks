[![arXiv](http://img.shields.io/badge/cs.LG-arXiv%3A1907.06592-B31B1B.svg)](https://arxiv.org/abs/1907.06592)
[![citation](http://img.shields.io/badge/citation-0091FF.svg)](https://scholar.google.com/scholar?q=Sparsely%20Activated%20Networks.%20arXiv%202020)
[![template](http://img.shields.io/badge/template-EEE0B1.svg)](https://github.com/pbizopoulos/a-makefile-for-developing-containerized-latex-technical-documents-template)
[![test-draft-version-document-reproducibility](https://github.com/pbizopoulos/sparsely-activated-networks/workflows/test-draft-version-document-reproducibility/badge.svg)](https://github.com/pbizopoulos/sparsely-activated-networks/actions?query=workflow%3Atest-draft-version-document-reproducibility)

# Sparsely Activated Networks
This repository contains the code that generates **Sparsely Activated Networks** appeared in TNNLS.

## Requirements
- [Docker](https://docs.docker.com/get-docker/)
    - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit) (Optional)
- [Make](https://www.gnu.org/software/make/)

## Instructions
1. `git clone https://github.com/pbizopoulos/sparsely-activated-networks`
2. `cd sparsely-activated-networks/`
3. `sudo systemctl start docker`
4. make options
    - `make` # Generate the draft (fast) version document.
    - `make VERSION=--full` # Generate the full (slow) version document.
    - `make clean` # Remove the tmp/ directory.
