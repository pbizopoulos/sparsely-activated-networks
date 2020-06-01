# Sparsely Activated Networks
This repository contains the code that generates the results of the paper **Sparsely Activated Networks**.

ArXiv link: <https://arxiv.org/abs/1907.06592>

# Instructions
The syntax of the `make` command is as follows:

`make [docker] [ARGS="[--full] [--gpu]"]`

where `[...]` denotes an optional argument.

For example you can choose one of the following:
- `make`
	- Requires local installation of requirements.txt and texlive-full.
	- Takes ~5 minutes and populates the figures and table.
- `make ARGS="--full --gpu"`
	- Requires local installation of requirements.txt and texlive-full.
	- Takes a week on an NVIDIA Titan X.
- `make docker`
	- Requires local installation of docker.
	- Takes ~5 minutes.
- `make docker ARGS="--full --gpu"`
	- Requires local installation of nvidia-container-toolkit.
	- Takes a week on an NVIDIA Titan X.
- `make clean`
	- Restores the repo in its initial state by removing all figures, tables and downloaded datasets.

# Citation
If you use this repository cite the following:
```
@article{bizopoulos2020sparsely,
	title={Sparsely activated networks},
	author={Bizopoulos, Paschalis and Koutsouris, Dimitrios},
	journal={IEEE Transactions on Neural Networks and Learning Systems},
	year={2020},
	publisher={IEEE}
}
```
