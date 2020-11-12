.POSIX:

ARGS= 
DEBUG_ARGS=--interactive --tty
MAKEFILE_DIR=$(dir $(realpath Makefile))
ifeq (, $(shell which nvidia-smi))
	DOCKER_GPU_ARGS=
else
	DOCKER_GPU_ARGS=--gpus all
endif

cache/ms.pdf: ms.tex ms.bib results/completed
	docker container run \
		--rm \
		--volume $(MAKEFILE_DIR):/usr/src/app \
		ghcr.io/pbizopoulos/texlive-full \
		-outdir=cache/ -usepretex="\pdfinfoomitdate=1\pdfsuppressptexinfo=-1\pdftrailerid{}" -gg -pdf -cd /usr/src/app/ms.tex
	@if [ -f cache/.tmp.pdf ]; then \
		cmp cache/ms.pdf cache/.tmp.pdf && echo 'ms.pdf unchanged.' || echo 'ms.pdf changed.'; fi
	@cp cache/ms.pdf cache/.tmp.pdf

results/completed: Dockerfile requirements.txt $(shell find . -maxdepth 1 -name '*.py')
	rm -rf results/*
	docker image build --tag sparsely-activated-networks .
	docker container run \
		$(DEBUG_ARGS) \
		--rm \
		--volume $(MAKEFILE_DIR):/usr/src/app \
		$(DOCKER_GPU_ARGS) \
		sparsely-activated-networks \
		$(ARGS)
	touch results/completed

clean:
	rm -rf __pycache__/ cache/* results/*
