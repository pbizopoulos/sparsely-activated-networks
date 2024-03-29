.POSIX:

cmd = /bin/sh -c 'latexmk -outdir=tmp/ -pdf ms.tex && touch tmp/ms.bbl && cp tmp/ms.bbl . && tar cf tmp/tex.tar ms.bbl ms.bib ms.tex $$(grep "^INPUT ./" tmp/ms.fls | uniq | cut -b 9-) && rm ms.bbl'

all: tmp tmp/all-done

check: tmp tmp/check-done

clean:
	rm -rf tmp/

.dockerignore:
	printf '*\n' > $@

.gitignore:
	printf 'tmp/\n' > $@

Dockerfile:
	printf 'FROM texlive/texlive\nWORKDIR /usr/src/app\nRUN apt-get update && apt-get install -y python3-pip\nRUN python3 -m pip install --break-system-packages rebiber@https://github.com/yuchenlin/rebiber/tarball/main\n' > $@

ms.bib:
	touch $@

ms.tex:
	printf "\\\documentclass{article}\n\n\\\begin{document}\nTitle\n\\\end{document}\n" > $@

tmp:
	mkdir $@

tmp/all-done: .dockerignore .gitignore Dockerfile ms.bib ms.tex
	docker container run \
		$$(test -t 0 && printf '%s' '--interactive --tty') \
		--detach-keys 'ctrl-^,ctrl-^' \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		$$(docker image build --quiet .) $(cmd)
	touch $@

tmp/check-done: .dockerignore .gitignore Dockerfile tmp/all-done tmp/check-bib-done tmp/check-tex-done
	if ls -ap | grep -v -E -x './|../|.dockerignore|.gitignore|Dockerfile|Makefile|ms.bib|ms.tex|prm/|python/|tmp/' | grep -q .; then false; fi
	test $$(basename $$(pwd)) = "latex"
	touch $@

tmp/check-bib-done: ms.bib
	docker container run \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		$$(docker image build --quiet .) /bin/sh -c 'checkcites tmp/ms.aux && rebiber --input_bib ms.bib --remove url'
	touch $@

tmp/check-tex-done: ms.tex
	docker container run \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):/usr/src/app/ \
		$$(docker image build --quiet .) /bin/sh -c 'chktex ms.tex && lacheck ms.tex'
	touch $@
