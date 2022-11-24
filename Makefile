.POSIX:

.PHONY: all check clean

aux_file_name = ms.aux
bib_file_name = ms.bib
bib_target = $$(test -s $(bib_file_name) && printf 'bin/check-bib')
tex_file_name = ms.tex
tex_target = $$(test -s $(tex_file_name) && printf 'bin/check-tex')
work_dir = /work

all: bin/all

check: bin/check

clean:
	rm -rf bin/

$(bib_file_name):
	touch $(bib_file_name)

$(tex_file_name):
	printf "\\\documentclass{article}\n\n\\\begin{document}\nTitle\n\\\end{document}\n" > $(tex_file_name)

.dockerignore:
	printf '*\n' > .dockerignore

.gitignore:
	printf 'bin/\n' > .gitignore

bin:
	mkdir bin

bin/all: $(bib_file_name) $(tex_file_name) .dockerignore .gitignore bin
	docker container run \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		texlive/texlive latexmk -gg -pdf -outdir=bin/ $(tex_file_name)
	touch bin/ms.bbl && cp bin/ms.bbl .
	docker container run \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		texlive/texlive /bin/sh -c 'tar cf bin/tex.tar ms.bbl $(bib_file_name) $(tex_file_name) $$(grep "^INPUT ./" bin/ms.fls | uniq | cut -b 9-)'
	rm ms.bbl
	touch bin/all

bin/check: .dockerignore .gitignore bin
	$(MAKE) $(bib_target) $(tex_target)

bin/check-bib: $(bib_file_name) .dockerignore .gitignore bin/all
	docker container run \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		texlive/texlive /bin/sh -c '\
		checkcites bin/$(aux_file_name)'
	docker container run \
		--env HOME=$(work_dir)/bin \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		python /bin/sh -c '\
		python3 -m pip install --no-cache-dir --upgrade pip && \
		python3 -m pip install --no-cache-dir betterbib rebiber && \
		cd bin/ && \
		.local/bin/rebiber -u && \
		.local/bin/rebiber -i ../$(bib_file_name) && \
		.local/bin/betterbib update --in-place --sort-by-bibkey --tab-indent ../$(bib_file_name) && \
		sed --in-place 1,3d ../$(bib_file_name)'
	touch bin/check-bib

bin/check-tex: $(tex_file_name) .dockerignore .gitignore bin
	docker container run \
		--rm \
		--user $$(id -u):$$(id -g) \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		texlive/texlive /bin/sh -c '\
		chktex $(tex_file_name) && \
		lacheck $(tex_file_name)'
	touch bin/check-tex
