SUBDIRS = docs notes paper refs src
PROJNAME = supersat

.PHONY: help all $(SUBDIRS)
.DEFAULT_GOAL: help

help:
	@echo To run all Makefiles for $(PROJNAME), enter command 'make all'
	@echo
	@echo You may wish to run sub-Makefiles individually first, found in \
	/docs, /notes, /paper, /refs, /src subdirectories.
	@echo
	@echo If 'make all' runs without errors, the final .pdf of the paper \
	will be found in the /paper subdirectory. The code \
	documentation can be accessed by opening \
	/docs/build/index.html in a browser window.

all:$(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

docs:src

paper:src refs
