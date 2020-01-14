PAPERDIR = ~/proj/20supersat/paper
REFSDIR = ~/proj/20supersat/refs

BIBTEXFILE = $(REFSDIR)/refs.bib
REFSFILE = $(REFSDIR)/refs.txt

PROJLIBNAME = supersatlib

.PHONY: help clearall
.DEFAULT_GOAL := $(BIBTEXFILE)

help:
	@echo To update bibtex file, enter command 'make'.
	@echo The file refs.bib will appear in the /refs subdirectory,\
	as well as in the /paper subdirectory for LaTeX compilation.
	@echo 
	@echo To clear all references from the papis database, enter \
	command 'make clearall'.
	@echo This will not alter the 'refs.txt' file but will touch it.
	@echo WARNING: This is a permanent delete of the reference files!

$(BIBTEXFILE): $(REFSFILE)
	@$(REFSDIR)/update-proj-lib
	@rm -f $(BIBTEXFILE)
	@rm -f $(PAPERDIR)/refs.bib
	@papis -l $(PROJLIBNAME) export --all \
		--format bibtex \
		--out $(BIBTEXFILE)
	@cp $(BIBTEXFILE) $(PAPERDIR)/refs.bib

clearall:
	@rm -rf $(REFSDIR)/.cache/database/*
	@rm -rf $(REFSDIR)/docfiles/*
	@touch $(REFSDIR)/refs.txt
