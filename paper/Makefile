#loosely based on https://github.com/pinga-lab/paper-template/blob/master/manuscript/Makefile

FIGSDIR = ~/proj/20supersat/figures
PAPERDIR = ~/proj/20supersat/paper

PAPERAUX = $(PAPERDIR)/supersat_paper.aux
PAPERPDF = $(PAPERDIR)/supersat_paper.pdf
PAPERTEX = $(PAPERDIR)/supersat_paper.tex

#various latex dependencies
BIB = $(wildcard $(PAPERDIR)/*.bib)
BST = $(wildcard $(PAPERDIR)/*.bst)
CLS = $(wildcard $(PAPERDIR)/*.cls)
STY = $(wildcard $(PAPERDIR)/*.sty)
JPG = $(wildcard $(FIGSDIR)/*.jpg)
PNG = $(wildcard $(FIGSDIR)/*.png)

.PHONY: help clean
.DEFAULT_GOAL := $(PAPERPDF)

help:
	@echo To compile paper, enter command 'make'.
	@echo To clear auxiliary files, enter command 'make clean'. \
	[Caution, this will permanantly remove aux, bbl, blg, \
	log, and synctex files in the /paper subdirectory]

clean:
	@rm -f $(PAPERDIR)/*.aux
	@rm -f $(PAPERDIR)/*.bbl
	@rm -f $(PAPERDIR)/*.blg
	@rm -f $(PAPERDIR)/*.log
	@rm -f $(PAPERDIR)/*.gz

$(PAPERPDF):$(PAPERTEX) $(BIB) $(BST) $(CLS) $(FIGS) $(STY) 
	@texliveonfly $(PAPERTEX)
	@bibtex $(PAPERAUX)
	@texliveonfly $(PAPERTEX)
	@texliveonfly $(PAPERTEX)

