#loosely based on https://github.com/pinga-lab/paper-template/blob/master/manuscript/Makefile

FIGSDIR = ~/proj/20supersat/figures
NOTESDIR = ~/proj/20supersat/notes

NOTESAUX = $(NOTESDIR)/supersat_notes.aux
NOTESPDF = $(NOTESDIR)/supersat_notes.pdf
NOTESTEX = $(NOTESDIR)/supersat_notes.tex

#various latex dependencies
BIB = $(wildcard $(NOTESDIR)/*.bib)
BST = $(wildcard $(NOTESDIR)/*.bst)
CLS = $(wildcard $(NOTESDIR)/*.cls)
STY = $(wildcard $(NOTESDIR)/*.sty)
JPG = $(wildcard $(FIGSDIR)/*.jpg)
PNG = $(wildcard $(FIGSDIR)/*.png)

.PHONY: help clean
.DEFAULT_GOAL := $(NOTESPDF)

help:
	@echo To compile notes, enter command 'make'.
	@echo To clear auxiliary files, enter command 'make clean'. \
	[Caution, this will permanantly remove aux, bbl, blg, \
	log, and synctex files in the /notes subdirectory]

clean:
	@rm -f $(NOTESDIR)/*.aux
	@rm -f $(NOTESDIR)/*.bbl
	@rm -f $(NOTESDIR)/*.blg
	@rm -f $(NOTESDIR)/*.log
	@rm -f $(NOTESDIR)/*.gz

$(NOTESPDF):$(NOTESTEX) $(BIB) $(BST) $(CLS) $(FIGS) $(STY) 
	@texliveonfly $(NOTESTEX)
	#@bibtex $(NOTESAUX)
	#@texliveonfly $(NOTESTEX)
	#@texliveonfly $(NOTESTEX)

