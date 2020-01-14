PACKAGES:=$(filter-out slash,$(subst /, slash,$(wildcard */)))
NOTESFILE=~/proj/20supersat/notes/supersat_notes.tex
PAPERFILE=~/proj/20supersat/paper/supersat_paper.tex

.PHONY: all help $(PACKAGES)
.DEFAULT_GOAL := all

help:
	@echo "To make all figures from all packages, enter command \
	'make' or 'make all'.";
	@for pkg in $(PACKAGES); do echo "To make figures in package \
	"$$pkg", enter command 'make "$$pkg"'."; done

all: $(PACKAGES)

#some mumbo jumbo, because you can't expand static wildcard % in make \
	wildcard function the regular way
.SECONDEXPANSION:

$(PACKAGES): %: $$(addprefix ../figures/,$$(subst _figsrc.py,_figure.png,$$(wildcard $$*/*_figsrc.py)))

#back to regular expansion
../figures/%_figure.png: %_figsrc.py
	@python "$<"
	@touch $(NOTESFILE)
	@touch $(PAPERFILE)
