.PHONY: report.pdf all clean

all: report.pdf

report.pdf: report.tex
	latexmk -f -pdf -bibtex report.tex

clean:
	latexmk -CA 

