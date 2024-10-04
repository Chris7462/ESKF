ESKFLie=eskf_lie
BIBTEX=/usr/bin/bibtex
LATEX=/usr/bin/pdflatex
#DVIPDFMX=/usr/bin/dvipdfmx

default:
	${LATEX} ${ESKFLie}.tex
	${LATEX} ${ESKFLie}.tex
	${BIBTEX} ${ESKFLie}.aux
	${LATEX} ${ESKFLie}.tex
	${LATEX} ${ESKFLie}.tex

clean:
	rm -f *.run.xml *.bcf *.ps *.log *.dvi *.aux *.lof *.toc *.lot *.blg *.bbl *.out *~ #*.pdf ;cd ${MPDIR};make clean
