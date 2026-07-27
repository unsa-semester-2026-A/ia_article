DOC = article
BUILD_DIR = latex_report/build
PDF_TARGET = $(BUILD_DIR)/$(DOC).pdf
LATEX = pdflatex
LATEX_FLAGS = -output-directory=$(BUILD_DIR) -interaction=nonstopmode -halt-on-error
BIBTEX = bibtex
SOURCES = $(DOC).tex latex_report/content/*.tex latex_report/style/*.tex latex_report/references.bib

FILTER = grep --color=always -iE "warning|error|underfull|overfull|^!" || true

all: $(PDF_TARGET)

$(PDF_TARGET): $(SOURCES)
	@mkdir -p $(BUILD_DIR)
	@echo "-> Primera pasada de LaTeX..."
	@$(LATEX) $(LATEX_FLAGS) $(DOC).tex | $(FILTER)
	
	@echo "-> Procesando bibliografía..."
	@$(BIBTEX) $(BUILD_DIR)/$(DOC) | grep --color=always -iE "warning|error" || true
	
	@echo "-> Segunda pasada de LaTeX..."
	@$(LATEX) $(LATEX_FLAGS) $(DOC).tex | $(FILTER)
	
	@echo "-> Tercera pasada de LaTeX..."
	@$(LATEX) $(LATEX_FLAGS) $(DOC).tex | $(FILTER)
	
	@echo "-> Compilación completada. Log completo en $(BUILD_DIR)/$(DOC).log"


fast:
	@mkdir -p $(BUILD_DIR)
	@echo "-> Compilación rápida..."
	@$(LATEX) $(LATEX_FLAGS) $(DOC).tex | $(FILTER)

clean:
	@echo "-> Limpiando archivos generados..."
	@rm -rf $(BUILD_DIR)
	@rm -f $(DOC).pdf

rebuild: clean all

.PHONY: all fast clean rebuild
