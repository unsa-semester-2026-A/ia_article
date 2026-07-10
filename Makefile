DOC = article
BUILD_DIR = latex_report/build
LATEX = pdflatex
LATEX_FLAGS = -output-directory=$(BUILD_DIR) -interaction=nonstopmode -halt-on-error
BIBTEX = bibtex

FILTER = grep --color=always -iE "warning|error|underfull|overfull|^!" || true

all: $(DOC).pdf

$(DOC).pdf:
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

.PHONY: all fast clean rebuild $(DOC).pdf
