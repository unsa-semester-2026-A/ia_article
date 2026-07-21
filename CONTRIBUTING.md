# Guía de Contribución y Flujo de Trabajo Metodológico (CONTRIBUTING.md)

Este documento define el flujo de trabajo oficial y los estándares de desarrollo que todo el equipo (y sus respectivos agentes de IA) debe seguir estrictamente para la codificación, formateo y despliegue del proyecto.

---

## 🚀 El Flujo de Trabajo Recomendado (5 Pasos)

Para evitar la acumulación de código sucio en producción, prevenir errores en cascada y optimizar el uso de GPUs en la nube, seguimos este flujo secuencial:

### 1. Prototipado Rápido (Notebooks de Exploración)
* **Acción**: Realiza pruebas iniciales y exploración rápida de datos en un notebook de borrador (`.ipynb`) dentro del directorio `experiments/notebooks/`.
* **Restricción**: Usa solo una pequeña muestra del dataset (pocos clips o imágenes) para acelerar el desarrollo interactivo.

### 2. Modularización en el Paquete de Producción
* **Acción**: Una vez validada la lógica en el prototipo, traslada y encapsula el código en clases o funciones modulares limpias dentro de la carpeta `experiments/src/` (bajo su submódulo correspondiente, ej. `src/evaluation/` o `src/pseudo_labeling/`).
* **Restricción**: Todo el código de producción debe ser parametrizable y configurable (evitando variables hardcodeadas o argumentos CLI innecesarios).

### 3. Pruebas Unitarias Colocadas (Pytest)
* **Acción**: Escribe pruebas unitarias que validen la robustez del código. Las pruebas deben estar localizadas directamente al costado del archivo que testean (ej. `test_parser.py` al lado de `parser.py`).
* **Comprobación**: Corre las pruebas localmente usando:
  ```bash
  uv run pytest src/
  ```
  Todas las pruebas deben pasar al 100% de manera exitosa antes de realizar cualquier commit.

### 4. Estilo de Código y Formateo Obligatorio (Ruff)
Ruff es nuestro formateador y linter oficial. **Ningún cambio de código debe confirmarse en Git sin pasar la validación completa de Ruff.**
* **Acción**: Formatea todos los archivos del paquete:
  ```bash
  uv run ruff format src/
  ```
* **Acción**: Corre el linter para comprobar docstrings (Google Style) y estilo:
  ```bash
  uv run ruff check src/ --fix
  ```
* **Resultado Esperado**: El linter debe devolver un estado limpio (`All checks passed!`).

### 5. Despliegue en Notebook Orquestador Ligero (Colab / Kaggle)
* **Acción**: Para procesar los datos de forma masiva en producción (Kaggle/Colab), se utiliza un notebook principal orquestador y minimalista (ej. `final-notebook-optimized.ipynb`).
* **Flujo del Notebook**:
  1. Realiza una clonación superficial (`--depth 1 --sparse`) del repositorio para traer únicamente la carpeta de código `experiments/` (evitando descargar archivos pesados de datasets).
  2. Instala el paquete local en modo editable con dependencias de la nube: `%pip install -e .[cloud]`.
  3. Ejecuta los tests de forma preventiva en la VM de la nube con `!pytest src/`.
  4. Llama a ejecutar el módulo mediante el comando mágico `%run` (ej. `%run src/pseudo_labeling/pseudo_labeler.py`).

---

## 🛠️ Herramientas de Desarrollo Locales

Si trabajas de forma local con la herramienta `uv`, estos son los comandos de validación rápida:

* **Correr tests y verificar cobertura**:
  ```bash
  uv run pytest src/
  ```
* **Verificar tipado estático (Pyright)**:
  ```bash
  uv run pyright src/
  ```
* **Formatear e inspeccionar linter (Ruff)**:
  ```bash
  uv run ruff format src/ && uv run ruff check src/
  ```
