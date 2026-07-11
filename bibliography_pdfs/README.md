# Zettelkasten Bibliography & Notes System

Este directorio está organizado siguiendo el modelo **Zettelkasten** para gestionar las referencias bibliográficas, lecturas y la generación de conocimiento del proyecto.

El objetivo es facilitar el análisis de artículos científicos, permitiendo a los contribuidores resumir, enlazar y destilar ideas complejas en notas estructuradas y conectadas.

---

## 🗂️ Estructura General del Zettelkasten

El sistema se divide en tres tipos de notas fundamentales:

```
bibliography_pdfs/
├── README.md                 # Estas instrucciones
├── MANIFEST.md               # Inventario y control de artículos
├── literature_notes/         # Notas de literatura (específicas de cada PDF)
│   └── [BibTeX_Key]/         # Carpeta por cada artículo (ej. Ding2022DOTA)
│       ├── paper.pdf         # El archivo PDF original
│       ├── paper.txt         # Texto plano extraído con pdftotext
│       ├── ref.bib           # Entrada BibTeX del artículo
│       └── notes.md          # Resumen estructurado por secciones
├── permanent_notes/          # Notas permanentes (conceptos e ideas propias)
└── moc/                      # Mapas de Contenido (Map of Content)
```

---

## 📝 Los Tres Tipos de Notas

### 1. Notas de Literatura (`literature_notes/`)
Son notas tomadas directamente de los textos leídos. Cada artículo científico tiene su propia subcarpeta nombrada exactamente igual a su **BibTeX Key** (por ejemplo, `Ding2022DOTA` o `Zand2022OBB`).

Cada carpeta de nota de literatura debe contener exactamente los siguientes cuatro elementos:

1. **`paper.pdf`**: El archivo PDF original del artículo.
2. **`paper.txt`**: El texto completo del PDF en formato `.txt`, generado automáticamente usando la herramienta `pdftotext`:
   ```bash
   pdftotext paper.pdf paper.txt
   ```
3. **`ref.bib`**: El archivo con la cita en formato BibTeX del artículo.
4. **`notes.md`**: El documento principal de la nota de literatura.
   - **Requisito obligatorio**: Debe replicar **todas** las secciones y subsecciones del PDF (siguiendo su misma jerarquía de encabezados `#`, `##`, `###`).
   - **Contenido**: En lugar de copiar el texto completo del artículo, cada sección y subsección debe contener un **resumen entendible y destilado** con las ideas principales, metodologías, fórmulas clave, y resultados.

---

### 2. Notas Permanentes (`permanent_notes/`)
Son notas autónomas (atómicas) que contienen una única idea o concepto desarrollado con tus propias palabras. 
- No dependen del contexto del artículo original.
- Se crean a partir de las notas de literatura.
- Deben estar fuertemente enlazadas entre sí.
- Cada archivo se nombra de forma descriptiva (ej. `obb_vs_hbb_in_remote_sensing.md`).
- Deben incluir enlaces a la nota de literatura de origen (ej. `[[Ding2022DOTA]]`) y a otras notas permanentes relacionadas.

---

### 3. Mapas de Contenido (MOC - `moc/`)
Son notas que sirven de índice o "hub" temático. Agrupan y organizan notas permanentes y de literatura en torno a un tema o área de investigación (ej. `metodologia_deteccion_orientada.md` o `metricas_evaluacion.md`).
- Permiten navegar por el conocimiento sin imponer una jerarquía rígida.
- Facilitan la estructuración del borrador del artículo final.

---

## 🚀 Flujo de Trabajo para Contribuidores

Cuando agregues un nuevo artículo al repositorio, sigue estos pasos:

1. **Identificar la Clave BibTeX**: Define una clave única para el artículo (usualmente `[PrimerAutor][Año][PalabraClave]`, ej. `Ahmed2025`).
2. **Crear la Carpeta**: Crea una subcarpeta bajo `literature_notes/` con el nombre de la clave BibTeX (`literature_notes/Ahmed2025`).
3. **Guardar el PDF**: Coloca el archivo PDF en la carpeta y renombralo a `paper.pdf`.
4. **Extraer el Texto**: Ejecuta `pdftotext paper.pdf paper.txt` desde el directorio de la nota.
5. **Generar la Referencia**: Crea `ref.bib` y pega la entrada BibTeX del artículo.
6. **Crear la Plantilla de Notas**: Crea `notes.md` y estructura el esqueleto con todas las secciones y subsecciones del paper.
7. **Resumir**: Lee el artículo y completa cada sección de `notes.md` con un resumen claro y de alto valor.
8. **Crear Notas Permanentes y MOCs**: Conforme encuentres ideas clave o conceptos replicables, redacta notas permanentes e intégralas/enlázalas en los MOCs correspondientes.
