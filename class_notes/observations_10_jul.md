
### Correcciones Generales (Para todos)

* **Versionado de software:** Especificar siempre la versión de las herramientas utilizadas (por ejemplo, `Python v1.13.xxx`). *HECHO*
* **Trabajos Relacionados:** Es obligatorio incluir una tabla o un párrafo de resumen para consolidar la revisión bibliográfica. *HECHO*
* **Formato de texto:** Prohibido usar negritas para resaltar conceptos en el texto. *HECHO*
* **Notas al pie:** Evitar su uso en la medida de lo posible. *HECHO*
* **Tablas y Gráficos:** Cualquier tabla insertada en el documento debe ser mencionada y referenciada explícitamente en el cuerpo del texto.

---

### Correcciones Específicas (Nosotros)

**Estructura y Formato General**

* **Tipografía:** Corregir y unificar el tamaño de la fuente en todo el documento.
* **Párrafos:** Evitar por completo los párrafos de una sola línea u oración; deben tener un desarrollo adecuado.

**1. Introducción**

* **Contexto:** Eliminar la mención inicial sobre el problema del MTC y no mencionar directamente el concurso. Enfocar el inicio en el vacío de conocimiento (gap) y centrar la problemática en la ciudad de Lima.
* **Dataset:** No hacer ninguna mención al dataset en la introducción; esto debe reservarse estrictamente para su propia sección.
* **Estructura de párrafos:** Limitar la introducción de 1 a 6 párrafos distribuidos así:
* 1 a 6 párrafos para plantear el problema y sus limitaciones (por ejemplo, los problemas de detección).
* 2 párrafos dedicados a explicar la solución.
* 1 párrafo final indicando el objetivo.


* **Contribuciones:** Cambiar la lista numerada de contribuciones a un formato de párrafo redactado.

**2. Trabajos Relacionados (Related Works)**

* **Estructura:** Eliminar las separaciones por subsecciones para mantener una redacción fluida y continua.
* **Aprobación:** El "Summary and Research Gap" planteado está correcto.

**Fundamentos / Metodología**

* **Métricas:** Es indispensable hablar sobre las métricas. Expandir la explicación del rIoU a un párrafo completo y detallado.

**3. Dataset y Resultados**

* **Extensión:** Los párrafos que describen el dataset son muy cortos. Expandirlos para que tengan al menos entre 6 y 8 líneas.
* **Sección 3.3.6 (Known Annotation Bias):** Reconsiderar o eliminar esta subsección. Indicar que el dataset "está sucio" es exponer debilidades del trabajo y resulta contraproducente.
* **Sección 3.3.7 (Temporal Structure & Split Implication):** Es información valiosa, pero ocupa mucho espacio como sección independiente. Moverla e integrarla dentro de la sección "File Structure" (3.3.2).
* **Tablas:** Sintetizar y fusionar las tablas. La tabla de taxonomía (Table 2) es muy larga y debe combinarse. Eliminar las tablas redundantes y dejar solo la versión final.
* **Gráficos vs Tablas:** No duplicar información. Si hay una tabla, eliminar el gráfico correspondiente.
* Eliminar el gráfico de barras rojas (Figura 3).
* Corregir o eliminar el gráfico de barras verdes (Figura 2) porque presenta errores visuales.



**Secciones Finales**

* **Disponibilidad de Datos (Data availability):** Integrar las URLs directamente en una oración estructurada, no dejarlas sueltas.
