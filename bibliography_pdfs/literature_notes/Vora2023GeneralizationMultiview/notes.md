# Bringing Generalization to Deep Multi-View Pedestrian Detection

- **Key**: Vora2023GeneralizationMultiview
- **Year**: 2023
- **Venue**: IEEE Winter Conference on Applications of Computer Vision Workshops (WACVW)

## Resumen
El artículo aborda la detección de peatones desde múltiples vistas (Multi-view Detection, MVD), que es muy eficaz para manejar oclusiones en entornos concurridos. Los autores señalan que, aunque los métodos de aprendizaje profundo han avanzado significativamente, han pasado por alto el aspecto de la generalización, lo que los hace poco prácticos para implementaciones en el mundo real. Formalizan tres formas críticas de generalización: variación en el número de cámaras, variación en las posiciones de las cámaras y adaptación a nuevas escenas. Demuestran que los modelos actuales sufren de sobreajuste severo a una sola configuración y escena. Para solucionarlo, proponen el conjunto de datos sintético GMVD (con variaciones de clima, iluminación, número y posición de cámaras) y adaptan una arquitectura base introduciendo una etapa de agregación con *average pooling* libre de parámetros (que provee invarianza de permutación) y una regularización llamada *DropView*. Los experimentos muestran mejoras sustanciales en la capacidad de generalización.

## Secciones y Subsecciones

### I. Introducción
Presenta el problema de la detección multi-vista de peatones (MVD), cuyo objetivo es producir un mapa de ocupación en el plano del suelo (bird's eye view). Discute cómo los métodos han evolucionado de lo clásico a lo end-to-end con deep learning. Argumenta que la evaluación homogénea actual oculta graves deficiencias de generalización y propone evaluar tres formas específicas: variación en el número de cámaras, variación de configuración espacial y generalización a escenas inéditas.
* **Problemas atacados**: La falta de practicidad y robustez de los modelos de MVD ante fallos en cámaras, cambios en la disposición física de los dispositivos, o la necesidad de desplegar el sistema en un entorno completamente nuevo.
* **Limitaciones de ese entonces**: Los métodos del estado del arte se evalúan en condiciones sumamente homogéneas (mismo número y posición de cámaras, misma escena, solapamiento casi total entre entrenamiento y prueba como en Wildtrack, donde solo varían unos segundos). Si una cámara falla o cambia el orden de las entradas, el rendimiento colapsa.
* **Soluciones alcanzadas**: Formalización de las tres dimensiones de la generalización, propuesta del dataset GMVD para evaluar estos escenarios y desarrollo de directrices de diseño (invarianza de permutación y regularización) para robustecer los modelos.

### II. Trabajos Relacionados
Revisa la evolución de la detección multi-vista agrupándola en tres categorías principales.

#### II.A. Métodos Clásicos
Aborda los primeros enfoques que proyectan probabilidades de ocupación en rejillas discretas mediante la sustracción de fondo.
* **Problemas atacados**: La estimación de la ocupación espacial agregando información visual de cámaras calibradas.
* **Limitaciones de ese entonces**: Degradación del rendimiento a medida que aumenta la densidad de la multitud y el desorden visual, ya que la sustracción de fondo se vuelve ineficaz.
* **Soluciones alcanzadas**: Uso de técnicas clásicas como campos aleatorios condicionales (CRF) o inferencia de campo medio para la agregación, o el reemplazo de la sustracción de fondo por clasificadores manuales.

#### II.B. MVD Basado en Anclajes (Anchors)
Describe los enfoques que integran detectores profundos de peatones de 2D (como Faster R-CNN, SSD, YOLO) para procesar las vistas de manera independiente o conjunta.
* **Problemas atacados**: El uso de características ricas de redes neuronales profundas para mejorar la detección monocular previa a la fusión.
* **Limitaciones de ese entonces**: Las imprecisiones de los cuadros de anclaje predefinidos limitan la precisión. Además, proyectar el punto exacto de contacto con el suelo a partir de cajas 2D es propenso a errores geométricos y requiere balances manuales fuera de la red.
* **Soluciones alcanzadas**: Sustitución de la sustracción de fondo por detectores de objetos neuronales y algoritmos de clustering geométrico.

#### II.C. MVD Profundo End-to-End
Se enfoca en arquitecturas modernas libres de anclajes (como MVDet, MVDeTr, SHOT) que proyectan mapas de características directamente al plano del suelo.
* **Problemas atacados**: Evitar los errores de proyección monocular 2D y la sintonización manual de potenciales CRF.
* **Limitaciones de ese entonces**: Dependencia de módulos de agregación espacial con parámetros aprendidos vinculados al orden y número fijo de cámaras. Esto produce un sobreajuste a la configuración física específica del dataset.
* **Soluciones alcanzadas**: Integración de transformaciones de perspectiva y convoluciones de agregación en una red totalmente diferenciable de extremo a extremo, mejorando hasta un 14% de MODA sobre métodos previos.

### III. Conjunto de Datos Propuesto (GMVD)
Describe el diseño y la recolección del dataset sintético Generalized Multi-View Detection (GMVD) utilizando GTAV y Unity.
* **Problemas atacados**: La carencia de conjuntos de datos diversos y de gran escala para entrenar y evaluar la generalización en MVD, dado que Wildtrack y MultiViewX tienen escenas únicas con configuraciones estáticas.
* **Limitaciones de ese entonces**: Recopilar datos reales MVD densamente anotados es sumamente costoso, invasivo (privacidad) y limitado logísticamente por el COVID. Los conjuntos sintéticos existentes carecían de suficiente variedad ambiental y presentaban colisiones entre peatones virtuales.
* **Soluciones alcanzadas**: Creación de GMVD con 7 escenas (1 de prueba reservada), variaciones de clima, iluminación horaria y ropa. Se simulan retrasos de sincronización reales (20-100 ms) en GTAV y se evitan colisiones usando PersonX en Unity. Es el dataset más grande con 125,000 fotogramas anotados.

### IV. Método Propuesto
Detalla la arquitectura de detección multi-vista diseñada para la generalización, libre de anclajes y adaptable a un número variable de cámaras.

#### IV.A. Extracción de Características y Transformación de Perspectiva
Describe la obtención de características 2D y su mapeo al plano BEV (Bird's Eye View).
* **Problemas atacados**: Representación robusta de las imágenes 2D de entrada en un espacio común plano (suelo).
* **Limitaciones de ese entonces**: El mapeo de características debe preservar la información espacial sin añadir parámetros específicos que fuercen al modelo a memorizar una geometría de cámara fija.
* **Soluciones alcanzadas**: Uso de un extractor ResNet-18 con convoluciones dilatadas para mantener alta resolución espacial, proyectando las características C-channel a la rejilla de suelo usando las matrices de proyección homográfica de las cámaras.

#### IV.B. Agregación Espacial
Explica cómo se combinan las proyecciones de las distintas cámaras para predecir la ocupación.
* **Problemas atacados**: Fusión de información de N cámaras en un único mapa de características común.
* **Limitaciones de ese entonces**: Métodos previos concatenan características o usan capas dedicadas para cada cámara, haciendo que el modelo dependa de una cantidad fija de cámaras y sea sensible a su ordenamiento.
* **Soluciones alcanzadas**: 
  1. *Average Pooling* a lo largo de las vistas para obtener invarianza ante permutación del orden de las cámaras y compatibilidad nativa con cualquier cantidad de ellas sin parámetros adicionales.
  2. Regularización *DropView* que descarta aleatoriamente una cámara durante el entrenamiento para evitar la memorización.
  3. Predicción final usando 3 capas de convolución dilatada sobre el mapa de ocupación.

#### IV.C. Función de Pérdida
Define el criterio de optimización del mapa probabilístico contra el mapa real.
* **Problemas atacados**: Desbalance de clases y falta de precisión en la forma de los picos de probabilidad en el plano.
* **Limitaciones de ese entonces**: Funciones tradicionales como MSE no capturan la correlación espacial o la divergencia de distribución de forma óptima para la generalización.
* **Soluciones alcanzadas**: Uso empírico combinado de Divergencia de Kullback-Leibler (KLDiv) y la Correlación Cruzada de Pearson (CC), que maximiza la precisión de localización y minimiza los falsos positivos en escenas inéditas.

### V. Experimentos
Presenta los resultados cuantitativos y cualitativos comparando el método con el estado del arte.

#### V.A. Configuración Experimental
Describe los datasets usados (Wildtrack, MultiViewX, GMVD) y las métricas (MODA, MODP, Precision, Recall).
* **Problemas atacados**: Validación objetiva y justa de las capacidades de generalización.
* **Limitaciones de ese entonces**: Métodos previos solo reportan MODA en las particiones de prueba estándar de Wildtrack/MultiViewX que sufren de alto solapamiento con el entrenamiento.
* **Soluciones alcanzadas**: Planteamiento de pruebas específicas de "cámaras caídas" (entrenar con 7, probar con 4), "configuraciones cruzadas" y "transferencia de dominio sintético a real" (entrenar en MultiViewX, probar en Wildtrack).

#### V.B. Detalles de Implementación
Describe los parámetros de hardware, hiperparámetros de red y preprocesamiento de imágenes.
* **Problemas atacados**: Asegurar la reproducibilidad del entrenamiento de MVD.
* **Limitaciones de ese entonces**: Dificultades de entrenamiento estable con optimizadores estándar y susceptibilidad al sobreajuste si se entrena desde cero.
* **Soluciones alcanzadas**: Uso de pesos preentrenados de ImageNet en la ResNet-18, optimizador SGD con momento, planificador de tasa de aprendizaje One-Cycle y entrenamiento rápido en una sola GPU GTX 1080 Ti.

#### V.C. Resultados
Presenta y analiza las tablas de métricas obtenidas.
* **Problemas atacados**: Demostrar empíricamente las mejoras de generalización.
* **Limitaciones de ese entonces**: Los métodos previos colapsan ante el cambio de cámaras o escenas (p. ej., MODA de MVDet baja a 16.2% o 38.9% al quitar cámaras).
* **Soluciones alcanzadas**: El modelo propuesto con *DropView* supera a SHOT y MVDeTr en todos los retos de generalización por amplios márgenes (logrando hasta 77.0% y 79.2% de MODA con menos cámaras, y 70.7% de MODA en generalización de escena). Al entrenar en el dataset GMVD, se alcanza una MODA del 80.1% en Wildtrack sin haber visto ninguna imagen real durante el entrenamiento.

### VI. Discusión y Trabajo Futuro
Analiza el estado actual de la MVD y los siguientes pasos en la investigación.
* **Problemas atacados**: Identificación de los cuellos de botella remanentes para la adopción práctica de MVD.
* **Limitaciones de ese entonces**: La brecha de dominio entre lo sintético y lo real no se mitiga explícitamente en el pipeline actual.
* **Soluciones alcanzadas**: Reconocimiento de que Wildtrack debe usarse solo para pruebas y sugerencia del uso futuro de técnicas de adaptación de dominio no supervisada (UDA) para cerrar la brecha de distribución de características sintéticas/reales.

### VII. Conclusión
Sintetiza las contribuciones del trabajo.
* **Problemas atacados**: Rediseño del paradigma de desarrollo y evaluación de la detección multi-vista de peatones.
* **Limitaciones de ese entonces**: Enfoques previos excesivamente ajustados a entornos específicos que resultan inutilizables ante variaciones físicas cotidianas.
* **Soluciones alcanzadas**: Demostración de que la simplicidad del average pooling y la regularización DropView son suficientes para dotar de generalización práctica a los modelos profundos de MVD, complementado con el benchmark GMVD.
