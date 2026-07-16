# Fase 1: Limpieza con LaMa (04_lama_cleaning.md)

Este documento detalla las especificaciones de diseño, flujo de ejecución y control de calidad para la remoción de píxeles de vehículos estacionados no anotados utilizando el modelo de inpainting **LaMa (Large Mask Inpainting)**.

---

## 1. Objetivo
Sustituir los píxeles de los vehículos estacionados detectados en la Fase 0 por texturas realistas y coherentes del entorno (asfalto, veredas, marcas viales), eliminando de raíz la señal contradictoria en las imágenes de entrenamiento sin alterar las anotaciones de las cajas en movimiento.

---

## 2. Especificación de la Herramienta LaMa

* **Modelo Base:** `big-lama` (pesos pre-entrenados en el dataset Places2 de Yandex/Hugging Face). El modelo utiliza Convoluciones de Fourier Rápidas (FFCs) que capturan características globales y locales de textura en una sola pasada.
* **Licencia:** Apache License 2.0.
* **Entorno de Ejecución:** Inferencia acelerada por GPU en la nube (Google Colab / Kaggle). No se requiere entrenamiento ni ajuste de pesos.
* **Requisitos de Hardware:**
  - VRAM: ~4 GB para imágenes de resolución 1080p en batch size de 1.
  - Tiempo de procesamiento estimado: $0.15 - 0.3$ segundos por imagen en una GPU NVIDIA Tesla T4 o P100 (Google Colab / Kaggle).

---

## 3. Módulo: Generador y Dilatador de Máscaras Binarias

Antes de pasar la imagen a LaMa, las detecciones de la Fase 0 (en formato de caja rotada paramétrica) deben convertirse en una máscara binaria (imagen en escala de grises de 1 canal).

* **Entrada:** Imagen original (ancho $W$, alto $H$, 3 canales), lista de OBBs a borrar `[(cx, cy, w, h, angle_deg), ...]`.
* **Proceso de Generación de Máscara:**
  1. Inicializar una imagen en negro (valores a 0) de dimensiones $H \times W$.
  2. Para cada OBB en la lista:
     - Calcular los 4 vértices del polígono rotado.
     - **Dilatación Geométrica:** Incrementar las dimensiones $w$ y $h$ en **10 píxeles** en cada dirección (nuevo ancho: $w + 20$, nuevo alto: $h + 20$). Esto garantiza que la máscara cubra no solo la carrocería del vehículo, sino también las sombras proyectadas en el asfalto y los bordes difusos del vehículo que podrían dejar artefactos tras el inpainting.
     - Dibujar el polígono rotado dilatado relleno con color blanco (valor 255) en la máscara.
  3. Si la lista de OBBs para el frame está vacía, retornar una máscara de puros ceros.

---

## 4. Módulo: Procesamiento en Batch e Inpainting

Este módulo coordina el flujo de procesamiento masivo sobre todo el dataset de entrenamiento.

* **Entrada:** Directorio de imágenes originales, JSON de máscaras de la Fase 0, directorio de destino.
* **Lógica de Ejecución:**
  - **Caso A (Frame con vehículos a borrar):** Leer imagen original, generar la máscara binaria combinada, pasar la imagen y la máscara al modelo LaMa, y guardar la imagen resultante con el mismo nombre y extensión en el directorio destino.
  - **Caso B (Frame sin vehículos a borrar):** Copiar la imagen original directamente al directorio de destino sin realizar ningún procesamiento (ahorro de tiempo y cómputo).
  - **Caso C (Frame vacío / Target=none):** Copiar directamente.
* **Restricción de Entrada/Salida:** La imagen de salida debe tener **exactamente las mismas dimensiones** (ancho y alto) y formato (JPEG) que la imagen de entrada. Las anotaciones correspondientes en el dataset no se modifican, ya que solo estamos borrando los píxeles de los autos estacionados (los cuales no tenían cajas anotadas en el GT original).

---

## 5. Distribución de Cómputo (Paralelización en Google Colab/Kaggle)

Para procesar las 54,262 imágenes de forma rápida y gratuita, se utiliza Google Colab (VM con GPU T4) o notebooks de Kaggle en paralelo:

1. **Estrategia de Ejecución en Colab:**
   - Descargar el zip del dataset desde Google Drive directamente al disco efímero de la VM de Colab (ancho de banda >100 MB/s).
   - Ejecutar la limpieza de LaMa por lotes (batch size = 4 u 8) aprovechando los 16 GB de VRAM de la GPU T4.
   - **Optimización de Espacio (Redimensionamiento):** Durante el proceso de inpainting, cada imagen resultante se redimensiona a 640x640. Esto permite reducir el tamaño del dataset limpio a aproximadamente ~4.3 GB, garantizando que quepa sin problemas en los límites de almacenamiento persistente de Kaggle (20 GB).
2. **Consolidación en Drive:**
   - Comprimir las imágenes modificadas a un archivo zip (`smart_lama_640.zip`) y guardarlo en el Google Drive Pro de 1 TB.
3. **Tiempo Estimado:**
   - La inferencia de LaMa en T4 optimizada por lotes toma aproximadamente $0.2$ seg por imagen.
   - 54,262 imágenes a $0.2$ seg/img ≈ 10,852 segundos ≈ **3.0 horas** en una sola GPU T4 (o menos si se divide la tarea en 2-3 notebooks de Colab ejecutándose en paralelo bajo cuentas de distintos integrantes).

---

## 6. Estructura de Directorios Resultante

```
dataset_lama_cleaned/
├── train/
│   ├── images/   <── Imágenes limpiadas con LaMa (vehículos estacionados borrados)
│   └── labels/   <── Labels originales de YOLO-OBB (sin modificar)
└── val/
    ├── images/   <── Imágenes reales ORIGINALES (nunca modificadas por LaMa)
    └── labels/   <── Labels originales de YOLO-OBB
```

> [!IMPORTANT]
> **REGLA METODOLÓGICA:** El split de validación (`val/images`) **nunca debe ser alterado por LaMa**. El modelo debe ser evaluado contra el mundo real para comprobar si el detector final aprendió rasgos generales o si depende de artefactos generativos.

---

## 7. Protocolo de Auditoría Visual (Visual Quality Audit)

Una vez finalizado el proceso de inpainting, se realiza un control de calidad aleatorio:

1. **Muestreo:** Seleccionar de forma aleatoria $100$ imágenes donde se haya ejecutado el inpainting de LaMa.
2. **Criterios de Inspección:**
   - **Fusión de Texturas:** Las líneas de carril continuas o segmentadas deben continuar su trayectoria lineal de forma natural a través del área borrada.
   - **Ausencia de Borrones:** La textura del asfalto debe imitar el ruido visual de la calle real (sin parches lisos o borrosos exagerados).
   - **Precisión de Remoción:** Ningún vehículo en movimiento anotado (con caja en el GT) debe mostrar signos de haber sido borrado o deformado.
3. **Métrica de Calidad:**
   - Tasa de artefactos visuales graves $< 5\%$.
   - Tasa de borrado erróneo de objetos con anotaciones $= 0\%$.
4. Si falla la auditoría, re-calibrar la Fase 0 (pseudo-labeling) o ajustar el nivel de dilatación de la máscara.
