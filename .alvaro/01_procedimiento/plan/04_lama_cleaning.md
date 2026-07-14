# Fase 1: Limpieza con LaMa (04_lama_cleaning.md)

Este documento detalla las especificaciones de diseño, flujo de ejecución y control de calidad para la remoción de píxeles de vehículos estacionados no anotados utilizando el modelo de inpainting **LaMa (Large Mask Inpainting)**.

---

## 1. Objetivo
Sustituir los píxeles de los vehículos estacionados detectados en la Fase 0 por texturas realistas y coherentes del entorno (asfalto, veredas, marcas viales), eliminando de raíz la señal contradictoria en las imágenes de entrenamiento sin alterar las anotaciones de las cajas en movimiento.

---

## 2. Especificación de la Herramienta LaMa

* **Modelo Base:** `big-lama` (pesos pre-entrenados en el dataset Places2 de Yandex/Hugging Face). El modelo utiliza Convoluciones de Fourier Rápidas (FFCs) que capturan características globales y locales de textura en una sola pasada.
* **Licencia:** Apache License 2.0.
* **Entorno de Ejecución:** Inferencia local pura. No se requiere entrenamiento ni ajuste de pesos.
* **Requisitos de Hardware:**
  - VRAM: ~4 GB para imágenes de resolución 1080p en batch size de 1.
  - Tiempo de procesamiento estimado: $0.3 - 0.5$ segundos por imagen en una GPU NVIDIA GTX 1070.

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

## 5. Distribución de Cómputo (Paralelización Local)

Para procesar las 54,262 imágenes rápidamente, el trabajo se distribuye en las 3 PCs del laboratorio equipadas con GTX 1070:

1. **Partición del Dataset:** Dividir el conjunto de clips de entrenamiento de forma equitativa.
   - PC 1 procesa del clip 1 al 362 (~18,000 imágenes).
   - PC 2 procesa del clip 363 al 724 (~18,000 imágenes).
   - PC 3 procesa del clip 725 al 1088 (~18,262 imágenes).
2. Cada PC ejecuta el script de batch de forma local leyendo de su disco rígido y guardando en una carpeta local temporal.
3. Al finalizar, se consolidan las carpetas en un único dataset limpio en el Google Drive Pro de 1 TB.
4. **Tiempo Estimado:**
   - 18,000 imágenes a $0.4$ seg/img = 7,200 segundos ≈ **2.0 horas** de ejecución en paralelo.

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
