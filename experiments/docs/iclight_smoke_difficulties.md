# IC-Light: dificultades y decisiones de la prueba de humo

## Alcance

La prueba de humo procesa un lote acotado de imágenes reales a través del mismo
camino de geometría, letterbox, API local de IC-Light y restauración a 640×360
que usará el renderizado. Solo comprueba integridad de ejecución. La aceptación
visual de iluminación, sombras y realismo sigue siendo una revisión humana.

## Incidencias ya resueltas

| Dificultad | Causa comprobada | Corrección aplicada |
|---|---|---|
| Conflicto TensorFlow/Protobuf al arrancar | Dependencias opcionales de Transformers intentaban cargar TensorFlow | Se fija `USE_TF=0` y `TRANSFORMERS_NO_TF=1` en el entorno aislado. |
| La API `/process_relight` no aparecía | El demo upstream no expone de forma estable un endpoint de imagen única | El instalador instrumenta una ruta explícita, descubierta por `ICLightClient`. |
| `ValueError` vacío tras completar difusión | La cola/WebSocket de Gradio y la serialización de `Gallery` fallaban en el entorno Kaggle | Se desactiva la cola para la API local y el endpoint devuelve una sola `Image`. |
| Error al enviar un crop RGBA | `run_rmbg` upstream exige exactamente tres canales | El alpha se usa solo para la geometría y se convierte BGRA→BGR antes de IC-Light. |
| Métricas de GPU insuficientes | Las asignaciones de otros procesos no aparecen en métricas de PyTorch del proceso | Se muestrea `nvidia-smi` durante todo el lote y se conservan pico, media y muestras por GPU. |
| Salida ampliada y borrosa en smoke a 256 px | La petición pedía a IC-Light 512 px, pero `undo_letterbox` recortaba usando la geometría de 256 px | `image_width` e `image_height` ahora usan el mismo `working_size` que el pre/postprocesado. |

## Evidencia que debe guardarse en cada prueba

- `iclight_smoke_metrics.json`: parámetros, duración total y por imagen, RAM pico,
  GPU, salida, dimensiones, tamaño y errores.
- `images/*.jpg`: resultados para revisión humana, sin sobrescribir los de otro
  lote.
- Log del servidor de IC-Light: conserva trazas de callback ante un fallo.

## Límites conocidos

- IC-Light upstream usa una GPU por proceso. Esta prueba de una instancia debe
  mostrar actividad solo en esa GPU; aprovechar ambas GPUs será una decisión del
  render masivo mediante procesos aislados, no un requisito de esta prueba.
- La prueba no certifica que el slot sea semánticamente realista ni que esté
  libre de objetos. Esas reglas pertenecen a la selección de slots y al control
  visual posterior.
