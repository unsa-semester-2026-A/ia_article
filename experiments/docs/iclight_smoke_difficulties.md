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
## 2026-07-26 — Quality parameters and foreground backing

The first real-data smoke run used `256 px`, two diffusion steps, `CFG=2`, and
disabled the high-resolution refinement. Those were execution-only settings,
not settings that can assess image quality. The official background-conditioned
demo defaults to 20 steps, `CFG=7`, high-resolution scale `1.5`, and high-res
denoise `0.5`; the quality smoke therefore uses those values and a simple
scene-specific prompt: `aerial view of a parked car on an asphalt road,
daylight, realistic photograph`.

The official callback first applies RMBG to the foreground RGB image. Passing
a full-frame transparent layer flattened against black made that black canvas
part of the image and led to dark, distorted output. The renderer now flattens
transparent pixels against white before uploading, allowing RMBG to identify
the vehicle as a conventional foreground. This still requires manual review:
IC-Light was not trained specifically for tiny aerial vehicles.

## 2026-07-26 — Full-frame background hallucination

The background-conditioned callback returns a newly generated *full frame*;
the background image is conditioning, not a pixel-preservation guarantee. In
the quality smoke it produced a large grey region over an otherwise correct
road image. The production renderer now composites only the dilated,
one-pixel-feathered alpha/OBB region of the generated output over the original
background. This preserves every road pixel outside the intended vehicle and
keeps the prompt/model responsible only for the local foreground appearance.

## 2026-07-26 — Comparación visual de diez ejemplos

Para evitar que la composición con bordes suaves oculte defectos del modelo,
cada ejemplo conserva cinco artefactos ordenados: `lama_background.jpg`,
`inserted_vehicles.png`, `direct_overlay.jpg`, `iclight_full.jpg` e
`iclight_soft_composite.jpg`. El último es el único que preserva el fondo fuera
del OBB; `iclight_full.jpg` se guarda deliberadamente para evaluar las
alucinaciones de fondo. Los diez ejemplos alternan las cinco clases previstas
(combi, microbús, ómnibus, articulado y mototaxi), dos veces cada una.
