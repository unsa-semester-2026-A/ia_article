# Relevo: prueba visual IC-Light en Kaggle

Fecha de relevo: 2026-07-26 (America/Lima).

## Ejecución activa

- Notebook privado: `alvaroquispeunsa/ic-light-real-batch-smoke`
- URL: <https://www.kaggle.com/code/alvaroquispeunsa/ic-light-real-batch-smoke>
- Última versión enviada: **11**.
- Rama que el notebook clona: `chore/augmentation-evidence`.
- Dataset adjunto esperado: `alvaroquispeunsa/mtc-challenge`.

La ejecución ocurre en Kaggle; apagar el ordenador local no la detiene. Al
retomar, comprobar el estado y descargar resultados con:

```bash
kaggle kernels status alvaroquispeunsa/ic-light-real-batch-smoke
kaggle kernels output alvaroquispeunsa/ic-light-real-batch-smoke -p /tmp/iclight-v11-output -o
```

## Alcance de la versión 11

Se generan diez ejemplos de revisión humana. Las cinco clases planificadas se
alternan dos veces cada una: combi, microbús, ómnibus, articulado y mototaxi.
La selección usa un orden aleatorio determinista y lee como máximo 2,000
etiquetas para obtener los crops, en vez de recorrer todo el conjunto.

Para cada ejemplo, el directorio
`iclight_real_batch_smoke/comparisons/frame_*` debe contener:

1. `raw_original.jpg`: fotograma crudo de referencia.
2. `lama_background.jpg`: fondo LaMa usado por la inferencia.
3. `inserted_vehicles.png`: vehículos RGBA escogidos y colocados según OBB.
4. `direct_overlay.jpg`: LaMa más vehículos por composición alfa directa.
5. `iclight_full.jpg`: salida completa del modelo, sin ocultar alucinaciones.
6. `iclight_soft_composite.jpg`: salida final que usa IC-Light solo en el OBB
   dilatado/suavizado y conserva el fondo LaMa exterior.

También se esperan `comparison_manifest.json` e
`iclight_smoke_metrics.json`, que incluyen clases, rutas, semillas, tiempos,
RAM y utilización de GPU.

## Parámetros vigentes

- Prompt: `aerial view of a parked car on an asphalt road, daylight, realistic photograph`.
- Prompt positivo: `best quality`.
- Pasos: 20.
- CFG: 7.
- Refinamiento: escala 1.5, denoise 0.5.
- Resolución: 576×320, restaurada a 640×360.

## Criterio de continuación

No usar ninguna imagen para entrenamiento sin inspección humana de los cinco
artefactos por ejemplo. En particular, comparar `iclight_full.jpg` frente a
`iclight_soft_composite.jpg`: la segunda debe eliminar daños del fondo sin
recortar ni degradar visiblemente el vehículo.
