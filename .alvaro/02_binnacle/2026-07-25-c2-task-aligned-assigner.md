# Incidencia operativa: memoria en C2 (Base 2)

Fecha: 2026-07-25

La ejecución C2 con `batch=96` (48 imágenes por cada una de las dos Tesla T4)
mostró repetidamente `CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU`.
El entrenamiento no se detuvo, pero la asignación de objetivos pasó a CPU y el
tiempo por época dejó de ser representativo; por ello se canceló antes de usar
más cuota de GPU.

Se incorporó una calibración operativa, separada de los resultados del artículo.
Prueba los lotes globales `96 → 48 → 32 → 24` sobre 384 imágenes comunes
raw/LaMa con mayor número de instancias y con el perfil de aumentación C2
intacto (Mosaic y MixUp incluidos). El primer candidato que no emite aquel
aviso queda fijado en `c2_batch_selection.json`; los ensayos no generan ni
suben checkpoints a Drive.

Si se necesita un lote menor que 96, se conserva un lote efectivo global de
96 mediante `nbs=96` y acumulación de gradientes, y se ajusta el decaimiento de
pesos para mantener el escalado de la receta original. Así cambia únicamente
la configuración de recursos de C2, no los datos, las etiquetas, la semilla,
las épocas, la paciencia ni la partición de validación. Si 96, 48 y 32 fallan,
el respaldo previsto es 24; si tampoco 24 resulta limpio, C2 no debe lanzarse
hasta rediseñar la receta y documentarlo como una nueva decisión experimental.

La base técnica es la documentación de Ultralytics: `TaskAlignedAssigner`
captura el OOM y usa CPU; en DDP no se aplica el reintento automático de lote.
Además, Ultralytics define `nbs` como lote nominal y calcula la acumulación a
partir de él.
