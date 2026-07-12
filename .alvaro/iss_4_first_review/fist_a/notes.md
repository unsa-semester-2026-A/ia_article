# Notas de la investigación

## Prompt 1

Prompt: `How are oriented bounding boxes (OBB) used for vehicle detection in aerial imagery?`

**Resultado**

- ventajas del OBB
  - lo más importante aquí es que se usa pra la dirección correcta.
- dice que se podría usar el downstream traffic analysis, pero para eso creoo
  que sería necesrio preservar la dirección del vehículo
- dice que la dirección correcta es importante
- el set de entrenamiento del mtc tiene el ángulo de inclinación correcto, pero
  la dirección no
- algunos datasets:
  - VSAI: uses traditional cameras, vehicles only, OBB
  - HIT-UAV: thermal cameras, even people and bicycles, standard and obb

- primeros:
  - primero era detección horizontal y luego recién el ángulo
- luego fue e2e. 
- desafíos:
  - predecir el ángulo correctamente
  - **periodicity**: cuando en lugar de decir 0deg, dices 359 deg. Son casi
    idénticos pero el problema es que la función de pérdida no piensa igual.
- en uav ayudan a separar los vehículos que se solapan. 
- angle-free: no usar ángulo, sino otros datos que también lo impliquen
- anchor-free: no tener lados predefinidos
