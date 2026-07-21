# Plan de Refactorización Detallado - Inferencia Base 0

Este documento contiene la planificación final simplificada y desacoplada para realizar la inferencia, tracking y recolección de métricas de hardware de la **Base 0** (Zero-Shot) en el SMART Challenge 2026.

---

## 1. Contexto del Proyecto
Se está desarrollando un pipeline de visión computacional para investigación (SMART Challenge 2026, MTC Perú) utilizando YOLO26s-OBB. El objetivo actual es implementar la ejecución de la línea base "Base 0". Consiste en realizar inferencia Zero-Shot con tracking sobre el conjunto de validación en Kaggle, y registrar el consumo de hardware y tiempos de inferencia puros para persistir todos los datos en formato JSON para evaluaciones futuras (el filtrado temporal de vehículos estáticos, homografías inter-frame y cálculos de métricas se realizarán en una etapa de post-procesamiento posterior).

---

## 2. Objetivo de la Tarea
Refactorizar el código para centrarse únicamente en la inferencia, tracking y guardado de datos usando Programación Orientada a Objetos (patrón Template Method e Inyección de Dependencias), removiendo los módulos de homografía, evaluación y tracking intermedio innecesarios, y estructurando la persistencia en formato JSON por lotes (evitando desbordamiento de RAM/OOM).

---

## 3. Estructura de Directorios Requerida
El código fuente bajo `src/` se organizará de la siguiente forma:

```text
src/
├── inference/
│   ├── __init__.py
│   ├── base_inference.py       # Clase abstracta para manejo del modelo, hardware y métricas.
│   └── runners/
│       └── run_base_0.py       # Script principal de ejecución para la Base 0.
└── utils/
    ├── __init__.py
    └── io_manager.py           # I/O: lectura de imágenes, metadatos y subida a Google Drive.
```

---

## 4. Firmas y Especificaciones de Clases (API Contracts)

### 4.1 `base_inference.py` (`BaseInferencePipeline`)
Clase base abstracta de la que heredan todos los runners del estudio de ablación. Define la interfaz de ejecución y controla el monitoreo de hardware.

```python
from abc import ABC, abstractmethod
from typing import Any

class BaseInferencePipeline(ABC):
    """Clase base abstracta para orquestar la inferencia de modelos YOLO OBB."""

    def __init__(self, config: dict[str, Any], model_path: str) -> None:
        """Inicializa parámetros base de hardware, modelo y contadores.

        Args:
            config: Configuración general (device, conf, iou, imgsz, etc.).
            model_path: Ruta a los pesos del modelo YOLO (.pt).
        """
        self.config: dict[str, Any] = config
        self.model_path: str = model_path
        self.device: str | int = config.get("device", 0)
        
        # Historial de tiempos y consumo de hardware
        self.preprocess_times: list[float] = []
        self.inference_times: list[float] = []
        self.postprocess_times: list[float] = []
        
        self.total_frames_processed: int = 0
        self.start_time: float = 0.0
        self.total_time_seconds: float = 0.0
        self.peak_cpu_ram_gb: float = 0.0
        self.peak_gpu_vram_gb: float = 0.0

    def start_hardware_monitoring(self) -> None:
        """Reinicia las estadísticas de la GPU (VRAM) usando torch.cuda.reset_peak_memory_stats 
        y registra el tiempo de inicio total.
        """
        pass

    def record_hardware_metrics(self) -> dict[str, Any]:
        """Consulta el consumo máximo de memoria RAM (resource.ru_maxrss) y VRAM (torch.cuda.max_memory_allocated).
        Calcula también la velocidad promedio del modelo.

        Returns:
            Estructura compatible con 'inference_metrics.json'.
        """
        pass

    @abstractmethod
    def execute(self) -> dict[str, Any]:
        """Ejecuta el pipeline completo de inferencia y guardado por lotes."""
        pass
```

### 4.2 `io_manager.py` (`IOManager`)
Clase gestora de entrada/salida genérica y totalmente desacoplada de rutas de carpetas específicas y IDs constantes de Google Drive. Esto permite reutilizarla en cualquier fase de inferencia (Base 0, Base 1, Base 2, etc.) inyectando los parámetros en las llamadas correspondientes.

```python
from typing import Any
import numpy as np

class IOManager:
    """Gestor genérico de datos de entrada/salida locales e integración con la API de Google Drive."""

    def __init__(self, token_path: str | None = None) -> None:
        """Inicializa el gestor de E/S.

        Args:
            token_path: Ruta al archivo token.json de Google Drive para inicialización del servicio.
        """
        self.token_path: str | None = token_path
        self.drive_service: Any | None = None

    def list_files_in_dir(self, dir_path: str, extension: str = None) -> list[str]:
        """Lista archivos en un directorio local, opcionalmente filtrados por extensión.

        Args:
            dir_path: Directorio a escanear.
            extension: Extensión para filtrar (ej. '.jpg').

        Returns:
            Lista ordenada de nombres de archivo.
        """
        pass

    def load_csv(self, file_path: str) -> list[dict]:
        """Carga un archivo CSV genérico y retorna sus filas como diccionarios.

        Args:
            file_path: Ruta al archivo CSV.

        Returns:
            Lista de diccionarios representando cada fila.
        """
        pass

    def load_image(self, image_path: str) -> np.ndarray:
        """Carga una imagen desde el disco usando OpenCV.

        Args:
            image_path: Ruta absoluta al archivo de imagen.

        Returns:
            Matriz numpy de la imagen cargada (BGR).
        """
        pass

    def save_json(self, data: Any, local_path: str) -> str:
        """Guarda un diccionario o lista en formato JSON en la ruta local indicada.

        Args:
            data: Contenido a serializar.
            local_path: Ruta destino en disco.

        Returns:
            Ruta absoluta del archivo JSON persistido.
        """
        pass

    def _get_drive_service(self) -> Any | None:
        """Inicializa y retorna la conexión OAuth2 con Google Drive API.
        
        Usa las credenciales indicadas en self.token_path si están disponibles.
        """
        pass

    def upload_file_to_drive(self, local_path: str, drive_folder_id: str, mime_type: str = "application/json") -> str | None:
        """Sube un archivo local a una carpeta específica en Google Drive. No contiene IDs por defecto.

        Args:
            local_path: Ruta al archivo en disco.
            drive_folder_id: ID de la carpeta destino de Google Drive (Debe ser inyectado, NUNCA por defecto).
            mime_type: Tipo de contenido MIME.

        Returns:
            El ID del archivo subido en Google Drive, o None si ocurre un fallo.
        """
        pass
```

### 4.3 `run_base_0.py` (`Base0Runner`)
El script orquestador inyectará las configuraciones locales y del Drive necesarias en sus métodos.

```python
from typing import Any
from src.inference.base_inference import BaseInferencePipeline

class Base0Runner(BaseInferencePipeline):
    """Pipeline de ejecución de inferencia Zero-Shot para la Base 0."""

    def __init__(self, config: dict[str, Any], model_path: str) -> None:
        """Inicializa el modelo YOLO y resuelve dinámicamente las clases de vehículos.

        Mapeo dinámico:
        VEHICLE_CLASS_IDS = []
        for idx, name in model.names.items():
            clean_name = name.lower().replace('-', ' ').replace('_', ' ')
            if any(keyword in clean_name for keyword in ['small vehicle', 'large vehicle', 'car', 'bus', 'truck', 'vehicle']):
                VEHICLE_CLASS_IDS.append(int(idx))
                
        # Según la verificación, para YOLO26m-OBB en DOTA esto mapea correctamente a [9, 10]:
        # Dynamically resolved vehicle class IDs: [9, 10]

        Mapeo a Clases MTC (0 a 8 para las 9 clases vehiculares, 0-indexed):
        - Clases tipo 'small vehicle'/'car' -> MTC 0 (auto)
        - Clases tipo 'large vehicle'/'bus'/'truck' -> MTC 6 (camion)
        - Ignora otras categorías.
        """
        super().__init__(config, model_path)
        # Inicializa self.model = YOLO(model_path)
        # Mapea dinámicamente usando VEHICLE_CLASS_IDS.
        # Inicializa self.io_manager pasándole config.get("token_path").

    def execute(self) -> dict[str, Any]:
        """Ejecuta el pipeline completo de inferencia:
        1. Obtiene los clips de validación con self.io_manager.get_validation_clips().
        2. Procesa secuencialmente cada clip, frame por frame.
        3. Registra preprocess, inference y postprocess del forward pass.
        4. Formatea los outputs frame a frame (extrayendo xyxyxyxy para obb_corners, mapeando clases, etc.).
        5. Guarda localmente y subepredictions_raw.json a Google Drive por cada clip finalizado inyectando
           el ID de carpeta de la Base 0: "1wXieZvOZDE5KzZiYyESbf8xU-C2AGPWJ".
        6. Al finalizar todos los clips, genera y sube inference_metrics.json.

        Returns:
            Diccionario con las rutas locales y IDs de archivo en Google Drive de los JSONs generados.
        """
        pass
```

---

## 5. Auditoría de Variables de la API de YOLO26

Para verificar la factibilidad técnica del pipeline sin cálculos posteriores, validamos la obtención de las siguientes variables directamente de la API de Ultralytics YOLO26:

1. **Tiempos de velocidad desglosados (`speed_ms`)**:
   - YOLO26 retorna en su objeto `Results` un diccionario `results[0].speed` que contiene tiempos en milisegundos por frame:
     - `results[0].speed['preprocess']`
     - `results[0].speed['inference']`
     - `results[0].speed['postprocess']`
   - Esto permite registrar el tiempo exacto de preprocesamiento, inferencia pura del modelo, y postprocesamiento NMS (si aplica) de forma nativa.
2. **Dimensiones de la imagen original e inferencia**:
   - `results[0].orig_shape` retorna una tupla `(height, width)` con la resolución original de la imagen en disco.
   - El tamaño de inferencia (`inference_shape`) es el parámetro configurado e inyectado al modelo (`imgsz=640`).
3. **Vértices del polígono orientado (`obb_corners`)**:
   - El objeto `results[0].obb.xyxyxyxy` contiene los 4 vértices del polígono orientado como un tensor de coordenadas `(N, 4, 2)`.
   - Para registrar el formato plano requerido de 8 flotantes (`[x1, y1, x2, y2, x3, y3, x4, y4]`), basta con aplanar y convertir el tensor a lista en Python:
     `results[0].obb.xyxyxyxy[idx].cpu().numpy().flatten().tolist()`

---

## 6. Estructura Exacta de los Archivos de Datos a Guardar

### Archivo 1: `predictions_raw.json` (Uno por Clip)
Este archivo se guardará localmente y se subirá a Google Drive al finalizar de procesar cada `clip_id`.
```json
{
  "clip_id": "v_01tp2eyell",
  "inference_shape": [640, 640],
  "frames": [
    {
      "frame_idx": 15,
      "original_shape": [1080, 1920],
      "speed_ms": {
        "preprocess": 1.2,
        "inference": 15.4,
        "postprocess": 2.1
      },
      "detections": [
        {
          "track_id": 42,
          "class_id": 0,
          "score": 0.87,
          "obb_corners": [10.5, 20.1, 50.0, 20.1, 50.0, 80.5, 10.5, 80.5]
        },
        {
          "track_id": -1,
          "class_id": 6,
          "score": 0.26,
          "obb_corners": [100.0, 40.0, 140.0, 40.0, 140.0, 70.0, 100.0, 70.0]
        }
      ]
    }
  ]
}
```

### Archivo 2: `inference_metrics.json` (Único al final)
Generado una sola vez al finalizar todo el lote de validación.
```json
{
  "experiment_condition": "Base_0_Zero_Shot",
  "hardware": "Tesla_T4",
  "total_frames_processed": 10852,
  "total_time_seconds": 185.3,
  "peak_vram_mb": 4500,
  "average_speed_ms": {
    "preprocess": 1.1,
    "inference": 14.8,
    "postprocess": 1.9
  },
  "theoretical_fps": 55.5
}
```

---

## 7. Restricciones Técnicas y de Memoria (Kaggle/Colab)
* **Gestión de Memoria RAM (Evitar OOM)**: Queda **PROHIBIDO** el uso de DataFrames de Pandas para acumular las predicciones de los frames a lo largo del bucle. El proceso de inferencia debe acumular los datos en diccionarios y listas nativas de Python, realizar el `json.dump()` por lotes (es decir, guardar el JSON y subirlo a Drive al finalizar de procesar cada `clip_id`), y liberar memoria mediante el recolector de basura (`gc.collect()`).
* **Hiperparámetros de Confianza y Detección**:
  - `conf = 0.001` (Adecuado umbral extremo acordado. Vital capturar todas las detecciones de bajísima confianza para reconstruir perfectamente la curva Precision-Recall durante la evaluación posterior sin sesgos).
  - `iou = 0.45`
  - `imgsz = 640`
