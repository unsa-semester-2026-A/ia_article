import gc
import sys
import time
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from src.inference.base_inference import BaseInferencePipeline
from src.utils.io_manager import IOManager

# Tamaño de lote para inferencia en GPU.
# Ajustar según la VRAM disponible (16 es conservador para Tesla T4 con imgsz=640).
_DEFAULT_BATCH_SIZE = 16


class Base0Runner(BaseInferencePipeline):
    """Pipeline de ejecución de inferencia Zero-Shot para la Base 0."""

    def __init__(self, config: dict[str, Any], model_path: str) -> None:
        """Inicializa el modelo YOLO, resuelve dinámicamente las clases de vehículos
        y construye un diccionario de mapeo directo DOTA -> MTC.
        """
        super().__init__(config, model_path)
        self.model = YOLO(model_path)
        self.io_manager = IOManager(token_path=config.get("token_path"))
        self.batch_size: int = config.get("batch_size", _DEFAULT_BATCH_SIZE)

        # Construir diccionario de mapeo directo {dota_class_id: mtc_class_id}
        # en el __init__ para evitar recalcular por cada detección en el bucle.
        self.dota_to_mtc: dict[int, int] = {}
        for idx, name in self.model.names.items():
            clean = name.lower().replace("-", " ").replace("_", " ")
            if any(kw in clean for kw in ("small vehicle", "car")):
                self.dota_to_mtc[int(idx)] = 0   # MTC: auto
            elif any(kw in clean for kw in ("large vehicle", "bus", "truck")):
                self.dota_to_mtc[int(idx)] = 6   # MTC: camión

        print(f"[Base0Runner] Mapeo DOTA -> MTC construido: {self.dota_to_mtc}")

    def _process_batch_results(
        self, results: list, frame_offset: int
    ) -> list[dict[str, Any]]:
        """Extrae datos de un lote de resultados YOLO y acumula métricas de velocidad.

        Args:
            results: Lista de objetos Results retornados por model.predict().
            frame_offset: Índice del primer frame del lote dentro del clip.

        Returns:
            Lista de diccionarios frame_data listos para insertar en el JSON del clip.
        """
        batch_frames: list[dict[str, Any]] = []

        for i, r in enumerate(results):
            # Acumular tiempos en la clase base (suma + conteo, sin listas)
            self.accumulate_speed(r.speed)

            frame_data: dict[str, Any] = {
                "frame_idx": frame_offset + i,
                "original_shape": list(r.orig_shape),
                "speed_ms": r.speed,
                "detections": [],
            }

            if r.obb is not None and len(r.obb):
                corners = r.obb.xyxyxyxy.cpu().numpy()   # (N, 4, 2)
                classes = r.obb.cls.cpu().numpy().astype(int)
                scores = r.obb.conf.cpu().numpy()

                for j in range(len(classes)):
                    mtc_id = self.dota_to_mtc.get(int(classes[j]))
                    if mtc_id is not None:
                        frame_data["detections"].append({
                            "track_id": -1,
                            "class_id": mtc_id,
                            "score": round(float(scores[j]), 6),
                            "obb_corners": corners[j].flatten().tolist(),
                        })

            batch_frames.append(frame_data)

        return batch_frames

    def _save_and_upload(
        self, data: Any, local_path: Path, drive_folder_id: str | None
    ) -> dict[str, Any]:
        """Guarda datos en JSON local y opcionalmente sube a Google Drive.

        Args:
            data: Diccionario a serializar.
            local_path: Ruta de destino local.
            drive_folder_id: ID de carpeta en Drive (o None para omitir subida).

        Returns:
            Registro con ruta local e ID de Drive.
        """
        self.io_manager.save_json(data, local_path)
        drive_id = None
        if drive_folder_id:
            drive_id = self.io_manager.upload_file_to_drive(local_path, drive_folder_id)
            if drive_id:
                print(f"    ✅ Subido a Drive: {local_path.name} (id: {drive_id})")
            else:
                print(f"    ⚠️  Fallo subida a Drive: {local_path.name}")
        return {"local": str(local_path), "drive_id": drive_id}

    def execute(self) -> dict[str, Any]:
        """Ejecuta el pipeline completo de inferencia por lotes.

        Incluye protección ante interrupciones: si la sesión se corta,
        el bloque finally guarda las métricas parciales acumuladas hasta
        ese momento. Además, si un clip ya tiene su JSON de salida, se
        omite para permitir reanudación sin reprocesar.
        """
        self.start_hardware_monitoring()

        metadata_path = Path(self.config["metadata_path"])
        images_dir = Path(self.config["images_dir"])
        output_dir = Path(self.config["output_dir"])
        drive_folder_id: str | None = self.config.get("drive_folder_id")
        imgsz: int = self.config.get("imgsz", 640)

        # 1. Obtener clips de validación
        csv_data = self.io_manager.load_csv(metadata_path)
        seen: set[str] = set()
        clip_ids: list[str] = []
        for row in csv_data:
            cid = row.get("clip_id", "")
            if row.get("split") == "val" and cid and cid not in seen:
                seen.add(cid)
                clip_ids.append(cid)

        # === RESTRICCIÓN DE PRUEBA: LIMITAR A 5 CLIPS ===
        clip_ids = clip_ids[:5]
        print(f"[Base0Runner] Clips a procesar (limitado a 5): {clip_ids}")

        generated_files: list[dict[str, Any]] = []

        # 2. Procesar secuencialmente cada clip con protección try/finally
        try:
            for clip_num, clip_id in enumerate(clip_ids, start=1):
                clip_dir = images_dir / clip_id

                # --- Reanudación: saltar si el JSON de este clip ya existe ---
                clip_json_path = output_dir / f"{clip_id}_predictions.json"
                if clip_json_path.exists():
                    print(f"  [{clip_num}/{len(clip_ids)}] {clip_id}: ya procesado, omitiendo.")
                    generated_files.append({"local": str(clip_json_path), "drive_id": "already_uploaded"})
                    continue

                if clip_dir.is_dir():
                    frame_paths = self.io_manager.list_files_in_dir(clip_dir, extension=".jpg")
                else:
                    frame_paths = self.io_manager.list_files_in_dir(images_dir, pattern=f"{clip_id}*.jpg")

                if not frame_paths:
                    print(f"  [{clip_num}/{len(clip_ids)}] {clip_id}: sin frames, omitiendo.")
                    continue

                print(f"  [{clip_num}/{len(clip_ids)}] {clip_id}: {len(frame_paths)} frames...")

                clip_results: dict[str, Any] = {
                    "clip_id": clip_id,
                    "inference_shape": [imgsz, imgsz],
                    "frames": [],
                }

                # Procesar en lotes para aprovechar la GPU
                for batch_start in range(0, len(frame_paths), self.batch_size):
                    batch_paths = frame_paths[batch_start : batch_start + self.batch_size]
                    # frame_paths ya son Paths absolutos (de list_files_in_dir con glob)
                    results = self.model.predict(
                        source=[str(p) for p in batch_paths],
                        conf=self.config.get("conf", 0.001),
                        iou=self.config.get("iou", 0.45),
                        imgsz=imgsz,
                        device=self.device,
                        verbose=False,
                    )
                    batch_frames = self._process_batch_results(results, frame_offset=batch_start)
                    clip_results["frames"].extend(batch_frames)

                # Guardar y subir a Drive
                file_record = self._save_and_upload(clip_results, clip_json_path, drive_folder_id)
                generated_files.append(file_record)

                # Liberar memoria forzosamente por cada clip para evitar OOM
                del clip_results
                gc.collect()

        except (KeyboardInterrupt, Exception) as e:
            print(f"\n[Base0Runner] ⚠️  Interrupción detectada: {type(e).__name__}: {e}")
            print("[Base0Runner] Guardando métricas parciales antes de salir...")

        finally:
            # 3. SIEMPRE guardar métricas (completas o parciales)
            metrics = self.record_hardware_metrics()
            metrics_path = output_dir / "inference_metrics.json"
            metrics_record = self._save_and_upload(metrics, metrics_path, drive_folder_id)
            generated_files.append(metrics_record)

            print(f"[Base0Runner] Finalizado. {self.total_frames_processed} frames procesados.")
            sys.stdout.flush()
            sys.stderr.flush()

        return {
            "status": "success",
            "files": generated_files,
            "metrics": metrics,
        }
