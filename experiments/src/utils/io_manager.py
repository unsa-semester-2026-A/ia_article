import csv
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Any
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


class IOManager:
    """Gestor genérico de datos de entrada/salida locales e integración con la API de Google Drive."""

    def __init__(self, token_path: str | None = None) -> None:
        """Inicializa el gestor de E/S.

        Args:
            token_path: Ruta al archivo token.json de Google Drive para inicialización del servicio.
        """
        self.token_path: Path | None = Path(token_path) if token_path else None
        self.drive_service: Any | None = self._get_drive_service() if token_path else None

    def list_files_in_dir(self, dir_path: str | Path, extension: str | None = None) -> list[Path]:
        """Lista archivos en un directorio local, opcionalmente filtrados por extensión.

        Args:
            dir_path: Directorio a escanear.
            extension: Extensión para filtrar (ej. '.jpg').

        Returns:
            Lista ordenada de rutas absolutas (Path) de archivos encontrados.
        """
        directory = Path(dir_path)
        if not directory.exists():
            return []

        if extension:
            files = sorted(directory.glob(f"*{extension}"))
        else:
            files = sorted(f for f in directory.iterdir() if f.is_file())
        return files

    def load_csv(self, file_path: str | Path) -> list[dict]:
        """Carga un archivo CSV genérico y retorna sus filas como diccionarios.

        Args:
            file_path: Ruta al archivo CSV.

        Returns:
            Lista de diccionarios representando cada fila.
        """
        path = Path(file_path)
        if not path.exists():
            return []

        with path.open(mode="r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def load_image(self, image_path: str | Path) -> np.ndarray:
        """Carga una imagen desde el disco usando OpenCV.

        Args:
            image_path: Ruta absoluta al archivo de imagen.

        Returns:
            Matriz numpy de la imagen cargada (BGR).
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
        return img

    def save_json(self, data: Any, local_path: str | Path) -> Path:
        """Guarda un diccionario o lista en formato JSON en la ruta local indicada.

        Args:
            data: Contenido a serializar.
            local_path: Ruta destino en disco.

        Returns:
            Ruta absoluta (Path) del archivo JSON persistido.
        """
        path = Path(local_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def _get_drive_service(self) -> Any | None:
        """Inicializa y retorna la conexión OAuth2 con Google Drive API.

        Usa las credenciales indicadas en self.token_path si están disponibles.
        Funciona de forma agnóstica en cualquier entorno (Kaggle, Colab, Local).
        """
        if not self.token_path or not self.token_path.exists():
            print(f"[IOManager] No se encontró token.json en {self.token_path}. Google Drive deshabilitado.")
            return None

        try:
            from google.auth.transport.requests import Request

            creds = Credentials.from_authorized_user_file(
                str(self.token_path), ["https://www.googleapis.com/auth/drive"]
            )
            # Refrescar token si ha expirado y se cuenta con refresh_token
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())

            service = build("drive", "v3", credentials=creds)
            return service
        except Exception as e:
            print(f"[IOManager] Error inicializando Drive Service: {e}")
            return None

    def upload_file_to_drive(
        self, local_path: str | Path, drive_folder_id: str, mime_type: str = "application/json"
    ) -> str | None:
        """Sube un archivo local a una carpeta específica en Google Drive. No contiene IDs por defecto.

        Args:
            local_path: Ruta al archivo en disco.
            drive_folder_id: ID de la carpeta destino de Google Drive (Debe ser inyectado, NUNCA por defecto).
            mime_type: Tipo de contenido MIME.

        Returns:
            El ID del archivo subido en Google Drive, o None si ocurre un fallo.
        """
        if not self.drive_service:
            print("[IOManager] Servicio Drive no inicializado. Omitiendo subida.")
            return None

        path = Path(local_path)
        try:
            file_metadata = {"name": path.name, "parents": [drive_folder_id]}
            media = MediaFileUpload(str(path), mimetype=mime_type, resumable=True)

            uploaded_file = (
                self.drive_service.files()
                .create(body=file_metadata, media_body=media, fields="id", supportsAllDrives=True)
                .execute()
            )

            return uploaded_file.get("id")
        except Exception as e:
            print(f"[IOManager] Fallo al subir {path} a Drive: {e}")
            return None
