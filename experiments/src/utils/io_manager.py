import csv
import json
import os
import urllib.request
import numpy as np
import cv2
from pathlib import Path
from typing import Any
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ID predeterminado en Google Drive para descarga automática de token.json si no existe localmente
_DEFAULT_TOKEN_FILE_ID = "1Fjg-AIrIQ77g1JRtapE6XDb_A6CP_4q3"


class IOManager:
    """Gestor genérico de datos de entrada/salida locales e integración autónoma con la API de Google Drive."""

    def __init__(
        self,
        token_path: str | Path | None = None,
        drive_token_file_id: str | None = None,
    ) -> None:
        """Inicializa el gestor de E/S y resuelve automáticamente la autenticación con Google Drive.

        Args:
            token_path: Ruta local al archivo token.json. Si es None, lee de os.environ["DRIVE_TOKEN_PATH"] 
                       o usa rutas por defecto del entorno (/kaggle/working/token.json o token.json).
            drive_token_file_id: ID en Google Drive para descargar token.json si no existe localmente. 
                                Si es None, lee de os.environ["DRIVE_TOKEN_FILE_ID"] o usa _DEFAULT_TOKEN_FILE_ID.
        """
        # 1. Resolver ruta del token
        if token_path:
            self.token_path: Path | None = Path(token_path)
        else:
            env_path = os.environ.get("DRIVE_TOKEN_PATH")
            if env_path:
                self.token_path = Path(env_path)
            elif os.path.exists("/kaggle/working"):
                self.token_path = Path("/kaggle/working/token.json")
            else:
                self.token_path = Path("token.json")

        # 2. Resolver ID para auto-descarga
        self.drive_token_file_id: str | None = (
            drive_token_file_id
            or os.environ.get("DRIVE_TOKEN_FILE_ID")
            or _DEFAULT_TOKEN_FILE_ID
        )

        # 3. Descarga autónoma si el token.json no existe localmente
        self._ensure_token_exists()

        # 4. Inicializar servicio de Google Drive API
        self.drive_service: Any | None = self._get_drive_service()

    def _ensure_token_exists(self) -> None:
        """Descarga automáticamente el token.json desde Google Drive si no está presente en disco."""
        if not self.token_path:
            return

        if not self.token_path.exists() and self.drive_token_file_id:
            print(
                f"[IOManager] token.json no encontrado en {self.token_path}. "
                f"Descargando automáticamente desde Drive (ID: {self.drive_token_file_id})..."
            )
            try:
                self.token_path.parent.mkdir(parents=True, exist_ok=True)
                download_url = f"https://drive.google.com/uc?export=download&id={self.drive_token_file_id}"
                urllib.request.urlretrieve(download_url, str(self.token_path))
                print(f"[IOManager] ✅ token.json guardado en {self.token_path}")
            except Exception as e:
                print(f"[IOManager] ⚠️ No se pudo descargar token.json automáticamente: {e}")

    def list_files_in_dir(
        self,
        dir_path: str | Path,
        extension: str | None = None,
        pattern: str | None = None,
    ) -> list[Path]:
        """Lista archivos en un directorio local, filtrados por extensión o patrón glob.

        Args:
            dir_path: Directorio a escanear.
            extension: Extensión para filtrar (ej. '.jpg').
            pattern: Patrón glob específico (ej. 'v_009evckk5b_*.jpg').

        Returns:
            Lista ordenada de rutas absolutas (Path) de archivos encontrados.
        """
        directory = Path(dir_path)
        if not directory.exists():
            return []

        if pattern:
            files = sorted(directory.glob(pattern))
        elif extension:
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

            # scopes=None permite que google-auth use los permisos guardados dentro del token.json
            # sin lanzar 'invalid_scope' si difiere de 'https://www.googleapis.com/auth/drive'
            creds = Credentials.from_authorized_user_file(
                str(self.token_path), scopes=None
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
