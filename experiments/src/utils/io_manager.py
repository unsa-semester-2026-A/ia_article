"""Generic Input/Output data manager and autonomous Google Drive API integration module."""

import csv
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Default file ID on Google Drive for automatic token.json download if missing locally
_DEFAULT_TOKEN_FILE_ID = "1Fjg-AIrIQ77g1JRtapE6XDb_A6CP_4q3"


class IOManager:
    """Generic I/O data manager and autonomous Google Drive API integration."""

    def __init__(
        self,
        token_path: str | Path | None = None,
        drive_token_file_id: str | None = None,
    ) -> None:
        """Initialize I/O manager and resolve Google Drive authentication.

        Args:
            token_path: Local path to token.json file. If None, reads from os.environ["DRIVE_TOKEN_PATH"]
                       or falls back to default environment paths (/kaggle/working/token.json or token.json).
            drive_token_file_id: Google Drive file ID to download token.json if missing locally.
                                If None, reads from os.environ["DRIVE_TOKEN_FILE_ID"] or uses _DEFAULT_TOKEN_FILE_ID.
        """
        # 1. Resolve token path
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

        # 2. Resolve file ID for auto-download
        self.drive_token_file_id: str | None = (
            drive_token_file_id
            or os.environ.get("DRIVE_TOKEN_FILE_ID")
            or _DEFAULT_TOKEN_FILE_ID
        )

        # 3. Autonomous download if token.json is missing locally
        self._ensure_token_exists()

        # 4. Initialize Google Drive API service
        self.drive_service: Any | None = self._get_drive_service()

    def _ensure_token_exists(self) -> None:
        """Download token.json from Google Drive if not present on disk."""
        if not self.token_path:
            return

        if not self.token_path.exists() and self.drive_token_file_id:
            print(
                f"[IOManager] token.json not found at {self.token_path}. "
                f"Downloading automatically from Drive (ID: {self.drive_token_file_id})..."
            )
            try:
                self.token_path.parent.mkdir(parents=True, exist_ok=True)
                download_url = f"https://drive.google.com/uc?export=download&id={self.drive_token_file_id}"
                urllib.request.urlretrieve(download_url, str(self.token_path))
                print(f"[IOManager] ✅ token.json saved to {self.token_path}")
            except Exception as e:
                print(f"[IOManager] ⚠️ Could not download token.json automatically: {e}")

    def list_files_in_dir(
        self,
        dir_path: str | Path,
        extension: str | None = None,
        pattern: str | None = None,
    ) -> list[Path]:
        """List files in a local directory, filtered by extension or glob pattern.

        Args:
            dir_path: Directory to scan.
            extension: Extension filter (e.g. '.jpg').
            pattern: Glob pattern filter (e.g. 'v_009evckk5b_*.jpg').

        Returns:
            Sorted list of absolute Path objects for matching files.
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
        """Load a generic CSV file and return rows as dictionaries.

        Args:
            file_path: Path to the CSV file.

        Returns:
            List of dictionaries representing each row.
        """
        path = Path(file_path)
        if not path.exists():
            return []

        with path.open(mode="r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def load_image(self, image_path: str | Path) -> np.ndarray:
        """Load an image from disk using OpenCV.

        Args:
            image_path: Absolute path to the image file.

        Returns:
            Numpy array of the loaded BGR image.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        return img

    def save_json(self, data: Any, local_path: str | Path) -> Path:
        """Save a dictionary or list as a JSON file at the specified local path.

        Args:
            data: Content to serialize.
            local_path: Destination path on disk.

        Returns:
            Absolute Path object of the persisted JSON file.
        """
        path = Path(local_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def _get_drive_service(self) -> Any | None:
        """Initialize and return OAuth2 Google Drive API service.

        Uses credentials from self.token_path if available.
        Works agnostically across any environment (Kaggle, Colab, Local).
        """
        if not self.token_path or not self.token_path.exists():
            print(
                f"[IOManager] token.json not found at {self.token_path}. Google Drive disabled."
            )
            return None

        try:
            from google.auth.transport.requests import Request

            # scopes=None allows google-auth to use authorized scopes inside token.json
            # avoiding 'invalid_scope' errors if scopes differ from exact strings
            creds = Credentials.from_authorized_user_file(
                str(self.token_path), scopes=None
            )
            # Refresh token if expired and refresh_token is present
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())

            service = build("drive", "v3", credentials=creds)
            return service
        except Exception as e:
            print(f"[IOManager] Error initializing Drive Service: {e}")
            return None

    def upload_file_to_drive(
        self,
        local_path: str | Path,
        drive_folder_id: str,
        mime_type: str = "application/json",
    ) -> str | None:
        """Upload a local file to a specific Google Drive folder.

        Args:
            local_path: Path to the local file.
            drive_folder_id: Target Google Drive folder ID.
            mime_type: MIME content type.

        Returns:
            Uploaded Google Drive file ID, or None on failure.
        """
        if not self.drive_service:
            print("[IOManager] Drive service not initialized. Skipping upload.")
            return None

        path = Path(local_path)
        try:
            file_metadata = {"name": path.name, "parents": [drive_folder_id]}
            media = MediaFileUpload(str(path), mimetype=mime_type, resumable=True)

            uploaded_file = (
                self.drive_service.files()
                .create(
                    body=file_metadata,
                    media_body=media,
                    fields="id",
                    supportsAllDrives=True,
                )
                .execute()
            )

            return uploaded_file.get("id")
        except Exception as e:
            print(f"[IOManager] Failed to upload {path} to Drive: {e}")
            return None
