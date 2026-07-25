"""Generic Input/Output data manager and autonomous Google Drive API integration module."""

import csv
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


class IOManager:
    """Generic I/O data manager and autonomous Google Drive API integration."""

    def __init__(
        self,
        token_path: str | Path | None = None,
        require_drive: bool = True,
    ) -> None:
        """Initialize I/O manager and resolve Google Drive authentication.

        Authentication strategy:
            1. If token.json exists locally, load credentials from it.
            2. Otherwise, attempt to retrieve the JSON token from Kaggle Secrets
               (secret key: ``DRIVE_TOKEN_JSON``) and persist it locally.
            3. Refresh expired OAuth tokens automatically and save the updated token.

        Args:
            token_path: Local path to token.json file. If None, reads from
                ``os.environ["DRIVE_TOKEN_PATH"]`` or falls back to default
                environment paths (/kaggle/working/token.json or token.json).
            require_drive: Whether an unreachable Drive is fatal. True protects a
                long run from producing results it cannot persist. False lets
                callers whose output is disposable, or who report the failure
                themselves, degrade to local-only I/O.
        """
        self.require_drive = require_drive

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

        # 2. Automatically generate token.json from Kaggle Secrets if missing
        self._ensure_token_from_kaggle_secrets()

        # 3. Initialize Google Drive API service (with auto-refresh)
        self.drive_service: Any | None = self._get_drive_service()

    def _ensure_token_from_kaggle_secrets(self) -> None:
        """Retrieve token.json from Kaggle Secrets if not present on disk.

        Raises:
            RuntimeError: If the secret cannot be read and ``require_drive`` is set.
                A production run that cannot persist its results would waste hours
                of GPU quota, so it must not start. Callers whose artifacts are
                disposable pass ``require_drive=False`` and continue without Drive.
        """
        if not self.token_path or self.token_path.exists():
            return

        try:
            from kaggle_secrets import UserSecretsClient

            secrets = UserSecretsClient()
            token_data = secrets.get_secret("DRIVE_TOKEN_JSON")
            if not token_data:
                raise ValueError("Secret 'DRIVE_TOKEN_JSON' returned empty data.")

            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            with self.token_path.open("w", encoding="utf-8") as f:
                f.write(token_data)
            print(f"[IOManager] ✅ token.json created from Kaggle Secrets -> {self.token_path}")
        except Exception as e:
            err_msg = f"[IOManager] ❌ Failed to retrieve Kaggle secret 'DRIVE_TOKEN_JSON': {e}"
            print(err_msg)
            if self.require_drive:
                raise RuntimeError(err_msg) from e
            print(
                "[IOManager] ⚠️ Continuing without Google Drive: "
                "artifacts will only be written locally.",
                flush=True,
            )

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
        Automatically refreshes expired tokens and persists the new token to disk.
        Works agnostically across any environment (Kaggle, Colab, Local).
        """
        if not self.token_path or not self.token_path.exists():
            print(
                f"[IOManager] token.json not found at {self.token_path}. Google Drive disabled."
            )
            return None

        try:
            # scopes=None allows google-auth to use authorized scopes inside token.json
            # avoiding 'invalid_scope' errors if scopes differ from exact strings
            creds = Credentials.from_authorized_user_file(
                str(self.token_path), scopes=None
            )
            # Refresh token if expired and refresh_token is present
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                # Persist the refreshed token so subsequent runs don't re-auth
                with self.token_path.open("w", encoding="utf-8") as f:
                    f.write(creds.to_json())
                print("[IOManager] 🔄 OAuth token refreshed and saved.")

            service = build("drive", "v3", credentials=creds)
            return service
        except Exception as e:
            print(f"[IOManager] Credentials or OAuth error: {e}")
            return None

    def _find_existing_file_id(
        self, file_name: str, drive_folder_id: str
    ) -> str | None:
        """Search for an existing file by name in a Google Drive folder.

        Args:
            file_name: Name of the file to search for.
            drive_folder_id: Google Drive parent folder ID.

        Returns:
            File ID if found, or None.
        """
        try:
            query = (
                f"'{drive_folder_id}' in parents and name = '{file_name}' and "
                "trashed = false"
            )
            results = (
                self.drive_service.files()
                .list(
                    q=query,
                    fields="files(id, name)",
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            files = results.get("files", [])
            return files[0]["id"] if files else None
        except Exception:
            return None

    def upload_file_to_drive(
        self,
        local_path: str | Path,
        drive_folder_id: str,
        mime_type: str = "application/json",
        remote_name: str | None = None,
    ) -> str | None:
        """Upload a local file to a specific Google Drive folder.

        If a file with the same name already exists in the destination folder,
        it is **updated** (overwritten) using its file_id instead of creating
        a duplicate.  This prevents Drive storage saturation from repeated syncs.

        Args:
            local_path: Path to the local file.
            drive_folder_id: Target Google Drive folder ID.
            mime_type: MIME content type.
            remote_name: Name to store the file under. Defaults to the local file
                name. Callers that share a destination folder across runs must pass
                a run-specific name, since every framework writes generic names such
                as ``last.pt`` and one run would otherwise overwrite another.

        Returns:
            Google Drive file ID (created or updated), or None on failure.
        """
        if not self.drive_service:
            print("[IOManager] Drive service not initialized. Skipping upload.")
            return None

        path = Path(local_path)
        target_name = remote_name or path.name
        try:
            media = MediaFileUpload(str(path), mimetype=mime_type, resumable=True)
            existing_id = self._find_existing_file_id(target_name, drive_folder_id)

            if existing_id:
                # Update (overwrite) the existing file
                updated_file = (
                    self.drive_service.files()
                    .update(
                        fileId=existing_id,
                        media_body=media,
                        fields="id",
                        supportsAllDrives=True,
                    )
                    .execute()
                )
                return updated_file.get("id")
            else:
                # Create a new file
                file_metadata = {"name": target_name, "parents": [drive_folder_id]}
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

    def download_file_from_drive(
        self,
        file_name: str,
        drive_folder_id: str,
        local_destination_path: str | Path,
    ) -> Path | None:
        """Search for a file by name in a Google Drive folder and download it locally.

        A truncated checkpoint fails to load only after the session has already
        spent its setup minutes, so the downloaded size is compared against the
        size reported by Drive and a mismatch discards the file.

        Args:
            file_name: Name of the file to search for (e.g. 'last.pt').
            drive_folder_id: Google Drive parent folder ID.
            local_destination_path: Local destination path for the downloaded file.

        Returns:
            Path object of the downloaded file, or None if not found, incomplete,
            or failed.
        """
        if not self.drive_service:
            print("[IOManager] Drive service not initialized. Skipping download.")
            return None

        dest_path = Path(local_destination_path)
        try:
            from googleapiclient.http import MediaIoBaseDownload

            query = (
                f"'{drive_folder_id}' in parents and name = '{file_name}' and "
                "trashed = false"
            )
            results = (
                self.drive_service.files()
                .list(
                    q=query,
                    fields="files(id, name, size)",
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            files = results.get("files", [])
            if not files:
                print(
                    f"[IOManager] File '{file_name}' not found in Drive folder {drive_folder_id}."
                )
                return None

            file_id = files[0]["id"]
            expected_size = int(files[0].get("size") or 0)
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            request = self.drive_service.files().get_media(
                fileId=file_id, supportsAllDrives=True
            )
            with dest_path.open("wb") as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()

            actual_size = dest_path.stat().st_size
            if expected_size and actual_size != expected_size:
                print(
                    f"[IOManager] ❌ {file_name} is incomplete "
                    f"({actual_size} of {expected_size} bytes). Discarding.",
                    flush=True,
                )
                dest_path.unlink(missing_ok=True)
                return None

            print(
                f"[IOManager] ✅ Downloaded {file_name} from Drive -> {dest_path} "
                f"({actual_size} bytes)",
                flush=True,
            )
            return dest_path
        except Exception as e:
            print(
                f"[IOManager] ⚠️ Error downloading {file_name} from Drive: {e}",
                flush=True,
            )
            return None
