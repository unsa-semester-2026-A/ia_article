"""Google Drive API helper utilities for file sharing and persistence.

This module provides common functions to interact with Google Drive, including
credential loading, file uploads, folder resolution, and creation.
"""

import io
import os
import sys
from typing import Optional

# Type annotation for the API service (googleapiclient.discovery.Resource)
from googleapiclient.discovery import Resource


def get_drive_service(token_path: str) -> Optional[Resource]:
    """Helper to build Drive API service using team authorized credentials JSON.

    Args:
        token_path: Path to the authorized user credentials JSON.

    Returns:
        Resource: Google Drive API service client or None if initialization fails.
    """
    if not os.path.exists(token_path):
        print(f"Credentials token not found at: {token_path}", file=sys.stderr)
        return None
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        creds = Credentials.from_authorized_user_file(
            token_path, ["https://www.googleapis.com/auth/drive.file"]
        )
        return build("drive", "v3", credentials=creds)
    except Exception as e:
        print(f"Error initializing Google Drive API: {e}", file=sys.stderr)
        return None


def get_project_root_folder_id(
    service: Resource, known_folder_id: str
) -> Optional[str]:
    """Resolves the parent folder ID of a known folder ID.

    Used to dynamically locate the root directory of the shared project.

    Args:
        service: Google Drive API service instance.
        known_folder_id: ID of a folder known to exist (e.g. 02_pseudo_labeling).

    Returns:
        str: ID of the parent folder or None if not resolved.
    """
    try:
        folder = (
            service.files()
            .get(fileId=known_folder_id, fields="parents", supportsAllDrives=True)
            .execute()
        )
        parents = folder.get("parents", [])
        if parents:
            return str(parents[0])
    except Exception as e:
        print(f"Error resolving project root folder ID: {e}", file=sys.stderr)
    return None


def find_or_create_folder(
    service: Resource, folder_name: str, parent_folder_id: str
) -> Optional[str]:
    """Finds a subfolder by name under parent_folder_id or creates it if not found.

    Args:
        service: Google Drive API service instance.
        folder_name: Target folder name.
        parent_folder_id: ID of the parent folder.

    Returns:
        str: ID of the resolved or created folder, or None if failed.
    """
    try:
        query = (
            f"'{parent_folder_id}' in parents and "
            f"name = '{folder_name}' and "
            f"mimeType = 'application/vnd.google-apps.folder' and "
            f"trashed = false"
        )
        results = (
            service.files()
            .list(
                q=query,
                fields="files(id, name)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        files = results.get("files", [])
        if files:
            return str(files[0]["id"])

        # Create the folder if it does not exist
        metadata = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_folder_id],
        }
        folder = (
            service.files()
            .create(body=metadata, fields="id", supportsAllDrives=True)
            .execute()
        )
        return str(folder.get("id"))
    except Exception as e:
        print(f"Error finding/creating folder '{folder_name}': {e}", file=sys.stderr)
        return None


def upload_file_to_drive(
    service: Resource,
    local_path: str,
    parent_folder_id: str,
    mime_type: str = "application/octet-stream",
) -> Optional[str]:
    """Uploads a local file to a specific Drive folder ID using resumable upload API.

    Args:
        service: Google Drive API service instance.
        local_path: Path to the local file to upload.
        parent_folder_id: ID of the parent folder on Google Drive.
        mime_type: Mimetype of the file.

    Returns:
        str: ID of the uploaded file on Google Drive, or None if failed.
    """
    try:
        from googleapiclient.http import MediaIoBaseUpload

        filename = os.path.basename(local_path)
        with open(local_path, "rb") as f:
            file_data = f.read()

        flujo_archivo = io.BytesIO(file_data)
        metadata = {
            "name": filename,
            "mimeType": mime_type,
            "parents": [parent_folder_id],
        }
        media = MediaIoBaseUpload(flujo_archivo, mimetype=mime_type, resumable=True)
        archivo = (
            service.files()
            .create(
                body=metadata,
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            )
            .execute()
        )
        return str(archivo.get("id"))
    except Exception as e:
        print(
            f"Error uploading file {local_path} to Google Drive: {e}",
            file=sys.stderr,
        )
        return None
