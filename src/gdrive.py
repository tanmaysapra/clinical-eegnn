"""Google Drive checkpoint I/O for the full-epoch-budget Arch4R notebook
(`notebooks/src/11_arch4r_fullrun.py`).

Authenticates as a real Google *user* account via an OAuth refresh token
(client id/secret/refresh token supplied as Kaggle Secrets), not a service
account -- a bare service account has zero storage quota of its own on a
personal (non-Workspace) Gmail account, so direct uploads to a shared folder
fail with "Service Accounts do not have storage quota" the first time a real
checkpoint is written. Uploading as the user's own account avoids that
entirely (see `notebooks/src/11_arch4r_fullrun.py`'s setup cell for the
one-time local script that mints the refresh token).

The google-api-python-client / google-auth imports are lazy (only pulled in
when `DriveClient` actually needs to build a live service) so importing this
module never requires those packages to be installed -- useful for anything
that only wants `DriveClient` for type-checking or testing against a fake
`service` double (see `tests/test_gdrive.py`).
"""

import os
import re


class DriveClient:
    def __init__(self, client_id=None, client_secret=None, refresh_token=None,
                folder_id=None, service=None):
        """`folder_id` is the default parent for list/upload/create -- the
        dedicated Drive folder you shared checkpoint access to, not Drive's
        actual root (a plain "My Drive" root works fine for a real user
        account, but scoping to one folder keeps checkpoints out of the way
        of everything else in the account). Pass `service` directly (an
        already-built googleapiclient Drive v3 resource, or a test double)
        to skip OAuth entirely."""
        self.folder_id = folder_id
        self._service = service if service is not None else self._build_service(
            client_id, client_secret, refresh_token)

    @staticmethod
    def _build_service(client_id, client_secret, refresh_token):
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        creds = Credentials(
            token=None,
            refresh_token=refresh_token,
            client_id=client_id,
            client_secret=client_secret,
            token_uri="https://oauth2.googleapis.com/token",
        )
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    # ------------------------------------------------------------- queries

    def list(self, pattern=None, folder_id=None):
        """Return `[(file_id, name), ...]` under `folder_id` (default
        `self.folder_id`), optionally filtered by a regex `pattern` matched
        against the file name."""
        parent = folder_id if folder_id is not None else self.folder_id
        q_parts = ["trashed = false"]
        if parent:
            q_parts.append(f"'{parent}' in parents")
        q = " and ".join(q_parts)

        files, page_token = [], None
        while True:
            resp = self._service.files().list(
                q=q, spaces="drive", fields="nextPageToken, files(id, name)",
                pageToken=page_token).execute()
            files.extend(resp.get("files", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        rx = re.compile(pattern) if pattern else None
        return [(f["id"], f["name"]) for f in files if rx is None or rx.search(f["name"])]

    # ------------------------------------------------------------- content

    def download(self, file_id, local_path, downloader_cls=None):
        if downloader_cls is None:
            from googleapiclient.http import MediaIoBaseDownload
            downloader_cls = MediaIoBaseDownload

        local_path = str(local_path)
        parent_dir = os.path.dirname(local_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        request = self._service.files().get_media(fileId=file_id)
        with open(local_path, "wb") as fh:
            downloader = downloader_cls(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        return local_path

    def upload(self, local_path, filename=None, folder_id=None):
        from googleapiclient.http import MediaFileUpload

        parent = folder_id if folder_id is not None else self.folder_id
        metadata = {"name": filename or os.path.basename(str(local_path))}
        if parent:
            metadata["parents"] = [parent]
        media = MediaFileUpload(str(local_path), resumable=True)
        f = self._service.files().create(
            body=metadata, media_body=media, fields="id").execute()
        return f["id"]

    def delete(self, file_id):
        if file_id is None:
            return
        self._service.files().delete(fileId=file_id).execute()

    # -------------------------------------------------------------- folders

    def get_or_create_folder(self, name, parent_id=None):
        parent = parent_id if parent_id is not None else self.folder_id
        q_parts = [f"name = '{name}'",
                  "mimeType = 'application/vnd.google-apps.folder'", "trashed = false"]
        if parent:
            q_parts.append(f"'{parent}' in parents")
        resp = self._service.files().list(
            q=" and ".join(q_parts), spaces="drive", fields="files(id, name)").execute()
        hits = resp.get("files", [])
        if hits:
            return hits[0]["id"]

        metadata = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
        if parent:
            metadata["parents"] = [parent]
        f = self._service.files().create(body=metadata, fields="id").execute()
        return f["id"]

    def move(self, file_id, new_parent_id):
        f = self._service.files().get(fileId=file_id, fields="parents").execute()
        prev_parents = ",".join(f.get("parents", []))
        return self._service.files().update(
            fileId=file_id, addParents=new_parent_id,
            removeParents=prev_parents, fields="id, parents").execute()
