"""Tests for src/gdrive.py against a fake Drive v3 `service` double -- no
network, no real Google libraries required to exercise the request-shaping
logic (list query construction, upload/move/create bodies)."""

import pytest

from src import gdrive


class _FakeRequest:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class _FakeFiles:
    """Records every call made through `.files().<method>(...)` so tests can
    assert on the request shape, and returns canned responses."""

    def __init__(self, list_pages=None, get_result=None, create_result=None):
        self.calls = []
        self._list_pages = list(list_pages or [{"files": []}])
        self._get_result = get_result or {"id": "file1", "parents": ["old_parent"]}
        self._create_result = create_result or {"id": "new_file_id"}

    def list(self, **kwargs):
        self.calls.append(("list", kwargs))
        page = self._list_pages.pop(0) if self._list_pages else {"files": []}
        return _FakeRequest(page)

    def get_media(self, **kwargs):
        self.calls.append(("get_media", kwargs))
        return _FakeRequest(None)

    def get(self, **kwargs):
        self.calls.append(("get", kwargs))
        return _FakeRequest(self._get_result)

    def create(self, **kwargs):
        self.calls.append(("create", kwargs))
        return _FakeRequest(self._create_result)

    def update(self, **kwargs):
        self.calls.append(("update", kwargs))
        return _FakeRequest({"id": kwargs.get("fileId"), "parents": [kwargs.get("addParents")]})

    def delete(self, **kwargs):
        self.calls.append(("delete", kwargs))
        return _FakeRequest(None)


class _FakeService:
    def __init__(self, **files_kwargs):
        self._files = _FakeFiles(**files_kwargs)

    def files(self):
        return self._files


class _FakeDownloader:
    """Stands in for googleapiclient.http.MediaIoBaseDownload -- writes a
    fixed payload in a single (fh, request) chunk."""
    instances = []

    def __init__(self, fh, request):
        self.fh = fh
        self.request = request
        _FakeDownloader.instances.append(self)

    def next_chunk(self):
        self.fh.write(b"fake-checkpoint-bytes")
        return None, True


def _client(service=None, **kw):
    return gdrive.DriveClient(service=service or _FakeService(), **kw)


# --------------------------------------------------------------------- list

def test_list_scopes_query_to_folder_and_filters_by_pattern():
    service = _FakeService(list_pages=[{"files": [
        {"id": "a", "name": "arch4r_fullrun_seed1_encoder_only_epoch5.pt"},
        {"id": "b", "name": "unrelated.json"},
    ]}])
    client = _client(service=service, folder_id="root123")
    hits = client.list(pattern=r"\.pt$")
    assert hits == [("a", "arch4r_fullrun_seed1_encoder_only_epoch5.pt")]
    method, kwargs = service.files().calls[0]
    assert method == "list"
    assert "'root123' in parents" in kwargs["q"]


def test_list_paginates_until_no_next_page_token():
    service = _FakeService(list_pages=[
        {"files": [{"id": "a", "name": "one.pt"}], "nextPageToken": "tok2"},
        {"files": [{"id": "b", "name": "two.pt"}]},
    ])
    client = _client(service=service, folder_id="root123")
    hits = client.list()
    assert {h[0] for h in hits} == {"a", "b"}


# ------------------------------------------------------------------- upload

def test_upload_sets_filename_and_parent(tmp_path):
    local = tmp_path / "ckpt.pt"
    local.write_bytes(b"data")
    service = _FakeService(create_result={"id": "uploaded123"})
    client = _client(service=service, folder_id="root123")

    file_id = client.upload(local, filename="ckpt.pt")

    assert file_id == "uploaded123"
    method, kwargs = service.files().calls[-1]
    assert method == "create"
    assert kwargs["body"]["name"] == "ckpt.pt"
    assert kwargs["body"]["parents"] == ["root123"]


# ----------------------------------------------------------------- download

def test_download_writes_local_file(tmp_path):
    local = tmp_path / "sub" / "ckpt.pt"
    client = _client()
    client.download("file1", local, downloader_cls=_FakeDownloader)
    assert local.read_bytes() == b"fake-checkpoint-bytes"


# ------------------------------------------------------------------- delete

def test_delete_calls_api_with_file_id():
    service = _FakeService()
    client = _client(service=service)
    client.delete("file1")
    method, kwargs = service.files().calls[-1]
    assert method == "delete"
    assert kwargs["fileId"] == "file1"


def test_delete_is_noop_when_file_id_is_none():
    service = _FakeService()
    client = _client(service=service)
    client.delete(None)
    assert service.files().calls == []


# ---------------------------------------------------------- get_or_create

def test_get_or_create_folder_reuses_existing():
    service = _FakeService(list_pages=[{"files": [{"id": "existing", "name": "2026-01-01"}]}])
    client = _client(service=service, folder_id="root123")
    folder_id = client.get_or_create_folder("2026-01-01")
    assert folder_id == "existing"
    assert all(m != "create" for m, _ in service.files().calls)


def test_get_or_create_folder_creates_when_missing():
    service = _FakeService(list_pages=[{"files": []}], create_result={"id": "brand_new"})
    client = _client(service=service, folder_id="root123")
    folder_id = client.get_or_create_folder("2026-01-01")
    assert folder_id == "brand_new"
    method, kwargs = service.files().calls[-1]
    assert method == "create"
    assert kwargs["body"]["mimeType"] == "application/vnd.google-apps.folder"


# ------------------------------------------------------------------- move

def test_move_removes_old_parent_and_adds_new_one():
    service = _FakeService(get_result={"id": "file1", "parents": ["old_parent"]})
    client = _client(service=service)
    client.move("file1", "new_parent")
    method, kwargs = service.files().calls[-1]
    assert method == "update"
    assert kwargs["addParents"] == "new_parent"
    assert kwargs["removeParents"] == "old_parent"
