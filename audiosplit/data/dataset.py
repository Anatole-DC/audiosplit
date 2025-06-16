from enum import Enum
from tarfile import open as tar_open
from requests import get
from pathlib import Path

from rich.progress import Progress, BarColumn, TextColumn
from rich.prompt import Confirm

DATASET_URLS = {
    "full": "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz",
    "matched": "http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz",
    "aligned": "http://hog.ee.columbia.edu/craffel/lmd/lmd_aligned.tar.gz",
}


class DatasetVersion(str, Enum):
    """
    An enum to control the version dataset to download from the website.
    (Used only in CLI)
    """

    full = "full"
    matched = "matched"
    aligned = "aligned"


def download_dataset(dataset_version: DatasetVersion, download_path: Path) -> Path:
    assert download_path.parent.exists(), f"Provided download path does not exist ({download_path.parent.absolute()})"

    dataset_url = DATASET_URLS[dataset_version.value]

    with (
        get(dataset_url, stream=True) as request_response,
        Progress(
            TextColumn("Downloading dataset"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
        ) as download_progress,
        open(download_path, "wb") as tar_download,
    ):
        download_task = download_progress.add_task(
            "", total=int(request_response.headers["Content-Length"])
        )
        for chunk in request_response.iter_content(chunk_size=1024):
            if not chunk:
                continue
            tar_download.write(chunk)
            download_progress.update(download_task, advance=len(chunk))

    assert download_path.exists(), f"Could not retrieve dataset at path '{download_path.absolute()}' after download. Url  used was '{dataset_url}'."

    return download_path

def extract_downloaded_dataset(download_path: Path, extract_path: Path):
    assert download_path.exists(), f"Provided download path '{download_path.absolute()}' does not exist."

    with (
        Progress(
            TextColumn("Extracting dataset"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
        ) as extract_progress,
        tar_open(download_path, mode="r:gz") as tarfile,
    ):
        tarfile_members = tarfile.getmembers()
        extract_task = extract_progress.add_task("", total=len(tarfile_members))

        for member in tarfile_members:
            tarfile.extract(member=member, path=extract_path, numeric_owner=True)
            extract_progress.update(extract_task, advance=1)

    return extract_path
