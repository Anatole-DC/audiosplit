from typing_extensions import Annotated, List, Optional
from typer import Typer, Argument, Option
from pathlib import Path
from tarfile import open as tar_open
from requests import get
from enum import Enum

from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.prompt import Confirm

from audiosplit.config.environment import (
    RAW_DATA_DIRECTORY,
    PREPROCESSED_DATA_DIRECTORY,
)
from audiosplit.data.preprocessing import convert_all_midi_files

data_cli_app = Typer()


DATASET_URLS = {
    "full": "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz",
    "matched": "http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz",
    "aligned": "http://hog.ee.columbia.edu/craffel/lmd/lmd_aligned.tar.gz",
}


class DatasetVersion(str, Enum):
    full = "full"
    matched = "matched"
    aligned = "aligned"


@data_cli_app.command("download")
def download_data(
    dataset: Annotated[
        DatasetVersion, Option(help="The version of dataset do you want to download.")
    ] = DatasetVersion.matched,
):
    dataset_url = DATASET_URLS[dataset.value]
    print(dataset_url)
    temp_tar_path = RAW_DATA_DIRECTORY.parent / f"{dataset.value}.tar.gz"

    if temp_tar_path.exists() and not Confirm.ask(f"Archive of the {dataset.value} dataset found. Do you want to use it ?"):
        with (
            get(dataset_url, stream=True) as request_response,
            Progress(
                TextColumn("Downloading dataset"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}")
            ) as download_progress,
            open(temp_tar_path, "wb") as tar_download
        ):
            download_task = download_progress.add_task(
                "", total=int(request_response.headers["Content-Length"])
            )
            for chunk in request_response.iter_content(chunk_size=1024):
                if not chunk: continue
                tar_download.write(chunk)
                download_progress.update(download_task, advance=len(chunk))

    with (
        Progress(
            TextColumn("Extracting dataset"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}")
        ) as extract_progress,
        tar_open(temp_tar_path, mode="r:gz") as tarfile
    ):
        tarfile_members = tarfile.getmembers()
        extract_task = extract_progress.add_task("", total=len(tarfile_members))

        for member in tarfile_members:
            tarfile.extract(member=member, path=RAW_DATA_DIRECTORY.parent, numeric_owner=True)
            extract_progress.update(extract_task, advance=1)


@data_cli_app.command("convert")
def convert_raw_data(
    input: Annotated[
        Path, Option(help="The input path of the music data")
    ] = RAW_DATA_DIRECTORY,
    output: Annotated[
        Path, Option(help="The output path for the converted musics")
    ] = PREPROCESSED_DATA_DIRECTORY,
):
    assert (
        input.exists() and input.is_dir()
    ), f"{input.absolute()} does not exist or is not a directory"

    output.mkdir(parents=True, exist_ok=True)

    conversion_results = convert_all_midi_files(
        midi_directory=RAW_DATA_DIRECTORY, wav_directory=PREPROCESSED_DATA_DIRECTORY
    )

    return output.absolute()
