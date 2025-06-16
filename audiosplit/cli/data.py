from typing_extensions import Annotated
from typer import Typer, Option
from pathlib import Path

from rich.prompt import Confirm

from audiosplit.config.environment import (
    RAW_DATA_DIRECTORY,
    PREPROCESSED_DATA_DIRECTORY,
)
from audiosplit.data.preprocessing import convert_all_midi_files
from audiosplit.data.dataset import (
    DatasetVersion,
    download_dataset,
    extract_downloaded_dataset,
)


data_cli_app = Typer()


@data_cli_app.command("download", help="Download dataset from the online website.")
def download_data(
    dataset: Annotated[
        DatasetVersion, Option(help="The version of dataset do you want to download.")
    ] = DatasetVersion.matched,
):
    RAW_DATA_DIRECTORY.mkdir(exist_ok=True, parents=True)

    temp_tar_path = RAW_DATA_DIRECTORY / f"{dataset.value}.tar.gz"

    use_existing_dataset = temp_tar_path.exists() and Confirm.ask(
        f"Archive of the {dataset.value} dataset found. Do you want to use it ?"
    )
    if not temp_tar_path.exists() or not use_existing_dataset:
        assert download_dataset(
            dataset, temp_tar_path
        ).exists(), f"Could not retrieve dataset at path '{temp_tar_path.absolute()}' after download."

    extract_downloaded_dataset(temp_tar_path, RAW_DATA_DIRECTORY)


@data_cli_app.command("convert", help="Convert midi files into wav files.")
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

    convert_all_midi_files(
        midi_directory=RAW_DATA_DIRECTORY, wav_directory=PREPROCESSED_DATA_DIRECTORY
    )

    return output.absolute()
