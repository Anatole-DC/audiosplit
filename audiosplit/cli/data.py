from typing_extensions import Annotated, List, Optional
from typer import Typer, Argument, Option
from pathlib import Path

from audiosplit.config.environment import (
    RAW_DATA_DIRECTORY,
    PREPROCESSED_DATA_DIRECTORY,
)
from audiosplit.data.preprocessing import convert_all_midi_files

data_cli_app = Typer()


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
