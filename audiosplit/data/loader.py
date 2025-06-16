"""
Add all data loading related code in this module.
"""

from pathlib import Path
from typing import List
from random import sample

from audiosplit.config.environment import (
    PREPROCESSED_DATA_DIRECTORY,
    RAW_DATA_DIRECTORY,
)


def load_preprocessed_music_paths(number_of_samples: int = None) -> List[Path]:
    assert (
        PREPROCESSED_DATA_DIRECTORY.exists()
    ), f"Tried to load data but the preprocessed directory does not exist. Value given is {PREPROCESSED_DATA_DIRECTORY.absolute()}"
    assert (
        PREPROCESSED_DATA_DIRECTORY.is_dir()
    ), f"Preprocessed directory path exists but is not a directory. Value given is {PREPROCESSED_DATA_DIRECTORY.absolute()}"

    preprocessed_audio_file_paths: List[Path] = list(
        PREPROCESSED_DATA_DIRECTORY.glob("*.wav")
    )

    if number_of_samples is None:
        return preprocessed_audio_file_paths
    return sample(preprocessed_audio_file_paths, number_of_samples)


def load_midi_music_paths(number_of_samples: int = None) -> List[Path]:
    assert (
        RAW_DATA_DIRECTORY.exists()
    ), f"Tried to load data but the raw directory does not exist. Value given is {RAW_DATA_DIRECTORY.absolute()}"
    assert (
        RAW_DATA_DIRECTORY.is_dir()
    ), f"Raw directory path exists but is not a directory. Value given is {RAW_DATA_DIRECTORY.absolute()}"


if __name__ == "__main__":
    print(load_preprocessed_music_paths(5))
