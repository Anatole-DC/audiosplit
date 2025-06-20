from copy import deepcopy
from pathlib import Path
from typing import Dict
from pickle import dump as pkl_save

from numpy.typing import NDArray
from pretty_midi import (
    PrettyMIDI,
    Instrument,
    program_to_instrument_name,
    program_to_instrument_class,
)


Multitrack = Dict[str, NDArray]


def isolate_music_track(music: PrettyMIDI, instrument_index: int) -> PrettyMIDI:
    music_copy = deepcopy(music)
    music_copy.instruments = [music.instruments[instrument_index]]
    return music_copy


def multitrack_instrument_name(instrument: Instrument, instrument_index: int) -> str:
    instrument_name = (
        (
            program_to_instrument_name(instrument.program)
            if not instrument.is_drum
            else "drums"
        )
        .lower()
        .replace(" ", "-")
    )
    instrument_class = program_to_instrument_class(instrument.program).lower()
    return f"{instrument_name}_{instrument_index}_{instrument_class}"


def midi_to_multitrack(music: PrettyMIDI) -> Multitrack:
    """Extract a MIDI file to a multitrack object.

    Args:
        music (PrettyMIDI): The MIDI file to extact.

    Returns:
        Multitrack: The multitrack object.
    """

    multi_track_content: Multitrack = {"music": music.fluidsynth()}

    for index, instrument in enumerate(music.instruments):
        track_name = multitrack_instrument_name(instrument, index)
        music_track = isolate_music_track(music, index)
        multi_track_content[track_name] = music_track.fluidsynth()

    return multi_track_content


def save_multitrack(
    multitrack_music: Multitrack, output_directory: Path, music_name: str
) -> Path:
    assert (
        output_directory.exists()
    ), f"Output path '{output_directory.absolute()}' for multitrack save does not exist."
    assert (
        output_directory.is_dir()
    ), f"Output path '{output_directory.absolute()}' for multitrack save exists but is not a directory."

    file_path = output_directory / f"{music_name}.pkl"
    with open(file_path, "wb") as pkl_file:
        pkl_save(multitrack_music, pkl_file)

    return file_path
