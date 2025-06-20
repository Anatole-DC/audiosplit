from copy import deepcopy
from pathlib import Path
from typing import Dict, Self
from pickle import dump as pkl_save, load as pkl_load
from dataclasses import dataclass

from numpy import ndarray
from IPython.display import Audio as JupyterAudioPlayer
from pretty_midi import (
    PrettyMIDI,
    Instrument,
    program_to_instrument_name,
    program_to_instrument_class,
)


Music = ndarray
TrackName = str
MusicTracks = Dict[TrackName, Music]


@dataclass
class Multitrack:
    """Contains the original music with isolated individual music tracks."""

    name: str
    music: ndarray
    tracks: MusicTracks

    @property
    def instruments(self):
        return [instrument for instrument in self.tracks.keys()]

    @property
    def audios(self):
        return [music for music in self.tracks.values()]

    def music_player(self) -> JupyterAudioPlayer:
        return JupyterAudioPlayer(self.music, rate=44100)

    def track_player(self, track: int | str) -> JupyterAudioPlayer:
        if isinstance(track, str):
            assert (
                track in self.instruments
            ), f"The requested track '{track}' does not exist in audio."
            return JupyterAudioPlayer(self.tracks[track], rate=44100)
        assert track <= len(self.instruments)
        return JupyterAudioPlayer(self.audios[track], rate=44100)

    def save(self, output_directory: Path) -> Path:
        return save_multitrack(self, output_directory)

    @classmethod
    def from_multitrack_file(cls, multitrack_file: Path) -> Self:
        return load_multitrack(multitrack_file)

    @classmethod
    def from_midi(cls, midi_file: PrettyMIDI, music_name: str) -> Self:
        return midi_to_multitrack(midi_file, music_name)


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


def midi_to_multitrack(music: PrettyMIDI, name: str) -> Multitrack:
    """Extract a MIDI file to a multitrack object.

    Args:
        music (PrettyMIDI): The MIDI file to extact.
        name (str): The name of the music.

    Returns:
        Multitrack: The multitrack object.
    """

    multi_track_content: MusicTracks = {}

    for index, instrument in enumerate(music.instruments):
        track_name = multitrack_instrument_name(instrument, index)
        music_track = isolate_music_track(music, index)
        multi_track_content[track_name] = music_track.fluidsynth()

    return Multitrack(name, music.fluidsynth(), multi_track_content)


def save_multitrack(multitrack_music: Multitrack, output_directory: Path) -> Path:
    assert (
        output_directory.exists()
    ), f"Output path '{output_directory.absolute()}' for multitrack save does not exist."
    assert (
        output_directory.is_dir()
    ), f"Output path '{output_directory.absolute()}' for multitrack save exists but is not a directory."

    file_path = output_directory / f"{multitrack_music.name}.pkl"
    with open(file_path, "wb") as pkl_file:
        pkl_save(multitrack_music, pkl_file)

    return file_path


def load_multitrack(multitrack_path: Path) -> Multitrack:
    assert (
        multitrack_path.exists()
    ), f"Multitrack loading path '{multitrack_path.absolute()}' does not exist."
    assert (
        multitrack_path.is_file()
    ), f"Multitrack loading path '{multitrack_path.absolute()}' exist but is not a file."

    with open(multitrack_path, "rb") as pkl_file:
        multitrack_audio: Multitrack = pkl_load(pkl_file)

    assert isinstance(
        multitrack_audio, Multitrack
    ), f"Loaded multitrack '{multitrack_path.absolute()}' is not a valid Multitrack object."
    assert isinstance(
        multitrack_audio.music, ndarray
    ), f"Multitrack '{multitrack_path.absolute()}' is valid, but music is not valid."
    assert (
        len(multitrack_audio.music) > 0
    ), f"Multitrack '{multitrack_path.absolute()}' contains music, but music is empty."
    assert (
        len(multitrack_audio.tracks) > 0
    ), f"Multitrack '{multitrack_path.absolute()}' contains music but no tracks."

    return multitrack_audio
