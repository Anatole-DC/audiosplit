"""
Add all data preprocessing code in this module.
"""

import os
from pathlib import Path
from multiprocessing.pool import ThreadPool
from typing import List

import pretty_midi
import soundfile as sf
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
)
from rich.prompt import Confirm

from audiosplit.config.environment import DATA_DIRECTORY, NUMBER_OF_THREADS


def midi_to_wav_converter(midi_file: Path, wav_file: Path, sample_rate=44100):
    """
    Converts a MIDI file into an audio file.

    Arguments:
        midi_file (str): path to the input MIDI file
        wav_file (str): path to the output WAV file
        sample_rate (int): sample rate for the audio output (default: 44100)

    Returns True if the conversion is sucessful, else False
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(str(midi_file.absolute()))
        audio = midi_data.synthesize(fs=sample_rate)

        if len(audio) == 0:
            print("❌​ Warning: {midi_file} generated empty audio")
            return False

        sf.write(file=str(wav_file.absolute()), data=audio, samplerate=sample_rate)
        return True

    except Exception as e:
        print(f"​❌​ Error while converting {midi_file}: {e}")
        return False


def convert_all_midi_files(
    midi_directory: Path, wav_directory: Path, sample_rate=44100, size="all"
):
    """
    Converts all the midi files contained in midi_dir into wav files and stores them in the wav_dir

    Arguments:
        midi_dir (str): directory containing MIDI files (it may contain subfolders)
        wav_dir (str): directory where WAV files will be stored
        sample_rate (int): sample rate for the audio output (default: 44100)
        size (str / int): 'all' to process all files, int for the number of files to process

    Returns: a dictionnary with the number of converted files as well as failed conversions
    """
    assert (
        midi_directory.exists() and midi_directory.is_dir()
    ), "Source MIDI directory does not exist or is not a directory"
    assert (
        wav_directory.exists() and wav_directory.is_dir()
    ), "Destination WAV directory does not exist or is not a directory"

    # get the list of all the midi files in the midi_directory:
    midi_files: List[Path] = []
    for dirpath, _, filenames in midi_directory.walk():
        midi_files.extend(
            [
                Path(dirpath) / filename
                for filename in filenames
                if filename.lower().endswith(".mid")
            ]
        )

    if not midi_files:
        print(f"❌​ No MIDI files found in {midi_directory}")
        return {"success": 0, "failed": 0, "total": 0}

    print(f"👀👀​​ {len(midi_files)} MIDI files have been found. 👀​👀​")

    # Get existing paths in wav directory to avoid extra pre-processing
    processed_wav_file_paths = [file.stem for file in wav_directory.glob("*.wav")]
    if len(processed_wav_file_paths) != 0 and Confirm.ask(
        f"{len(processed_wav_file_paths)} preprocessed files were found, do you want to discard them in this preprocessing ?"
    ):
        midi_files = [
            file for file in midi_files if file.stem not in processed_wav_file_paths
        ]

    # if we want to process all the midi files
    if size == "all":
        files_to_process = midi_files
    else:
        files_to_process = midi_files[:size]
        ## if we want a random sample : files_to_process = random.sample(midi_files, size)

    print(f"🚀​ {len(files_to_process)} MIDI files will be processed")

    number_of_files_converted = number_of_failed_conversions = 0

    with Progress(
        TextColumn("Converting audio"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:

        progress_task = progress.add_task("", total=len(files_to_process))

        def audio_conversion_task(file_path: Path):
            wav_file_path = wav_directory / (file_path.stem + ".wav")

            # convert the midi file into wav
            result = midi_to_wav_converter(
                midi_file=file_path, wav_file=wav_file_path, sample_rate=sample_rate
            )

            progress.update(
                progress_task, advance=1, description=f"Processing {file_path.name}"
            )
            return result

        with ThreadPool(processes=NUMBER_OF_THREADS) as pool:
            all_word_urls = pool.starmap(
                audio_conversion_task, [(file_path,) for file_path in files_to_process]
            )

        number_of_files_converted = len([result for result in all_word_urls if result])
        number_of_failed_conversions = len(files_to_process) - number_of_files_converted

    print(f"✅​ {len(files_to_process)} MIDI files have been processed.")
    print(
        f"✅​ {number_of_files_converted} MIDI files have been successfuly converted into WAV files. They have been stored in {wav_directory}."
    )
    print(f"❌​ {number_of_failed_conversions} conversions have failed.")
    return {
        "success": number_of_files_converted,
        "failed": number_of_failed_conversions,
        "total": len(files_to_process),
    }


if __name__ == "__main__":
    convert_all_midi_files(
        midi_directory=DATA_DIRECTORY,
        wav_directory=os.path.join(DATA_DIRECTORY, "wav_data"),
        sample_rate=44100,
        size=15,
    )
