"""
Add all data preprocessing code in this module.
"""

import os
import pretty_midi
import soundfile as sf

from audiosplit.config.environment import DATA_DIRECTORY



def midi_to_wav_converter(midi_file: str, wav_file: str, sample_rate=44100):
    """
    Converts a MIDI file into an audio file.

    Arguments:
        midi_file (str): path to the input MIDI file
        wav_file (str): path to the output WAV file
        sample_rate (int): sample rate for the audio output (default: 44100)

    Returns True if the conversion is sucessful, else False
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        audio = midi_data.synthesize(fs=sample_rate)

        if len(audio) == 0:
            print("❌​ Warning: {midi_file} generated empty audio")
            return False

        sf.write(file=wav_file, data=audio, samplerate=sample_rate)
        return True


    except Exception as e:
        print(f"​❌​ Error while converting {midi_file}: {e}")
        return False




def convert_all_midi_files(midi_dir: str, wav_dir: str, sample_rate=44100, size='all'):
    """
    Converts all the midi files contained in midi_dir into wav files and stores them in the wav_dir

    Arguments:
        midi_dir (str): directory containing MIDI files (it may contain subfolders)
        wav_dir (str): directory where WAV files will be stored
        sample_rate (int): sample rate for the audio output (default: 44100)
        size (str / int): 'all' to process all files, int for the number of files to process

    Returns: a dictionnary with the number of converted files as well as failed conversions
    """
    # create the wav directory if it doesn't exist yet:
    os.makedirs(wav_dir, exist_ok=True)

    # get the list of all the midi files in the midi_directory:
    midi_files = []
    for dirpath, dirnames, filenames in os.walk(midi_dir):
        for filename in filenames:
            if filename.lower().endswith(".mid"):
                midi_files.append(os.path.join(dirpath, filename))

    if not midi_files:
        print(f"❌​ No MIDI files found in {midi_dir}")
        return {'success': 0,
                'failed': 0,
                'total': 0}

    print(f"👀👀​​ {len(midi_files)} MIDI files have been found. 👀​👀​")

    # if we want to process all the midi files
    if size == 'all':
        files_to_process = midi_files

    else:
        files_to_process = midi_files[:size]
        ## if we want a random sample : files_to_process = random.sample(midi_files, size)

    print(f"🚀​ {len(files_to_process)} MIDI files will be processed")

    number_of_files_converted = 0
    number_of_failed_conversions = 0

    for f in files_to_process:
        # get the name of the wav file (keeping all the infos from the origin dataset but replacing '/' and '\' by '-')
        wav_file_name = f.replace("\\", "-").replace("/", "-").replace(".mid", ".wav")
        wav_file = os.path.join(wav_dir, wav_file_name)

        # convert the midi file into wav
        if midi_to_wav_converter(midi_file=f,
                                 wav_file=wav_file,
                                 sample_rate=sample_rate):
            number_of_files_converted += 1
        else:
            number_of_failed_conversions += 1

    print(f"✅​ {len(files_to_process)} MIDI files have been processed.")
    print(f"✅​ {number_of_files_converted} MIDI files have been successfuly converted into WAV files. They have been stored in {wav_dir}.")
    print(f"❌​ {number_of_failed_conversions} conversions have failed.")
    return {'success': number_of_files_converted,
            'failed': number_of_failed_conversions,
            'total': len(files_to_process)}



if __name__ == '__main__':
    convert_all_midi_files(midi_dir=DATA_DIRECTORY,
                           wav_dir=os.path.join(DATA_DIRECTORY, 'wav_data'),
                           sample_rate=44100,
                           size=15)
