# Dataset

We are using [colin reffel's Lakh MIDI dataset][dataset-website], which contains over 176 581 MIDI files.

## Exploratory work about the dataset

> WIP

## Data pre-processing

To pre-process the data, we extract it using [pretty-midi][pretty-midi] and [soundfile][soundfile]. We extract individual instrument tracks for the training process.

When converting the data we are creating a [multitrack file](/audiosplit/data/multitrack.py), which is just a pickle file, containing all the individual audio tracks. We use this method for storage convenience, as it will be easier to retrieve all the audio this way.

Each individual instrument track is stored under the following name.

$$
{\color{red}\overbrace{bdcb99e965695a808b5c89be5c22a082}^{Song\ number}}\_
{\color{blue}\overbrace{string-ensemble-1}^{Instrument\ name}}\_
{\color{green}\overbrace{0}^{Instrument\ index}}\_
{\color{purple}\overbrace{ensemble}^{Instrument\ category}}
$$

> The instrument index is here when a single instrument is represented more than once in the same music.  
> ex: Two violins inside a music.

Refer to the [midi file documentation](./midi_files.md#general-informations-about-midi-classes) to learn more about the midi file structure.

[dataset-website]:https://colinraffel.com/projects/lmd/
[pretty-midi]:https://github.com/craffel/pretty-midi
[soundfile]:https://python-soundfile.readthedocs.io/en/0.13.1/
