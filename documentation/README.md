# Audiosplit : project's documentation

In this section, you will find all the project's core concepts documentation. This includes reasearch papers and existing models, technical concepts about the data we are manipulating, etc...

## Dataset

You can visit [this website][dataset-website] to learn more about the initial dataset we are using, or see [this section of the documentation](./dataset.md) to look at how we are using the dataset, and processing the data.

## Repository architecture

Look at [this section](./project_architecture.md) if you need information about the project's folders.

## Existing models

The existing approachs that we have currently found all use only a fraction of all the available classes. Usually the "drums", the "piano", the "voice", and the rest of the channels as an "other" channels.

- [Deezer spleeter project][spleeter]: Deezer's version of the project
- [Facebook's demucs][demucs]: Former facebook employee's version of the project
- [U-Net for music channels][spectrogram-channels-unet]: A U-Net segmentation models for splitting music channels.

## Core technical concepts

The following section explains the technical concepts behind the project.

### Midi files

[This section](./midi_files.md) goes in detail about MIDI files, which are the raw data we are using to train the model.


[spleeter]:https://research.deezer.com/projects/spleeter.html
[spectrogram-channels-unet]:https://arxiv.org/pdf/1810.11520
[demucs]:https://github.com/adefossez/demucs

[dataset-website]:https://colinraffel.com/projects/lmd/
