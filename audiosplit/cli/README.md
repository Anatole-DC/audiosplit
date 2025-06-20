# CLI

The CLI module is where to place all CLI related commands specifically related to your project. When you eventually distribute your package, these commands will be available to your project's users.

You can display all the commands by typing :

```bash
asplit --help
```

## Available commands

The following commands are available at the moment.

### Doctor

Audiosplit's diagnostic

### Data

To download the data :

```bash
asplit data download
```

To convert MIDI files to train data :

```bash
asplit data convert
```

## Contribute

If you are developping the CLI and your modifications are not applied when running the commands, run the following command to re-install the CLI.

```bash
pip install -e .
```

**Please remember to update this document as well when adding new commands !**
