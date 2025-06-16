"""
Package's CLI entry point.
"""

from typer import Typer

from .doctor import checks
from .data import data_cli_app


audiosplit_cli = Typer(
    name="Audiosplit",
    help="Audiosplit CLI for everything related to data, models, etc...",
)

audiosplit_cli.add_typer(data_cli_app, name="data")


@audiosplit_cli.command("doctor")
def doctor():
    checks()


def main():
    audiosplit_cli()


if __name__ == "__main__":
    main()
