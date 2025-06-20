"""
Store all package's environment variables.
"""

from os import environ
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Data related variables
DATA_DIRECTORY = Path(environ.get("DATA_DIRECTORY", "data"))

RAW_DATA_DIRECTORY = DATA_DIRECTORY / "raw"
PREPROCESSED_DATA_DIRECTORY = DATA_DIRECTORY / "preprocessed"

# Optimization variables
NUMBER_OF_THREADS = int(environ.get("NUMBER_OF_THREADS", 10))
