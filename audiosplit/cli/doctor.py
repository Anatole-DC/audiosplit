"""
All source code related to the doctor CLI command.
Performs healthcheck on project's architecture, dependencies, environment variables, etc...
"""

from typing import Callable
from rich.progress import Progress, SpinnerColumn, TextColumn

from audiosplit.config.environment import DATA_DIRECTORY

TaskFunction = Callable[[], bool]


def check_data_directory_exists() -> bool:
    return DATA_DIRECTORY.exists()


def check_dependencies_installed() -> bool:
    try:
        import sklearn
        import tensorflow
        import librosa
    except ImportError as error:
        print(f"Missing libraries {error}")
        return False
    return True


DOCTOR_TASKS = [
    {
        "task_description": "Validate data directory architecture",
        "task_function": check_data_directory_exists,
    },
    {
        "task_description": "Check dependencies",
        "task_function": check_dependencies_installed,
    },
]


def checks():
    with Progress(SpinnerColumn(), TextColumn("{task.description}")) as progress:

        # progress_task = progress.add_task("", total=len(DOCTOR_TASKS))

        progress_tasks = [
            progress.add_task(task["task_description"], total=1)
            for task in DOCTOR_TASKS
        ]

        def perform_task(task_id, task_function: TaskFunction, task_description: str):
            # progress.update(task_id, description=f"... {task_description}")

            is_task_success = task_function()
            task_result_symbol = [":cross_mark:", ":white_check_mark:"][is_task_success]

            progress.update(
                task_id,
                advance=1,
                description=f"{task_result_symbol} {task_description}",
            )
            return is_task_success

        for task_id, task in zip(progress_tasks, DOCTOR_TASKS):
            perform_task(task_id, **task)
