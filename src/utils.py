"""Utility functions and classes for logging and timing code execution."""

import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import wandb

PROJECT = "hcnn"

# Log in to wandb automatically when the module is imported.
wandb.login()


class Logger:
    """Encapsulates logging functionalities."""

    def __init__(self, dataset: str) -> None:
        """Initializes the Logger and creates a new wandb run.

        Args:
            dataset (str): The name of the dataset to be logged.
        """
        self.dataset = dataset
        self.run = wandb.init(project=PROJECT)
        self.run.name = dataset + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def __enter__(self) -> "Logger":
        """Enters the runtime context for Logger.

        Returns:
            Logger: The current instance.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exits the runtime context and finishes the wandb run."""
        wandb.finish()

    class Timer:
        """A context manager for timing code execution and logging the results."""

        def __init__(
            self,
            label: str,
            t: int,
            log_vars: Optional[Callable[[], Dict[str, Any]]] = None,
        ) -> None:
            """Initializes the Timer context manager.

            Args:
                label (str): The label for the timed section.
                t (int): An indexing parameter (for example, the epoch).
                log_vars (Optional[Callable[[], Dict[str, Any]]], optional):
                    A callable that returns a dictionary of variable names and values
                    to log. Defaults to None.
            """
            self.label = label
            self.t = t
            self.log_vars = log_vars

        def __enter__(self) -> "Logger.Timer":
            """Starts the timer.

            Returns:
                Logger.Timer: The Timer instance.
            """
            self.start_time = time.time()
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            """Stops the timer and logs the elapsed time and optional variables."""
            end_time = time.time()
            elapsed_time = end_time - self.start_time
            # Prepare log data with elapsed time and the provided integer parameter.
            log_data = {self.label: elapsed_time, f"{self.label}_t": self.t}
            # If a callable to fetch additional variables is provided, update log data.
            if self.log_vars is not None:
                log_data.update(self.log_vars())
            wandb.log(log_data)

    def timeit(
        self,
        label: str,
        t: int,
        log_vars: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> "Logger.Timer":
        """Creates a Timer context manager to time a code block and log its metrics.

        Args:
            label (str): The label for the timed section.
            t (int): An integer parameter (e.g., a threshold).
            log_vars (Optional[Callable[[], Dict[str, Any]]], optional):
                A callable that returns a dictionary of variable names and values
                to be logged. Defaults to None.

        Returns:
            Logger.Timer: A Timer context manager.
        """
        return Logger.Timer(label, t, log_vars)


if __name__ == "__main__":
    # Example usage of Logger and Timer
    value_a = 10
    value_b = 20

    def capture_vars() -> Dict[str, Any]:
        """Capture variables to be logged."""
        return {"value_a": value_a, "value_b": value_b}

    # Using the Logger in a with statement
    with Logger("test-logger") as logger:
        print("Logger started. Check your wandb dashboard.")
        # Timing a code block using the Timer context manager
        with logger.timeit("processing_time", t=1, log_vars=capture_vars):
            # Simulate some processing
            time.sleep(2)
            # Update the values of the variables
            value_a = 15
            value_b = 25
        print("Timer block ended. Logged metrics to wandb.")
