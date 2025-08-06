"""Utility functions and classes for logging and timing code execution."""

import logging
import signal
import time
from typing import Any, Callable, Dict, Optional

import wandb

logger = logging.getLogger(__name__)


class Logger:
    """Encapsulates logging functionalities."""

    _logged_in = False

    def __init__(self, run_name: str, project_name: str = "hcnn") -> None:
        """Initializes the Logger and creates a new wandb run.

        Args:
            run_name (str): The name of the run to be logged.
            project_name (str): The name of the project.
        """
        if not Logger._logged_in:
            wandb.login()
            Logger._logged_in = True

        self.run_name = run_name
        self.run = wandb.init(
            project=project_name,
            name=self.run_name,
            id=self.run_name,
        )

    def __enter__(self) -> "Logger":
        """Enters the runtime context for Logger.

        Returns:
            Logger: The current instance.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exits the runtime context and finishes the wandb run."""
        wandb.finish()

    def log(
        self,
        t: int,
        data: Dict[str, Any],
    ):
        """Logs data.

        Args:
            t (int): An indexing parameter (for example, the epoch).
            data (Dict[str, Any]): A dictionary of variable names and values to log.
        """
        wandb.log(data, step=t)

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


class GracefulShutdown:
    """A context manager for graceful shutdowns."""

    stop = False

    def __init__(self, exit_message: Optional[str] = None):
        """Initializes the GracefulShutdown context manager.

        Args:
            exit_message (str): The message to log upon shutdown.
        """
        self.exit_message = exit_message

    def __enter__(self):
        """Register the signal handler."""

        def handle_signal(signum, frame):
            self.stop = True
            if self.exit_message:
                logger.info(self.exit_message)

        signal.signal(signal.SIGINT, handle_signal)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Unregister the signal handler."""
        pass
