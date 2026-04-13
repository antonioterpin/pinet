"""Utility functions and classes for logging and timing code execution."""

import logging
import signal
from typing import Any

import yaml

import wandb

logger = logging.getLogger(__name__)


def load_configuration(file_path: str) -> dict:
    """Load configuration file from yaml.

    Args:
        file_path: Path to the configuration file.

    Returns:
        dict: A dictionary containing the configuration parameters.
    """
    with open(file_path) as file:
        hyperparameters = yaml.safe_load(file)
    return hyperparameters


class Logger:
    """Encapsulates logging functionalities."""

    _logged_in = False

    def __init__(self, run_name: str, project_name: str = "hcnn") -> None:
        """Initializes the Logger and creates a new wandb run.

        Args:
            run_name: The name of the run to be logged.
            project_name: The name of the project.
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

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        """Exits the runtime context and finishes the wandb run.

        Args:
            exc_type: The exception type raised inside the context, if any.
            exc_value: The exception value raised inside the context, if any.
            traceback: The traceback raised inside the context, if any.
        """
        wandb.finish()

    def log(self, t: int, data: dict[str, Any]) -> None:
        """Logs data.

        Args:
            t: An indexing parameter (for example, the epoch).
            data: A dictionary of variable names and values to log.
        """
        wandb.log(data, step=t)


class GracefulShutdown:
    """A context manager for graceful shutdowns.

    Attributes:
        stop: Whether a shutdown signal has been received.
    """

    stop = False

    def __init__(self, exit_message: str | None = None) -> None:
        """Initializes the GracefulShutdown context manager.

        Args:
            exit_message: The message to log upon shutdown.
        """
        self.exit_message = exit_message

    def __enter__(self) -> "GracefulShutdown":
        """Register the signal handler.

        Returns:
            GracefulShutdown: The current instance.
        """

        def handle_signal(signum: int, frame: Any) -> None:
            self.stop = True
            if self.exit_message:
                logger.info(self.exit_message)

        signal.signal(signal.SIGINT, handle_signal)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        """Unregister the signal handler.

        Args:
            exc_type: The exception type raised inside the context, if any.
            exc_value: The exception value raised inside the context, if any.
            traceback: The traceback raised inside the context, if any.
        """
        pass
