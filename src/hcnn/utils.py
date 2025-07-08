"""Utility functions and classes for logging and timing code execution."""

import logging
import signal
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
import wandb

PROJECT = "hcnn"

# Log in to wandb automatically when the module is imported.
wandb.login()
logger = logging.getLogger(__name__)


class Logger:
    """Encapsulates logging functionalities."""

    def __init__(self, run_name: str) -> None:
        """Initializes the Logger and creates a new wandb run.

        Args:
            run_name (str): The name of the run to be logged.
        """
        self.run_name = run_name
        self.run = wandb.init(
            project=PROJECT,
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


# Inputs dataclasses
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EqualityInputs:
    """Dataclass representing inputs used in forming equality constraints.

    Attributes:
        b (Optional[jnp.ndarray]): Vector representing the RHS of the equality constraint.
            Shape (batch_size, n_constraints, 1)
        A (Optional[jnp.ndarray]): Matrix representing the LHS of the equality constraint.
            Shape (batch_size, n_constraints, dimension).
        Apinv (Optional[jnp.ndarray]): The pseudoinverse of the matrix A.
            Shape (batch_size, dimension, n_constraints).
    """

    b: Optional[jnp.ndarray] = None
    A: Optional[jnp.ndarray] = None
    Apinv: Optional[jnp.ndarray] = None

    def update(self, **kwargs):
        """Update some attribute by keyword."""
        return replace(self, **kwargs)


# Inputs dataclasses
# TODO: Add dataclass for box constraints.
# TODO: Add dataclass for Inequality constraints.


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Inputs:
    """A dataclass for encapsulating model input parameters.

    Attributes:
        x (jnp.ndarray): The point to be projected.
            Shape (batch_size, dimension, 1)
        eq (EqualityInputs):
            An instance containing auxiliary inputs
            related to equality constraints.
    """

    x: jnp.ndarray
    eq: Optional[EqualityInputs] = EqualityInputs()

    def update(self, **kwargs):
        """Update some attribute by keyword."""
        return replace(self, **kwargs)
