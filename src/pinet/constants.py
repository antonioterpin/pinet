"""Constants used throughout the pinet package."""


class Constants:
    """Centralized constants used throughout the pinet package.

    Attributes:
        SOC_CONSTRAINT_EPSILON: Epsilon for SOC norm divisions.
        EQUILIBRATION_DEFAULT_TOL: Default equilibration termination tolerance.
        EQUILIBRATION_DEFAULT_MAX_ITER: Default max equilibration iterations.
        EQUILIBRATION_DEFAULT_ORD: Default norm order for equilibration.
        EQUILIBRATION_DEFAULT_COL_SCALING: Whether to apply column scaling.
        EQUILIBRATION_DEFAULT_UPDATE_MODE: Equilibration update mode.
        EQUILIBRATION_DEFAULT_SAFEGUARD: Whether to safeguard equilibration.
        PROJECTION_DEFAULT_SIGMA: Default ADMM stepsize parameter.
        PROJECTION_DEFAULT_OMEGA: Default ADMM relaxation parameter.
        PROJECTION_DEFAULT_CHECK_EVERY: Default constraint-check period.
        PROJECTION_DEFAULT_TOL: Default constraint-violation tolerance.
        PROJECTION_DEFAULT_MAX_ITER: Default max projection iterations.
        PROJECTION_DEFAULT_CHECK_REDUCTION: Default batch reduction method.
    """

    # Numerical Stability
    # ==================
    # Epsilon for protecting against division by zero in SOC norm calculations
    SOC_CONSTRAINT_EPSILON: float = 1e-12

    # Equilibration Algorithm Defaults
    # =================================
    # Default tolerance for equilibration termination criterion
    EQUILIBRATION_DEFAULT_TOL: float = 1e-3

    # Default maximum number of equilibration iterations
    EQUILIBRATION_DEFAULT_MAX_ITER: int = 0

    # Default norm order for equilibration (1, 2, or inf)
    EQUILIBRATION_DEFAULT_ORD: float = 2.0

    # Enable column scaling in equilibration
    EQUILIBRATION_DEFAULT_COL_SCALING: bool = False

    # Update mode for equilibration: "Gauss" (sequential) or "Jacobi" (simultaneous)
    EQUILIBRATION_DEFAULT_UPDATE_MODE: str = "Gauss"

    # Enable safeguard to ensure condition number doesn't increase
    EQUILIBRATION_DEFAULT_SAFEGUARD: bool = False

    # Projection/ADMM Algorithm Parameters
    # =====================================
    # Default ADMM stepsize parameter
    PROJECTION_DEFAULT_SIGMA: float = 1.0

    # Default ADMM relaxation parameter (typically between 1 and 2)
    PROJECTION_DEFAULT_OMEGA: float = 1.7

    # Default frequency of constraint violation checks during projection
    PROJECTION_DEFAULT_CHECK_EVERY: int = 10

    # Default tolerance for constraint violation during projection
    PROJECTION_DEFAULT_TOL: float = 1e-3

    # Default maximum number of projection iterations
    PROJECTION_DEFAULT_MAX_ITER: int = 100

    # Default reduction method for batch constraint checks:
    # "max", "mean", or a float in [0, 1]
    PROJECTION_DEFAULT_CHECK_REDUCTION: str = "max"
