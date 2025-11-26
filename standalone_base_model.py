"""
Standalone BaseModel for ODE-based Models

A minimal implementation of BaseModel that can be used independently
for running ODE-based models like the ECG model from Project 1.

This is extracted from the compneuro package to allow standalone usage.
"""

import numpy as np
from collections import OrderedDict
import typing as tp


class BaseModel:
    """
    Minimal Base Model Class for ODE-based models

    This provides the essential functionality needed to run ODE models
    like the ECG model from Project 1.
    """

    # Default parameters - should be overridden in subclasses
    Default_Params: OrderedDict = OrderedDict()

    # Default states - should be overridden in subclasses
    # Format: {name: (initial_value, min_bound, max_bound)} or {name: initial_value}
    Default_States: OrderedDict = OrderedDict()

    def __init__(self, num: int = 1, **kwargs):
        """
        Initialize the model

        Args:
            num: number of parallel simulations (usually 1)
            **kwargs: parameter and state overrides
        """
        # Initialize parameters
        self.params = OrderedDict(self.Default_Params.copy())

        # Initialize states
        self.states = OrderedDict()
        self.initial_states = OrderedDict()
        self.bounds = OrderedDict()

        for var_name, var_val in self.Default_States.items():
            var_val = np.atleast_1d(var_val)
            if len(var_val) == 1:
                # Just initial value
                self.initial_states[var_name] = var_val[0].copy() if hasattr(var_val[0], 'copy') else var_val[0]
                self.states[var_name] = var_val[0].copy() if hasattr(var_val[0], 'copy') else var_val[0]
                self.bounds[var_name] = None
            elif len(var_val) == 3:
                # (initial, min, max)
                self.initial_states[var_name] = var_val[0].copy() if hasattr(var_val[0], 'copy') else var_val[0]
                self.states[var_name] = var_val[0].copy() if hasattr(var_val[0], 'copy') else var_val[0]
                self.bounds[var_name] = (var_val[1], var_val[2])
            else:
                raise ValueError(f"State variable {var_name} should have 1 or 3 elements, got {len(var_val)}")

        # Apply any overrides from kwargs
        for key, val in kwargs.items():
            if key in self.params:
                self.params[key] = val
            elif key in self.states:
                self.states[key] = val
                self.initial_states[key] = val
            else:
                raise ValueError(f"Unrecognized parameter/state: {key}")

    def ode(self, t: float, states: np.ndarray, stimuli: tp.Any = None) -> np.ndarray:
        """
        Define the ODE system: dx/dt = f(t, x, stimuli)

        This should be overridden in subclasses.

        Args:
            t: current time
            states: current state vector
            stimuli: external stimuli (optional)

        Returns:
            derivative vector dx/dt
        """
        raise NotImplementedError("Subclasses must implement the ode() method")

    def solve(self, t: np.ndarray, solver: str = "Euler", verbose: bool = True,
              **stimuli) -> tp.Dict[str, np.ndarray]:
        """
        Solve the ODE system over time array t

        Args:
            t: time array
            solver: integration method ("Euler" only for now)
            verbose: whether to print progress
            **stimuli: additional stimuli passed to ode()

        Returns:
            Dictionary with state variable names as keys and time series as values
        """
        if solver != "Euler":
            raise ValueError("Only 'Euler' solver is implemented in this standalone version")

        # Get initial states as array
        state_names = list(self.states.keys())
        y0 = np.array([self.states[name] for name in state_names])

        # Prepare output arrays
        nt = len(t)
        n_states = len(state_names)
        results = {name: np.zeros(nt) for name in state_names}

        # Set initial values
        for i, name in enumerate(state_names):
            results[name][0] = y0[i]

        # Euler integration
        y = y0.copy()
        dt_prev = t[1] - t[0] if len(t) > 1 else 0

        for i in range(1, nt):
            dt = t[i] - t[i-1]

            # Apply bounds if specified
            for j, name in enumerate(state_names):
                if self.bounds[name] is not None:
                    min_val, max_val = self.bounds[name]
                    y[j] = np.clip(y[j], min_val, max_val)

            # Compute derivatives
            dydt = self.ode(t[i-1], y, stimuli)

            # Euler step
            y = y + dt * dydt

            # Store results
            for j, name in enumerate(state_names):
                results[name][i] = y[j]

            if verbose and i % max(1, nt//10) == 0:
                print(".1f")

        if verbose:
            print("100.0f")

        return results


# Example usage and ECG model for testing
if __name__ == "__main__":
    # Example: Simple harmonic oscillator
    class HarmonicOscillator(BaseModel):
        Default_Params = OrderedDict([
            ("omega", 2 * np.pi),  # frequency
            ("damping", 0.1),      # damping coefficient
        ])

        Default_States = OrderedDict([
            ("x", 1.0),      # position
            ("v", 0.0),      # velocity
        ])

        def ode(self, t, states, stimuli=None):
            x, v = states
            omega = self.params["omega"]
            damping = self.params["damping"]

            # dx/dt = v
            # dv/dt = -omega^2 * x - damping * v
            return np.array([
                v,
                -omega**2 * x - damping * v
            ])

    # Test the harmonic oscillator
    print("Testing Harmonic Oscillator...")
    model = HarmonicOscillator(omega=2*np.pi, damping=0.1)
    t = np.linspace(0, 10, 1000)
    result = model.solve(t, verbose=True)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, result["x"], label="Position")
    plt.title("Harmonic Oscillator")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, result["v"], label="Velocity", color='orange')
    plt.xlabel("Time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Standalone BaseModel test completed successfully!")