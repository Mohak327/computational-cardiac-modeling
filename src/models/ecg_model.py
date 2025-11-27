import numpy as np
from collections import OrderedDict
from src.models.base_model import BaseModel

class ECG_Model(BaseModel):
    PARAMS_DICT = {
        "healthy": dict(C=1.35, H=3.0, beta=4.0, gamma=7.0),
        "tachycardia": dict(C=1.35, H=2.848, beta=4.0, gamma=21.0)
    }

    DEFAULT_STATES = OrderedDict([
        ("x1", (0.0, -np.inf, np.inf)),
        ("x2", (0.0, -np.inf, np.inf)),
        ("x3", (0.1, -np.inf, np.inf)),
        ("x4", (0.0, -np.inf, np.inf)),
    ])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = None
        self.alpha_dict = {
            "healthy": np.asarray([-0.024, 0.0216, -0.0012, 0.12], float),
            "tachycardia": np.asarray([0.0, -0.1, 0.0, 0.0], float)
        }

        self.states = OrderedDict([
            ("x1", 0.0),
            ("x2", 0.0),
            ("x3", 0.1),
            ("x4", 0.0),
        ])

        self.bounds = OrderedDict([
            ("x1", (-np.inf, np.inf)),
            ("x2", (-np.inf, np.inf)),
            ("x3", (-np.inf, np.inf)),
            ("x4", (-np.inf, np.inf)),
        ])

    def _plot_ecg(self, t, ECG, case):
        from src.utils.plotting import plot_ecg_signal
        plot_ecg_signal(t, ECG, title=f"{case.capitalize()} ECG Signal")

    def ode(self, t, states, stimuli=None):
        C = self.params["C"]
        H = self.params["H"]
        beta = self.params["beta"]
        gamma = self.params["gamma"]

        x1, x2, x3, x4 = states

        x1_ode = x1 - x2 - (C*x1*x2) - x1*(x2**2)
        x2_ode = H*x1 - 3*x2 + C*x1*x2 + x1*x2**2 + beta*(x4 - x2)
        x3_ode = x3 - x4 - (C*x3*x4) - x3*(x4**2)
        x4_ode = H*x3 - 3*x4 + C*x3*x4 + x3*(x4**2) + 2*beta*(x2 - x4)

        return gamma * np.array([x1_ode, x2_ode, x3_ode, x4_ode])

    def ecg(self, X):
        arr = np.asarray(X)

        if arr.ndim == 1:
            return float(self.alpha @ arr)

        if arr.shape[0] == 4:
            return self.alpha @ arr

        return arr @ self.alpha

    def run(self, t, case="healthy", solver="Euler", plot=True):
        try:
            params = self.PARAMS_DICT[case]
        except KeyError:
            raise ValueError(f"Unknown case type '{case}'. Use one of: {list(self.PARAMS_DICT.keys())}")
        self.alpha = self.alpha_dict[case]
        self.params.update(params)
        
        self.states = self.DEFAULT_STATES.copy()
        
        y0 = list(self.states.values())
        out = self.solve(t, solver=solver, verbose=False, y0=y0)
        try:
            X = np.vstack([out[k] for k in ("x1", "x2", "x3", "x4")]).T
        except KeyError as e:
            raise KeyError(f"Missing key in output: {e}. Available keys: {list(out.keys())}")
        except ValueError as e:
            raise ValueError(f"Shape mismatch in output arrays: {e}. Array shapes: {[out[k].shape for k in ('x1', 'x2', 'x3', 'x4')]}")
        ECG = self.ecg(X)
        if plot:
            self._plot_ecg(t, ECG, case)
        return ECG, X