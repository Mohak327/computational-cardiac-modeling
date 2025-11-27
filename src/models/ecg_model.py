import numpy as np
from collections import OrderedDict
from src.models.base_model import BaseModel

class ECG_Model(BaseModel):
    PARAMS_DICT = {
        "healthy": dict(C=1.35, H=0.12, beta=4.0, gamma=7.0),
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
        self.alpha_dict = {
            "healthy": np.asarray([-0.024, 0.0216, -0.0012, 0.12], float),
            "tachycardia": np.asarray([0.0, -0.1, 0.0, 0.0], float)
        }
        self.alpha = None
        self.params = {}
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
        self.set_case("healthy")  # Default case

    def set_case(self, case):
        try:
            self.params = self.PARAMS_DICT[case].copy()
            self.alpha = self.alpha_dict[case]
        except KeyError:
            raise ValueError(f"Unknown case type '{case}'. Use one of: {list(self.PARAMS_DICT.keys())}")

    def _plot_ecg(self, t, ECG, case):
        from src.utils.plotting import plot_ecg_signal
        print("plot_ecg", t, ECG)
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

    def ecg_rhs(self, t, y):
        C = self.params["C"]
        H = self.params["H"]
        beta = self.params["beta"]
        gamma = self.params["gamma"]

        x1, x2, x3, x4 = y  # Unpack state vector, 2 coupled oscillators

        # RHS computation, overflow safe
        with np.errstate(over='ignore', invalid='ignore'):
            F = np.array([
            #oscillator 1 - SA
            x1 - x2 - C*x1*x2 - x1*(x2**2),
            H*x1 - 3*x2 + C*x1*x2 + x1*(x2**2) + beta*(x4 - x2),
            #oscillator 2 - AV
            x3 - x4 - C*x3*x4 - x3*(x4**2),
            H*x3 - 3*x4 + C*x3*x4 + x3*(x4**2) + 2*beta*(x2 - x4),
            ], float)
        F[~np.isfinite(F)] = 0.0  # Replace NaN/Inf with zero to prevent blowouts
        return gamma * F  # Global timescale factor gamma

    def rk4_boxed(self, case, y0, t, box: float = 3.0) -> np.ndarray:
        y = np.zeros((len(t), len(y0)))
        y[0] = np.clip(y0, -box, box)

        for k in range(len(t) - 1):
            t0, dt = t[k], t[k + 1] - t[k]
            yk = np.clip(y[k], -box, box)
            self.set_case(case)
            k1 = self.ecg_rhs(t0, yk)
            k2 = self.ecg_rhs(t0 + dt / 2, np.clip(yk + 0.5 * dt * k1, -box, box))
            k3 = self.ecg_rhs(t0 + dt / 2, np.clip(yk + 0.5 * dt * k2, -box, box))
            k4 = self.ecg_rhs(t0 + dt, np.clip(yk + dt * k3, -box, box))

            y[k + 1] = np.clip(yk + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6, -box, box)
        return y

    def ecg(self, X):
        arr = np.asarray(X)

        if arr.ndim == 1:
            return float(self.alpha @ arr)

        if arr.shape[0] == 4:
            return self.alpha @ arr

        return arr @ self.alpha

    def run(self, t, case=None, solver="Euler", plot=True):
        if case:
            self.set_case(case)

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

    def prc_xcorr(self, rhs, params, t_base, X_base, T, t_ref, var_index,
                  steps=120, amp_rel=0.02, dt_base=1e-4, dt_prc=4e-4,
                  window_cycles=1.2, guard_frac=0.08, box=3.0):
        """
        Compute phase response curve (PRC) using cross-correlation.
        """
        # Unpack parameters for rhs
        C, H, beta, gamma = self.params["C"], self.params["H"], self.params["beta"], self.params["gamma"]
        args = (C, H, beta, gamma)

        # Phases of perturbation (endpoint excluded)
        phases = np.linspace(0, 1, steps, endpoint=False)
        dphi = np.zeros_like(phases)

        # Scale kick to variable's natural variability
        eps = amp_rel * max(1e-10, np.std(X_base[:, var_index]))

        # Short local time grid for each perturbation window
        t_loc = np.arange(0, window_cycles * T, dt_prc)
        guard = int(guard_frac * T / dt_prc)  # Ignore initial fraction of window
        N = len(t_base)

        for i, phi in enumerate(phases):
            # Pick state on limit cycle that corresponds to phase phi
            t0 = t_ref - (1.0 - phi) * T
            j = int(np.floor((t0 - t_base[0]) / dt_base)) % N
            x0 = X_base[j].copy()

            # Unperturbed trajectory
            Y0 = self.rk4_boxed("healthy", x0, t_loc, box=box)

            # Perturb initial condition - instantaneous state jump on chosen variable
            xk = x0.copy()
            xk[var_index] += eps
            Y1 = self.rk4_boxed("healthy", xk, t_loc, box=box)

            # Compare after guard region
            seg0, seg1 = Y0[guard:, var_index], Y1[guard:, var_index]

            # Time lag between perturbed and unperturbed short traces
            lag = self.xcorr_shift(seg0, seg1, dt_prc)  # seconds

            # Convert to phase shift in cycles, subtract mean later for zero-centering
            dphi[i] = (-lag) / T  # dphi = (t unpert - t pert) / T

        return phases, dphi - dphi.mean()  # Remove tiny DC bias for cleaner, zero-centering

    def xcorr_shift(self, a, b, dt):
        """
        Compute cross-correlation lag (robust deltaT between two short traces).
        """
        # Remove DC & normalize energy to make correlation scale-free
        a = a - np.mean(a)
        b = b - np.mean(b)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0  # No variation in degenerate case

        r = np.correlate(a / na, b / nb, mode='full')  # Cross-correlation
        k0 = int(np.argmax(r))  # Index of best alignment

        # Parabolic refinement around discrete max
        if 0 < k0 < len(r) - 1:
            y1, y2, y3 = r[k0 - 1], r[k0], r[k0 + 1]
            den = y1 - 2 * y2 + y3
            if den != 0:
                k0 = k0 + 0.5 * (y1 - y3) / den

        N = len(a)
        return (k0 - (N - 1)) * dt  # Convert index to time lag

    def period_and_tref(self, sig, t, dt, max_bpm=300, smooth_sigma_s=0.008):
        """
        Estimate the period and reference peak time of a signal.

        Parameters:
        - sig: The signal array.
        - t: The time array corresponding to the signal.
        - dt: The time step.
        - max_bpm: Maximum beats per minute to consider.
        - smooth_sigma_s: Smoothing parameter for the signal.

        Returns:
        - T: Estimated period of the signal.
        - tref: Reference time of the last peak.
        - pk: Indices of detected peaks.
        """
        from scipy.signal import find_peaks
        from scipy.ndimage import gaussian_filter1d

        def try_find(s):
            min_dist = max(1, int((60.0 / max_bpm) / dt))
            std = float(np.nanstd(s))
            for sigma_s in [smooth_sigma_s, smooth_sigma_s / 2, 0.0]:
                s2 = s if sigma_s <= 0 else gaussian_filter1d(s, sigma=max(1, int(sigma_s / dt)))
                for frac in [0.4, 0.3, 0.2, 0.1, 0.05, 0.0]:
                    pk, _ = find_peaks(s2, prominence=frac * std, distance=min_dist)
                    if len(pk) >= 4:
                        return pk
            return np.array([], int)

        # Try direct peaks, then inverted signal if not enough peaks
        pk = try_find(sig)
        if len(pk) < 4:
            pk = try_find(-sig)
        if len(pk) >= 4:
            T = float(np.mean(np.diff(t[pk][-6:]))) if len(pk) >= 7 else float(np.mean(np.diff(t[pk])))
            return T, float(t[pk[-1]]), pk

        # Fallback to autocorrelation
        x = sig - np.mean(sig)
        ac = np.correlate(x, x, mode='full')[len(x) - 1:]
        k0 = int((60 / max_bpm) / dt)
        k = k0 + int(np.argmax(ac[k0:]))
        return k * dt, float(t[-1] - k * dt), np.arange(len(t) // 2, len(t), k)

    def shift_only(self, ph, dph, target_min=None, target_pos=None):
        """
        Circularly shift the phase response curve (PRC) to align specific features.

        Parameters:
        - ph: Phase array.
        - dph: Phase response array.
        - target_min: Target phase for the minimum value.
        - target_pos: Target phase for the positive peak.

        Returns:
        - ph: Shifted phase array.
        - dph: Shifted phase response array.
        """
        x = dph - np.mean(dph)  # Remove DC bias
        k = 0

        if target_min is not None:
            i_min = int(np.argmin(x))  # Index of minimum value
            k += int(round((target_min - ph[i_min]) * len(ph)))  # Convert desired shift to index

        if target_pos is not None:
            # After first shift, align positive peak
            x2 = np.roll(x, k)
            i_pos = int(np.argmax(x2))
            k += int(round((target_pos - ph[i_pos]) * len(ph)))

        x = np.roll(x, k)  # Circular shift
        return np.linspace(0, 1, len(ph), endpoint=False), x