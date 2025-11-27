"""
Analysis Utilities

This file contains utility functions for analyzing ECG signals.
Includes functions for PRC calculations, cross-correlation, etc.
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def period_and_tref(sig, t, dt, max_bpm=300, smooth_sigma_s=0.008):
    """
    Estimate the period and reference peak time of a signal.

    Args:
        sig: Signal array.
        t: Time array.
        dt: Time step.
        max_bpm: Maximum beats per minute.
        smooth_sigma_s: Smoothing parameter.

    Returns:
        Tuple of (period, reference time, peaks).
    """
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

    pk = try_find(sig)
    if len(pk) < 4:
        pk = try_find(-sig)
    if len(pk) >= 4:
        T = float(np.mean(np.diff(t[pk][-6:]))) if len(pk) >= 7 else float(np.mean(np.diff(t[pk])))
        return T, float(t[pk[-1]]), pk

    x = sig - np.mean(sig)
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    k0 = int((60 / max_bpm) / dt)
    k = k0 + int(np.argmax(ac[k0:]))
    return k * dt, float(t[-1] - k * dt), np.arange(len(t) // 2, len(t), k)

def xcorr_shift(a, b, dt):
    a = a - np.mean(a)
    b = b - np.mean(b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    r = np.correlate(a/na, b/nb, mode='full')
    k0 = int(np.argmax(r))
    if 0 < k0 < len(r)-1:
        y1, y2, y3 = r[k0-1], r[k0], r[k0+1]
        den = y1 - 2*y2 + y3
        if den != 0: k0 = k0 + 0.5*(y1-y3)/den
    N = len(a)
    return (k0-(N-1))*dt

def prc_xcorr(rhs, params, t_base, X_base, T, t_ref, var_index,
              steps=120, amp_rel=0.02, dt_base=1e-4, dt_prc=4e-4,
              window_cycles=1.2, guard_frac=0.08, box=3.0):
    C, H, beta, gamma = params['C'], params['H'], params['beta'], params['gamma']
    args = (C, H, beta, gamma)
    phases = np.linspace(0, 1, steps, endpoint=False)
    dphi = np.zeros_like(phases)
    eps = amp_rel * max(1e-10, np.std(X_base[:, var_index]))

    t_loc = np.arange(0, window_cycles*T, dt_prc)
    guard = int(guard_frac*T/dt_prc)
    N = len(t_base)
    for i, phi in enumerate(phases):
        t0 = t_ref - (1.0 - phi)*T
        j = int(np.floor((t0 - t_base[0]) / dt_base)) % N
        x0 = X_base[j].copy()

        Y0 = rk2_boxed(rhs, x0, t_loc, args, box=box)
        xk = x0.copy()
        xk[var_index] += eps
        Y1 = rk2_boxed(rhs, xk, t_loc, args, box=box)

        seg0, seg1 = Y0[guard:, var_index], Y1[guard:, var_index]
        lag = xcorr_shift(seg0, seg1, dt_prc)
        dphi[i] = (-lag)/T
    return phases, dphi - dphi.mean()

def shift_only(ph, dph, target_min=None, target_pos=None):
    x = dph - np.mean(dph)
    k = 0
    if target_min is not None:
        i_min = int(np.argmin(x))
        k += int(round((target_min - ph[i_min]) * len(ph)))
    if target_pos is not None:
        x2 = np.roll(x, k)
        i_pos = int(np.argmax(x2))
        k += int(round((target_pos - ph[i_pos]) * len(ph)))
    x = np.roll(x, k)
    return np.linspace(0, 1, len(ph), endpoint=False), x

def rk2_boxed(rhs, y0, t, args, box=3.0):
    """
    Midpoint method (RK2) integrator with value boxing.

    Args:
        rhs: Right-hand side function of the ODE.
        y0: Initial state vector.
        t: Time array.
        args: Additional arguments for the RHS function.
        box: Value boxing limit.

    Returns:
        Array of integrated states over time.
    """
    y = np.zeros((len(t), len(y0)))
    y[0] = np.clip(y0, -box, box)

    for k in range(len(t)-1):
        dt = t[k+1] - t[k]
        yk = np.clip(y[k], -box, box)
        k1 = rhs(t[k], yk, *args)
        ym = np.clip(yk + 0.5 * dt * k1, -box, box)
        k2 = rhs(t[k] + 0.5 * dt, ym, *args)
        y[k+1] = np.clip(yk + dt * k2, -box, box)

    return y