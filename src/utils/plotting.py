"""
Plotting Utilities

This file contains utility functions for generating plots.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_ecg_signal(t, ecg_signal, title="ECG Signal"):
    """
    Plot an ECG signal over time.

    Args:
        t: Time array.
        ecg_signal: ECG signal array.
        title: Title of the plot.
    """
    plt.figure(figsize=(10, 3))
    plt.plot(t, ecg_signal)
    plt.title(title)
    plt.xlabel("Time (sec)")
    plt.ylabel("ECG")
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_prc(ph2a, dph2a, ph4a, dph4a):
    plt.figure(figsize=(7.6, 4.2))
    plt.plot(ph2a, dph2a, '-o', ms=3, lw=1.8, label='With respect to x₂')
    plt.plot(ph4a, dph4a, '-o', ms=3, lw=1.8, label='With respect to x₄')
    plt.axhline(0, color='k', lw=0.8, alpha=0.6)
    plt.xlim(0, 1)
    plt.xlabel('Phase')
    plt.ylabel('Phase shift (cycles)')
    plt.title('Winfree PRC (Healthy Heart)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_phase_advance(ph2a, advance_ms):
    plt.figure(figsize=(6.5, 4))
    plt.plot(ph2a, advance_ms, '-o', lw=1.8, ms=3, color='deeppink')
    plt.axhline(0, color='k', lw=0.8, alpha=0.6)
    plt.xlabel('Normalized membrane potential (phase)')
    plt.ylabel('Phase advance (ms)')
    plt.title('SA Node Phase Response Curve (Model vs. Fig 4a Convention)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()