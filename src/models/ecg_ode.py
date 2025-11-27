# Define the Winfree 4-variable ECG ODE
def ecg_rhs(t, y, C, H, beta, gamma):
    x1, x2, x3, x4 = y  # Unpack state vector, 2 coupled oscillators

    # RHS computation, overflow safe
    with np.errstate(over='ignore', invalid='ignore'):
        F = np.array([
            # Oscillator 1 - SA
            x1 - x2 - C*x1*x2 - x1*(x2**2),
            H*x1 - 3*x2 + C*x1*x2 + x1*(x2**2) + beta*(x4 - x2),
            # Oscillator 2 - AV
            x3 - x4 - C*x3*x4 - x3*(x4**2),
            H*x3 - 3*x4 + C*x3*x4 + x3*(x4**2) + 2*beta*(x2 - x4),
        ], float)
        F[~np.isfinite(F)] = 0.0  # Replace NaN/Inf with zero to prevent blowouts
    return gamma * F  # Global timescale factor gamma

# Define fixed-step integrators with value boxing
def rk4_boxed(rhs, y0, t, args, box=3.0):
    y = np.zeros((len(t), len(y0)))
    y[0] = np.clip(y0, -box, box)

    for k in range(len(t)-1):
        t0, dt = t[k], t[k+1]-t[k]
        yk = np.clip(y[k], -box, box)
        k1 = rhs(t0, yk, *args)
        k2 = rhs(t0+dt/2, np.clip(yk+0.5*dt*k1, -box, box), *args)
        k3 = rhs(t0+dt/2, np.clip(yk+0.5*dt*k2, -box, box), *args)
        k4 = rhs(t0+dt, np.clip(yk+dt*k3, -box, box), *args)
        y[k+1] = np.clip(yk + dt*(k1+2*k2+2*k3+k4)/6, -box, box)
    return y

def rk2_boxed(rhs, y0, t, args, box=3.0):
    y = np.zeros((len(t), len(y0)))
    y[0] = np.clip(y0, -box, box)

    for k in range(len(t)-1):
        dt = t[k+1]-t[k]
        yk = np.clip(y[k], -box, box)
        k1 = rhs(t[k], yk, *args)
        ym = np.clip(yk + 0.5*dt*k1, -box, box)
        k2 = rhs(t[k]+0.5*dt, ym, *args)
        y[k+1] = np.clip(yk + dt*k2, -box, box)
    return y