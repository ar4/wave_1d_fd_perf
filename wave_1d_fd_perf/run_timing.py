"""Measure the runtime of the propagators."""
from timeit import repeat
import numpy as np
import pandas as pd
from wave_1d_fd_perf.propagators import (VPy1, VPy2, VNumba1, VNumba2, VNumba3, VNumba4,
                             VFortran1, VFortran2, VFortran3,
                             VC1_gcc, VC2_gcc_O2, VC2_gcc_O3, VC2_gcc_Ofast,
                             VC2_gcc_Ofast_autopar, VC3_gcc, VC4_gcc,
                             VC5_gcc_O2, VC5_gcc_O3, VC5_gcc_Ofast)
from wave_1d_fd_perf.test_wave_1d_fd_perf import ricker

def run_timing_num_steps():
    """Time implementations as num_steps varies."""

    num_repeat = 2

    num_steps = range(0, 50, 25)
    model_size = 1000
    versions = _versions()

    times = pd.DataFrame(columns=['version', 'num_steps', 'model_size', 'time'])

    for nsteps in num_steps:
        model = _make_model(model_size, nsteps)
        times = _time_versions(versions, model, num_repeat, times)

    return times


def run_timing_model_size():
    """Time implementations as model size varies."""

    num_repeat = 2

    num_steps = 10
    model_sizes = range(100, 1100, 100)
    versions = _versions()

    times = pd.DataFrame(columns=['version', 'num_steps', 'model_size', 'time'])

    for N in model_sizes:
        model = _make_model(N, num_steps)
        times = _time_versions(versions, model, num_repeat, times)

    return times


def _versions():
    """Return a list of versions to be timed."""
    return [{'class': VPy1, 'name': 'Python v1'},
            {'class': VPy2, 'name': 'Python v2'},
            {'class': VNumba1, 'name': 'Numba v1'},
            {'class': VNumba2, 'name': 'Numba v2'},
            {'class': VNumba3, 'name': 'Numba v3'},
            {'class': VNumba4, 'name': 'Numba v4'},
            {'class': VFortran1, 'name': 'Fortran v1'},
            {'class': VFortran2, 'name': 'Fortran v2'},
            {'class': VFortran3, 'name': 'Fortran v3'},
            {'class': VC1_gcc, 'name': 'C v1 (gcc, -O3)'},
            {'class': VC2_gcc_O2, 'name': 'C v2 (gcc, -O2)'},
            {'class': VC2_gcc_O3, 'name': 'C v2 (gcc, -O3)'},
            {'class': VC2_gcc_Ofast, 'name': 'C v2 (gcc, -Ofast)'},
            {'class': VC2_gcc_Ofast_autopar, 'name': 'C v2 (gcc, -Ofast, autoparallel)'},
            {'class': VC3_gcc, 'name': 'C v3 (gcc, -O3)'},
            {'class': VC4_gcc, 'name': 'C v4 (gcc, -O3)'},
            {'class': VC5_gcc_O2, 'name': 'C v5 (gcc, -O2)'},
            {'class': VC5_gcc_O3, 'name': 'C v5 (gcc, -O3)'},
            {'class': VC5_gcc_Ofast, 'name': 'C v5 (gcc, -Ofast)'}]


def _make_model(N, nsteps):
    """Create a model with a given number of elements and time steps."""
    model = np.random.random(N).astype(np.float32) * 3000 + 1500
    max_vel = 4500
    dx = 5
    dt = 0.6 * dx / max_vel
    source = ricker(25, nsteps, dt, 0.05)
    sx = int(N/2)
    return {'model': model, 'dx': dx, 'dt': dt, 'nsteps': nsteps,
            'sources': np.array([source]), 'sx': np.array([sx])}


def _time_versions(versions, model, num_repeat, dataframe):
    """Loop over versions and append the timing results to the dataframe."""
    num_steps = model['nsteps']
    model_size = len(model['model'])
    for v in versions:
        time = _time_version(v['class'], model, num_repeat)
        dataframe = dataframe.append({'version': v['name'],
                                      'num_steps': num_steps,
                                      'model_size': model_size,
                                      'time': time}, ignore_index=True)
    return dataframe


def _time_version(version, model, num_repeat):
    """Time a particular version."""
    v = version(model['model'], model['dx'], model['dt'])

    def closure():
        """Closure over variables so they can be used in repeat below."""
        v.step(model['nsteps'], model['sources'], model['sx'])
 
    return np.min(repeat(closure, number=1, repeat=num_repeat))

if __name__ == '__main__':
    print(run_timing_num_steps())
