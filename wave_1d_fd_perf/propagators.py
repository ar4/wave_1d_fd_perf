"""Propagate a 1D wavefield using different implementations of an 8th order
finite difference method so that runtimes can be compared.
"""
import os
import concurrent.futures
from ctypes import c_int, c_float
import numpy as np
from numba import jit
import wave_1d_fd_perf
from wave_1d_fd_perf import vfortran1
from wave_1d_fd_perf import vfortran2
from wave_1d_fd_perf import vfortran3

class Propagator(object):
    """An 8th order finite difference propagator for the 1D wave equation."""
    def __init__(self, model, dx, dt=None):
        self.nx = len(model)
        self.dx = np.float32(dx)
        max_vel = np.max(model)
        if dt:
            self.dt = dt
        else:
            self.dt = 0.6 * self.dx / max_vel
        self.nx_padded = self.nx + 2*8
        self.model_padded = np.pad(model, (8, 8), 'edge')
        self.model_padded2_dt2 = self.model_padded**2 * self.dt**2
        self.wavefield = [np.zeros(self.nx_padded, np.float32),
                          np.zeros(self.nx_padded, np.float32)
                         ]
        self.current_wavefield = self.wavefield[0]
        self.previous_wavefield = self.wavefield[1]

class VPy1(Propagator):
    """A simple Python implementation."""
    def step(self, num_steps, sources=None, sources_x=None):
        """Propagate wavefield."""
        for step in range(num_steps):
            f = self.current_wavefield
            fp = self.previous_wavefield

            for x in range(8, self.nx_padded-8):
                f_xx = (-735*f[x-8]+15360*f[x-7]
                        -156800*f[x-6]+1053696*f[x-5]
                        -5350800*f[x-4]+22830080*f[x-3]
                        -94174080*f[x-2]+538137600*f[x-1]
                        -924708642*f[x+0]
                        +538137600*f[x+1]-94174080*f[x+2]
                        +22830080*f[x+3]-5350800*f[x+4]
                        +1053696*f[x+5]-156800*f[x+6]
                        +15360*f[x+7]-735*f[x+8])/(302702400*self.dx**2)
                fp[x] = (self.model_padded[x]**2 * self.dt**2 * f_xx
                         + 2*f[x] - fp[x])

            for i in range(sources.shape[0]):
                sx = sources_x[i] + 8
                source_amp = sources[i, step]
                fp[sx] += (self.model_padded[sx]**2 * self.dt**2 * source_amp)

            self.current_wavefield = fp
            self.previous_wavefield = f

        return self.current_wavefield[8:self.nx_padded-8]


class VPy2(Propagator):
    """Like VPy1, but using model_padded2_dt2."""
    def step(self, num_steps, sources=None, sources_x=None):
        """Propagate wavefield."""
        for step in range(num_steps):
            f = self.current_wavefield
            fp = self.previous_wavefield

            for x in range(8, self.nx_padded-8):
                f_xx = (-735*f[x-8]+15360*f[x-7]
                        -156800*f[x-6]+1053696*f[x-5]
                        -5350800*f[x-4]+22830080*f[x-3]
                        -94174080*f[x-2]+538137600*f[x-1]
                        -924708642*f[x+0]
                        +538137600*f[x+1]-94174080*f[x+2]
                        +22830080*f[x+3]-5350800*f[x+4]
                        +1053696*f[x+5]-156800*f[x+6]
                        +15360*f[x+7]-735*f[x+8])/(302702400*self.dx**2)
                fp[x] = (self.model_padded2_dt2[x] * f_xx
                         + 2*f[x] - fp[x])

            for i in range(sources.shape[0]):
                sx = sources_x[i] + 8
                source_amp = sources[i, step]
                fp[sx] += (self.model_padded2_dt2[sx] * source_amp)

            self.current_wavefield = fp
            self.previous_wavefield = f

        return self.current_wavefield[8:self.nx_padded-8]


class VNumba1(Propagator):
    """A Numba implementation of inner loop."""
    def step(self, num_steps, sources=None, sources_x=None):
        """Propagate wavefield."""

        @jit(nopython=True)
        def _numba_step(f, fp, nx_padded, model_padded, dt, dx, sources,
                        sources_x):
            for x in range(8, nx_padded-8):
                f_xx = (-735*f[x-8]+15360*f[x-7]
                        -156800*f[x-6]+1053696*f[x-5]
                        -5350800*f[x-4]+22830080*f[x-3]
                        -94174080*f[x-2]+538137600*f[x-1]
                        -924708642*f[x+0]
                        +538137600*f[x+1]-94174080*f[x+2]
                        +22830080*f[x+3]-5350800*f[x+4]
                        +1053696*f[x+5]-156800*f[x+6]
                        +15360*f[x+7]-735*f[x+8])/(302702400*dx**2)
                fp[x] = (model_padded[x]**2 * dt**2 * f_xx
                         + 2*f[x] - fp[x])

            for i in range(sources.shape[0]):
                sx = sources_x[i] + 8
                fp[sx] += (model_padded[sx]**2 * dt**2 * sources[i])

        for step in range(num_steps):

            _numba_step(self.current_wavefield, self.previous_wavefield,
                        self.nx_padded, self.model_padded, self.dt,
                        self.dx, sources[:, step], sources_x)

            tmp = self.current_wavefield
            self.current_wavefield = self.previous_wavefield
            self.previous_wavefield = tmp

        return self.current_wavefield[8:self.nx_padded-8]


class VNumba2(Propagator):
    """A Numba implementation of outer loop."""
    def step(self, num_steps, sources=None, sources_x=None):
        """Propagate wavefield."""

        @jit(nopython=True)
        def _numba_step(f, fp, nx_padded, model_padded, dt, dx, sources,
                        sources_x, num_steps):

            for step in range(num_steps):
                for x in range(8, nx_padded-8):
                    f_xx = (-735*f[x-8]+15360*f[x-7]
                            -156800*f[x-6]+1053696*f[x-5]
                            -5350800*f[x-4]+22830080*f[x-3]
                            -94174080*f[x-2]+538137600*f[x-1]
                            -924708642*f[x+0]
                            +538137600*f[x+1]-94174080*f[x+2]
                            +22830080*f[x+3]-5350800*f[x+4]
                            +1053696*f[x+5]-156800*f[x+6]
                            +15360*f[x+7]-735*f[x+8])/(302702400*dx**2)
                    fp[x] = (model_padded[x]**2 * dt**2 * f_xx
                             + 2*f[x] - fp[x])

                for i in range(sources.shape[0]):
                    sx = sources_x[i] + 8
                    fp[sx] += (model_padded[sx]**2 * dt**2 * sources[i, step])

                tmp = f
                f = fp
                fp = tmp

            return f, fp

        self.current_wavefield, self.previous_wavefield = \
                _numba_step(self.current_wavefield, self.previous_wavefield,
                            self.nx_padded, self.model_padded, self.dt,
                            self.dx, sources, sources_x, num_steps)

        return self.current_wavefield[8:self.nx_padded-8]


class VNumba3(Propagator):
    """A parallel Numba implementation."""
    def step(self, num_steps, sources=None, sources_x=None):
        """Propagate wavefield."""

        @jit(nopython=True, nogil=True)
        def _numba_inner(f, fp, model_padded, dt, dx, start, end):
            for x in range(start, end):
                f_xx = (-735*f[x-8]+15360*f[x-7]
                        -156800*f[x-6]+1053696*f[x-5]
                        -5350800*f[x-4]+22830080*f[x-3]
                        -94174080*f[x-2]+538137600*f[x-1]
                        -924708642*f[x+0]
                        +538137600*f[x+1]-94174080*f[x+2]
                        +22830080*f[x+3]-5350800*f[x+4]
                        +1053696*f[x+5]-156800*f[x+6]
                        +15360*f[x+7]-735*f[x+8])/(302702400*dx**2)
                fp[x] = (model_padded[x]**2 * dt**2 * f_xx
                         + 2*f[x] - fp[x])

        def _numba_step(f, fp, model_padded, dt, dx, sources,
                        sources_x, num_steps, num_chunks=4):

            chunk_len = int(np.ceil((f.shape[0]-16) / num_chunks))
            chunks = np.zeros(num_chunks+1, np.int)
            chunks[:-1] = np.arange(8, f.shape[0]-8, chunk_len)
            chunks[-1] = f.shape[0]-8
            executor = concurrent.futures.ThreadPoolExecutor()

            for step in range(num_steps):

                fs = []

                for chunk_idx, chunk in enumerate(chunks[:-1]):
                    fs.append(executor.submit(_numba_inner, f, fp,
                                              model_padded, dt, dx,
                                              chunk, chunks[chunk_idx+1]))
                concurrent.futures.wait(fs)

                for i in range(sources.shape[0]):
                    sx = sources_x[i] + 8
                    fp[sx] += (model_padded[sx]**2 * dt**2 * sources[i, step])

                tmp = f
                f = fp
                fp = tmp

            return f, fp

        self.current_wavefield, self.previous_wavefield = \
                _numba_step(self.current_wavefield, self.previous_wavefield,
                            self.model_padded, self.dt,
                            self.dx, sources, sources_x, num_steps)

        return self.current_wavefield[8:self.nx_padded-8]


class VNumba4(Propagator):
    """Same as VNumba3, but using model_padded2_dt2."""
    def step(self, num_steps, sources=None, sources_x=None):
        """Propagate wavefield."""

        @jit(nopython=True, nogil=True)
        def _numba_inner(f, fp, model_padded2_dt2, dx, start, end):
            for x in range(start, end):
                f_xx = (-735*f[x-8]+15360*f[x-7]
                        -156800*f[x-6]+1053696*f[x-5]
                        -5350800*f[x-4]+22830080*f[x-3]
                        -94174080*f[x-2]+538137600*f[x-1]
                        -924708642*f[x+0]
                        +538137600*f[x+1]-94174080*f[x+2]
                        +22830080*f[x+3]-5350800*f[x+4]
                        +1053696*f[x+5]-156800*f[x+6]
                        +15360*f[x+7]-735*f[x+8])/(302702400*dx**2)
                fp[x] = (model_padded2_dt2[x] * f_xx
                         + 2*f[x] - fp[x])

        def _numba_step(f, fp, model_padded2_dt2, dx, sources,
                        sources_x, num_steps, num_chunks=4):

            chunk_len = int(np.ceil((f.shape[0]-16) / num_chunks))
            chunks = np.zeros(num_chunks+1, np.int)
            chunks[:-1] = np.arange(8, f.shape[0]-8, chunk_len)
            chunks[-1] = f.shape[0]-8
            executor = concurrent.futures.ThreadPoolExecutor()

            for step in range(num_steps):

                fs = []

                for chunk_idx, chunk in enumerate(chunks[:-1]):
                    fs.append(executor.submit(_numba_inner, f, fp,
                                              model_padded2_dt2, dx,
                                              chunk, chunks[chunk_idx+1]))
                concurrent.futures.wait(fs)

                for i in range(sources.shape[0]):
                    sx = sources_x[i] + 8
                    fp[sx] += (model_padded2_dt2[sx] * sources[i, step])

                tmp = f
                f = fp
                fp = tmp

            return f, fp

        self.current_wavefield, self.previous_wavefield = \
                _numba_step(self.current_wavefield, self.previous_wavefield,
                            self.model_padded2_dt2,
                            self.dx, sources, sources_x, num_steps)

        return self.current_wavefield[8:self.nx_padded-8]


class VFortran1(Propagator):
    """A Fortran implementation."""
    def step(self, num_steps, sources=None, sources_x=None):
        """Propagate wavefield."""

        num_sources = sources.shape[0]
        source_len = sources.shape[1]
        vfortran1.vfortran1.step(self.current_wavefield,
                                 self.previous_wavefield,
                                 self.model_padded, self.dt, self.dx,
                                 sources, sources_x, num_steps,
                                 self.nx_padded, num_sources, source_len)

        if num_steps%2 != 0:
            tmp = self.current_wavefield
            self.current_wavefield = self.previous_wavefield
            self.previous_wavefield = tmp

        return self.current_wavefield[8:self.nx_padded-8]


class VFortran2(Propagator):
    """Like VFortran1, but using OpenMP on inner loop."""
    def step(self, num_steps, sources=None, sources_x=None):
        """Propagate wavefield."""

        num_sources = sources.shape[0]
        source_len = sources.shape[1]
        vfortran2.vfortran2.step(self.current_wavefield,
                                 self.previous_wavefield,
                                 self.model_padded, self.dt, self.dx,
                                 sources, sources_x, num_steps,
                                 self.nx_padded, num_sources, source_len)

        if num_steps%2 != 0:
            tmp = self.current_wavefield
            self.current_wavefield = self.previous_wavefield
            self.previous_wavefield = tmp

        return self.current_wavefield[8:self.nx_padded-8]


class VFortran3(Propagator):
    """Like VFortran2, but with model_padded2_dt2."""
    def step(self, num_steps, sources=None, sources_x=None):
        """Propagate wavefield."""

        num_sources = sources.shape[0]
        source_len = sources.shape[1]
        vfortran3.vfortran3.step(self.current_wavefield,
                                 self.previous_wavefield,
                                 self.model_padded2_dt2, self.dx,
                                 sources, sources_x, num_steps,
                                 self.nx_padded, num_sources, source_len)

        if num_steps%2 != 0:
            tmp = self.current_wavefield
            self.current_wavefield = self.previous_wavefield
            self.previous_wavefield = tmp

        return self.current_wavefield[8:self.nx_padded-8]


class VC1_gcc(Propagator):
    """A C implementation."""
    def __init__(self, model, dx, dt=None):
        super(VC1_gcc, self).__init__(model, dx, dt)

        print(wave_1d_fd_perf.__path__[0])
        print(os.listdir(wave_1d_fd_perf.__path__[0]))
        self._libvc1 = np.ctypeslib.load_library('libvc1_gcc', wave_1d_fd_perf.__path__[0])
        self._libvc1.step.argtypes = \
                [np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS', 'WRITEABLE')),
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS', 'WRITEABLE')),
                 c_int,
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS')),
                 c_float, c_float,
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=2,
                                        flags=('C_CONTIGUOUS')),
                 np.ctypeslib.ndpointer(dtype=np.int, ndim=1,
                                        flags=('C_CONTIGUOUS')),
                 c_int, c_int, c_int]

    def step(self, num_steps, sources=None, sources_x=None):
        """Propagate wavefield."""

        num_sources = sources.shape[0]
        source_len = sources.shape[1]
        self._libvc1.step(self.current_wavefield, self.previous_wavefield,
                          self.nx_padded, self.model_padded, self.dt, self.dx,
                          sources, sources_x, num_sources, source_len,
                          num_steps)

        if num_steps%2 != 0:
            tmp = self.current_wavefield
            self.current_wavefield = self.previous_wavefield
            self.previous_wavefield = tmp

        return self.current_wavefield[8:self.nx_padded-8]


class VC2_gcc_O2(Propagator):
    """Same as VC1, but using model_padded2_dt2."""
    def __init__(self, model, dx, dt=None):
        super(VC2_gcc_O2, self).__init__(model, dx, dt)
        self._libvc1 = np.ctypeslib.load_library('libvc2_gcc_O2', wave_1d_fd_perf.__path__[0])
        self._libvc1.step.argtypes = \
                [np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS', 'WRITEABLE')),
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS', 'WRITEABLE')),
                 c_int,
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS')),
                 c_float,
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=2,
                                        flags=('C_CONTIGUOUS')),
                 np.ctypeslib.ndpointer(dtype=np.int, ndim=1,
                                        flags=('C_CONTIGUOUS')),
                 c_int, c_int, c_int]

    def step(self, num_steps, sources=None, sources_x=None):
        """Propagate wavefield."""

        num_sources = sources.shape[0]
        source_len = sources.shape[1]
        self._libvc1.step(self.current_wavefield, self.previous_wavefield,
                          self.nx_padded, self.model_padded2_dt2, self.dx,
                          sources, sources_x, num_sources, source_len,
                          num_steps)

        if num_steps%2 != 0:
            tmp = self.current_wavefield
            self.current_wavefield = self.previous_wavefield
            self.previous_wavefield = tmp

        return self.current_wavefield[8:self.nx_padded-8]


class VC2_gcc_O3(Propagator):
    """Same as VC2, but using -O3."""
    def __init__(self, model, dx, dt=None):
        super(VC2_gcc_O3, self).__init__(model, dx, dt)
        self._libvc1 = np.ctypeslib.load_library('libvc2_gcc_O3', wave_1d_fd_perf.__path__[0])
        self._libvc1.step.argtypes = \
                [np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS', 'WRITEABLE')),
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS', 'WRITEABLE')),
                 c_int,
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS')),
                 c_float,
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=2,
                                        flags=('C_CONTIGUOUS')),
                 np.ctypeslib.ndpointer(dtype=np.int, ndim=1,
                                        flags=('C_CONTIGUOUS')),
                 c_int, c_int, c_int]

    def step(self, num_steps, sources=None, sources_x=None):
        """Propagate wavefield."""

        num_sources = sources.shape[0]
        source_len = sources.shape[1]
        self._libvc1.step(self.current_wavefield, self.previous_wavefield,
                          self.nx_padded, self.model_padded2_dt2, self.dx,
                          sources, sources_x, num_sources, source_len,
                          num_steps)

        if num_steps%2 != 0:
            tmp = self.current_wavefield
            self.current_wavefield = self.previous_wavefield
            self.previous_wavefield = tmp

        return self.current_wavefield[8:self.nx_padded-8]


class VC2_gcc_Ofast(Propagator):
    """Same as VC2, but using -Ofast."""
    def __init__(self, model, dx, dt=None):
        super(VC2_gcc_Ofast, self).__init__(model, dx, dt)
        self._libvc1 = np.ctypeslib.load_library('libvc2_gcc_Ofast', wave_1d_fd_perf.__path__[0])
        self._libvc1.step.argtypes = \
                [np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS', 'WRITEABLE')),
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS', 'WRITEABLE')),
                 c_int,
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS')),
                 c_float,
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=2,
                                        flags=('C_CONTIGUOUS')),
                 np.ctypeslib.ndpointer(dtype=np.int, ndim=1,
                                        flags=('C_CONTIGUOUS')),
                 c_int, c_int, c_int]

    def step(self, num_steps, sources=None, sources_x=None):
        """Propagate wavefield."""

        num_sources = sources.shape[0]
        source_len = sources.shape[1]
        self._libvc1.step(self.current_wavefield, self.previous_wavefield,
                          self.nx_padded, self.model_padded2_dt2, self.dx,
                          sources, sources_x, num_sources, source_len,
                          num_steps)

        if num_steps%2 != 0:
            tmp = self.current_wavefield
            self.current_wavefield = self.previous_wavefield
            self.previous_wavefield = tmp

        return self.current_wavefield[8:self.nx_padded-8]


class VC2_gcc_Ofast_autopar(Propagator):
    """Same as VC2, but using -Ofast and autoparallelization."""
    def __init__(self, model, dx, dt=None):
        super(VC2_gcc_Ofast_autopar, self).__init__(model, dx, dt)
        self._libvc1 = np.ctypeslib.load_library('libvc2_gcc_Ofast_autopar',
                                                 wave_1d_fd_perf.__path__[0])
        self._libvc1.step.argtypes = \
                [np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS', 'WRITEABLE')),
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS', 'WRITEABLE')),
                 c_int,
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS')),
                 c_float,
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=2,
                                        flags=('C_CONTIGUOUS')),
                 np.ctypeslib.ndpointer(dtype=np.int, ndim=1,
                                        flags=('C_CONTIGUOUS')),
                 c_int, c_int, c_int]

    def step(self, num_steps, sources=None, sources_x=None):
        """Propagate wavefield."""

        num_sources = sources.shape[0]
        source_len = sources.shape[1]
        self._libvc1.step(self.current_wavefield, self.previous_wavefield,
                          self.nx_padded, self.model_padded2_dt2, self.dx,
                          sources, sources_x, num_sources, source_len,
                          num_steps)

        if num_steps%2 != 0:
            tmp = self.current_wavefield
            self.current_wavefield = self.previous_wavefield
            self.previous_wavefield = tmp

        return self.current_wavefield[8:self.nx_padded-8]


class VC3_gcc(Propagator):
    """A C implementation with OpenMP of inner loop."""
    def __init__(self, model, dx, dt=None):
        super(VC3_gcc, self).__init__(model, dx, dt)
        self._libvc2 = np.ctypeslib.load_library('libvc3_gcc', wave_1d_fd_perf.__path__[0])
        self._libvc2.step.argtypes = \
                [np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS', 'WRITEABLE')),
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS', 'WRITEABLE')),
                 c_int,
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS')),
                 c_float, c_float,
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=2,
                                        flags=('C_CONTIGUOUS')),
                 np.ctypeslib.ndpointer(dtype=np.int, ndim=1,
                                        flags=('C_CONTIGUOUS')),
                 c_int, c_int, c_int]

    def step(self, num_steps, sources=None, sources_x=None):
        """Propagate wavefield."""

        num_sources = sources.shape[0]
        source_len = sources.shape[1]
        self._libvc2.step(self.current_wavefield, self.previous_wavefield,
                          self.nx_padded, self.model_padded, self.dt, self.dx,
                          sources, sources_x, num_sources, source_len,
                          num_steps)

        if num_steps%2 != 0:
            tmp = self.current_wavefield
            self.current_wavefield = self.previous_wavefield
            self.previous_wavefield = tmp

        return self.current_wavefield[8:self.nx_padded-8]


class VC4_gcc(Propagator):
    """A C implementation with OpenMP of outer loop."""
    def __init__(self, model, dx, dt=None):
        super(VC4_gcc, self).__init__(model, dx, dt)
        self._libvc3 = np.ctypeslib.load_library('libvc4_gcc', wave_1d_fd_perf.__path__[0])
        self._libvc3.step.argtypes = \
                [np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS', 'WRITEABLE')),
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS', 'WRITEABLE')),
                 c_int,
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS')),
                 c_float, c_float,
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=2,
                                        flags=('C_CONTIGUOUS')),
                 np.ctypeslib.ndpointer(dtype=np.int, ndim=1,
                                        flags=('C_CONTIGUOUS')),
                 c_int, c_int, c_int]

    def step(self, num_steps, sources=None, sources_x=None):
        """Propagate wavefield."""

        num_sources = sources.shape[0]
        source_len = sources.shape[1]
        self._libvc3.step(self.current_wavefield, self.previous_wavefield,
                          self.nx_padded, self.model_padded, self.dt, self.dx,
                          sources, sources_x, num_sources, source_len,
                          num_steps)

        if num_steps%2 != 0:
            tmp = self.current_wavefield
            self.current_wavefield = self.previous_wavefield
            self.previous_wavefield = tmp

        return self.current_wavefield[8:self.nx_padded-8]


class VC5_gcc_O2(Propagator):
    """Same as VC4 but using model_padded2_dt2."""
    def __init__(self, model, dx, dt=None):
        super(VC5_gcc_O2, self).__init__(model, dx, dt)
        self._libvc3 = np.ctypeslib.load_library('libvc5_gcc_O2', wave_1d_fd_perf.__path__[0])
        self._libvc3.step.argtypes = \
                [np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS', 'WRITEABLE')),
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS', 'WRITEABLE')),
                 c_int,
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS')),
                 c_float,
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=2,
                                        flags=('C_CONTIGUOUS')),
                 np.ctypeslib.ndpointer(dtype=np.int, ndim=1,
                                        flags=('C_CONTIGUOUS')),
                 c_int, c_int, c_int]

    def step(self, num_steps, sources=None, sources_x=None):
        """Propagate wavefield."""

        num_sources = sources.shape[0]
        source_len = sources.shape[1]
        self._libvc3.step(self.current_wavefield, self.previous_wavefield,
                          self.nx_padded, self.model_padded2_dt2, self.dx,
                          sources, sources_x, num_sources, source_len,
                          num_steps)

        if num_steps%2 != 0:
            tmp = self.current_wavefield
            self.current_wavefield = self.previous_wavefield
            self.previous_wavefield = tmp

        return self.current_wavefield[8:self.nx_padded-8]


class VC5_gcc_O3(Propagator):
    """Same as VC5 but using -O3."""
    def __init__(self, model, dx, dt=None):
        super(VC5_gcc_O3, self).__init__(model, dx, dt)
        self._libvc3 = np.ctypeslib.load_library('libvc5_gcc_O3', wave_1d_fd_perf.__path__[0])
        self._libvc3.step.argtypes = \
                [np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS', 'WRITEABLE')),
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS', 'WRITEABLE')),
                 c_int,
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS')),
                 c_float,
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=2,
                                        flags=('C_CONTIGUOUS')),
                 np.ctypeslib.ndpointer(dtype=np.int, ndim=1,
                                        flags=('C_CONTIGUOUS')),
                 c_int, c_int, c_int]

    def step(self, num_steps, sources=None, sources_x=None):
        """Propagate wavefield."""

        num_sources = sources.shape[0]
        source_len = sources.shape[1]
        self._libvc3.step(self.current_wavefield, self.previous_wavefield,
                          self.nx_padded, self.model_padded2_dt2, self.dx,
                          sources, sources_x, num_sources, source_len,
                          num_steps)

        if num_steps%2 != 0:
            tmp = self.current_wavefield
            self.current_wavefield = self.previous_wavefield
            self.previous_wavefield = tmp

        return self.current_wavefield[8:self.nx_padded-8]


class VC5_gcc_Ofast(Propagator):
    """Same as VC5 but using -Ofast."""
    def __init__(self, model, dx, dt=None):
        super(VC5_gcc_Ofast, self).__init__(model, dx, dt)
        self._libvc3 = np.ctypeslib.load_library('libvc5_gcc_Ofast', wave_1d_fd_perf.__path__[0])
        self._libvc3.step.argtypes = \
                [np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS', 'WRITEABLE')),
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS', 'WRITEABLE')),
                 c_int,
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                        shape=(self.nx_padded),
                                        flags=('C_CONTIGUOUS')),
                 c_float,
                 np.ctypeslib.ndpointer(dtype=np.float32, ndim=2,
                                        flags=('C_CONTIGUOUS')),
                 np.ctypeslib.ndpointer(dtype=np.int, ndim=1,
                                        flags=('C_CONTIGUOUS')),
                 c_int, c_int, c_int]

    def step(self, num_steps, sources=None, sources_x=None):
        """Propagate wavefield."""

        num_sources = sources.shape[0]
        source_len = sources.shape[1]
        self._libvc3.step(self.current_wavefield, self.previous_wavefield,
                          self.nx_padded, self.model_padded2_dt2, self.dx,
                          sources, sources_x, num_sources, source_len,
                          num_steps)

        if num_steps%2 != 0:
            tmp = self.current_wavefield
            self.current_wavefield = self.previous_wavefield
            self.previous_wavefield = tmp

        return self.current_wavefield[8:self.nx_padded-8]

# loopy
# C: gcc, clang
# pylint
