from setuptools import setup, Extension
import numpy.distutils.core

vfortran1 = numpy.distutils.core.Extension(name='wave_1d_fd_perf.vfortran1', sources=['wave_1d_fd_perf/vfortran1.f90'])
vfortran2 = numpy.distutils.core.Extension(name='wave_1d_fd_perf.vfortran2', sources=['wave_1d_fd_perf/vfortran2.f90'], extra_compile_args=['-fopenmp'])
vfortran3 = numpy.distutils.core.Extension(name='wave_1d_fd_perf.vfortran3', sources=['wave_1d_fd_perf/vfortran3.f90'], extra_compile_args=['-fopenmp'])

libvc1_gcc = Extension(name='wave_1d_fd_perf.libvc1_gcc', sources=['wave_1d_fd_perf/vc1.c'], extra_compile_args=['-march=native', '-O3', '-std=c99'])

libvc2_gcc_O2 = Extension(name='wave_1d_fd_perf.libvc2_gcc_O2', sources=['wave_1d_fd_perf/vc2.c'], extra_compile_args=['-march=native', '-O2', '-std=c99'])

libvc2_gcc_O3 = Extension(name='wave_1d_fd_perf.libvc2_gcc_O3', sources=['wave_1d_fd_perf/vc2.c'], extra_compile_args=['-march=native', '-O3', '-std=c99'])
	
libvc2_gcc_Ofast = Extension(name='wave_1d_fd_perf.libvc2_gcc_Ofast', sources=['wave_1d_fd_perf/vc2.c'], extra_compile_args=['-march=native', '-Ofast', '-std=c99'])

libvc2_gcc_Ofast_autopar = Extension(name='wave_1d_fd_perf.libvc2_gcc_Ofast_autopar', sources=['wave_1d_fd_perf/vc2.c'], extra_compile_args=['-march=native', '-Ofast', '-floop-parallelize-all', '-ftree-parallelize-loops=4', '-std=c99'])
	
libvc3_gcc = Extension(name='wave_1d_fd_perf.libvc3_gcc', sources=['wave_1d_fd_perf/vc3.c'], extra_compile_args=['-march=native', '-O3', '-fopenmp', '-std=c99'])

libvc4_gcc = Extension(name='wave_1d_fd_perf.libvc4_gcc', sources=['wave_1d_fd_perf/vc4.c'], extra_compile_args=['-march=native', '-O3', '-fopenmp', '-std=c99'])

libvc5_gcc_O2 = Extension(name='wave_1d_fd_perf.libvc5_gcc_O2', sources=['wave_1d_fd_perf/vc5.c'], extra_compile_args=['-march=native', '-O2', '-fopenmp', '-std=c99'])

libvc5_gcc_O3 = Extension(name='wave_1d_fd_perf.libvc5_gcc_O3', sources=['wave_1d_fd_perf/vc5.c'], extra_compile_args=['-march=native', '-O3', '-fopenmp', '-std=c99'])

libvc5_gcc_Ofast = Extension(name='wave_1d_fd_perf.libvc5_gcc_Ofast', sources=['wave_1d_fd_perf/vc5.c'], extra_compile_args=['-march=native', '-Ofast', '-fopenmp', '-std=c99'])

numpy.distutils.core.setup(
        name='wave_1d_fd_perf',
        version='0.0.4',
        description='Performance analysis of different implementations of 1d finite difference wave propagation',
        url='https://github.com/ar4/wave_1d_fd_perf',
        author='Alan Richardson',
        license='MIT',
        packages=['wave_1d_fd_perf'],
        install_requires=['numpy','pandas','numba'],
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
        ],
        ext_modules=[vfortran1, vfortran2, vfortran3, libvc1_gcc, libvc2_gcc_O2, libvc2_gcc_O3, libvc2_gcc_Ofast, libvc2_gcc_Ofast_autopar, libvc3_gcc, libvc4_gcc, libvc5_gcc_O2, libvc5_gcc_O3, libvc5_gcc_Ofast]
)
