from setuptools import setup
from Cython.Build import cythonize
import numpy
import multiprocessing

setup(
    ext_modules=cythonize(
        "funciones_paralelas.pyx", gdb_debug=False, language_level="3", nthreads=-1,  compiler_directives={"profile" : True}
    ),
    include_dirs=[numpy.get_include()],
)
