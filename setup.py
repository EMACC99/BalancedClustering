from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        "funciones_paralelas.pyx", gdb_debug=True, language_level="3"
    ),
    include_dirs=[numpy.get_include()],
)
