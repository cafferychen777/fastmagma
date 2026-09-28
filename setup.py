"""Build the portable numerical callbacks used by SciPy's quadrature routines."""

import sys

from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "magma_py._native",
            ["src/magma_py/_native.c"],
            libraries=[] if sys.platform == "win32" else ["m"],
        ),
        Extension("magma_py._input", ["src/magma_py/_input.c"]),
    ]
)
