"""Build the portable numerical callbacks used by SciPy's quadrature routines."""

import sys

from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "fastmagma._native",
            ["src/fastmagma/_native.c"],
            libraries=[] if sys.platform == "win32" else ["m"],
        ),
        Extension("fastmagma._input", ["src/fastmagma/_input.c"]),
    ]
)
