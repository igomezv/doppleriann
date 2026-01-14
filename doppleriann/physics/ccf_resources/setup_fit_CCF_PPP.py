#!/usr/bin/env python3
"""
Setup script to compile the C extension for fit_CCF.

Usage:
    python setup_fit_CCF_PPP.py build
"""

from setuptools import setup, Extension
import numpy
import pathlib

# Get the current directory (should be doppleriann/physics/ccf_resources/)
this_dir = pathlib.Path(__file__).parent.resolve()

# Define the extension module
module = Extension(
    name="fit_CCF",
    sources=[str(this_dir / "fit_CCF.c")],
    include_dirs=[numpy.get_include(), str(this_dir)],
    libraries=["gsl", "gslcblas", "m"],
    library_dirs=["/usr/local/lib", "/usr/lib"],
)
		
# Setup configuration
setup(name='fit_CCF',
      version='0.2',
      description="C extension for CCF fitting routines (GSL-based)",
      author='X. Dumusque, P. Plavchan, I. Gomez-Vargas',
      author_email='xavier.dumusque@unige.ch',
      url='',
      ext_modules=[module],
     )


