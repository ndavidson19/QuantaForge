from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("quantaforge.cython_modules.cy_portfolio", ["quantaforge/cython_modules/cy_portfolio.pyx"]),
    Extension("quantaforge.cython_modules.cy_strategy", ["quantaforge/cython_modules/cy_strategy.pyx"]),
]

setup(
    name='quantaforge',
    version='0.1.0',
    author='Nicholas Davidson',
    author_email='nrddodger@gmail.com',
    description='A comprehensive library for creating and backtesting trading strategies.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ndavidson/quantaforge',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'scikit-learn',
        'polars'
    ],
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
)
