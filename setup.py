from setuptools import setup, find_packages, Extension
import numpy as np

setup(
    name='quantaforge',
    version='0.1.3',
    author='Nicholas Davidson',
    author_email='nrddodger@gmail.com',
    description='A comprehensive library for creating and backtesting trading strategies.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ndavidson19/QuantaForge',
    package_data = {
        'quantaforge/cython_modules': ['cy_strategy.pyx','cy_portfolio.pyx'],
    },
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    setup_requires=[
        'Cython',
        'numpy'
    ],
    install_requires=[
        'sklearn',
        'polars',
        'Cython',
        'numpy',
        'yfinance'
        'ib_insync',
        'confluent_kafka'
    ],
    include_dirs=[np.get_include()],
)
