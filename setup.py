#!/usr/bin/env python

from setuptools import setup, find_packages
from setuptools.dist import Distribution

class BinaryDistribution(Distribution):
    def is_pure(self):
        return False

setup(name='speccy',
    version='0.0.1',
    url='https://github.com/TIDE-ITRH/speccy',
    description='Spectral analysis toolkit for data science',
    author='Lachlan Astfalck; Matt Rayson; Andrew Zulberti',
    author_email='lachlan.astfalck@uwa.edu.au',
    packages=find_packages(),
    install_requires=['numpy', 'scipy',],
    license='MIT',
    include_package_data=True,
    distclass=BinaryDistribution,
)
