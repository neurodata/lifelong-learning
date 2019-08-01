# adapted from https://github.com/neurodata/primitives-interfaces/blob/master/setup.py

import os
import sys
from setuptools import setup
from setuptools.command.install import install
from subprocess import check_output, call
from sys import platform

PACKAGE_NAME = 'lifelong_forests'
MINIMUM_PYTHON_VERSION = 3, 6
VERSION = '0.0.1'

def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


def read_package_variable(key):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join(PACKAGE_NAME, '__init__.py')
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ')
            if parts and parts[0] == key:
                return parts[-1].strip("'")
    assert False, "'{0}' not found in '{1}'".format(key, module_path)

check_python_version()
setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description='An implementation of lifelong forests',
    long_description='A package for development and implementation of lifelong forests',
    author='Hayden Helm',
    author_email="hh@jhu.edu",
    packages=[
              PACKAGE_NAME,
    ],
    },
    install_requires=['numpy', 'sklearn'],
    url='https://github.com/neurodata/lifelong-learning',
)
