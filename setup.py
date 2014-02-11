from ez_setup import use_setuptools
use_setuptools()

from setuptools import (
    find_packages,
    setup,
)

setup(
    name='parsnip',
    version='0.1.0',
    description='Python parsing and manipulation',
    author='Kent Frazier',
    author_email='kentfrazier@gmail.com',
    packages=find_packages(),
)
