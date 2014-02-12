from ez_setup import use_setuptools
use_setuptools()

from setuptools import (
    find_packages,
    setup,
)

setup(
    name='parsnip',
    version='0.1.0',
    description='Library for parsing and manipulating Python code',
    long_description='Library for parsing and manipulating Python code',
    author='Kent Frazier',
    author_email='kentfrazier@gmail.com',
    url='https://github.com/kentfrazier/parsnip',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
)
