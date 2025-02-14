from setuptools import setup, Extension

import os
import sys
import setuptools
import glob
import pybind11

__version__ = '0.0.2'

# with open("README.md", "r") as readme_file:
#     readme = readme_file.read()

extra_compile_args_dict = {
    'linux' : ['-w', '-ftemplate-backtrace-limit=0', '-std=c++11'],
    'linux2' : ['-w', '-ftemplate-backtrace-limit=0', '-std=c++11'],
    'darwin' : ['-w', '-ftemplate-backtrace-limit=0', '-std=c++11', '-stdlib=libc++'],
}

ext_modules = [
  Extension(
    "_atomic_depth",
    glob.glob('src/*.cc'),
    include_dirs = [ 'src', pybind11.get_include()],
    language = 'c++',
    extra_compile_args = extra_compile_args_dict[sys.platform],
    extra_link_args = ['-lz'],
    define_macros = [('DOCTEST_CONFIG_DISABLE', None)]
  )
]

setup(
    name = 'atomic_depth',
    version = __version__,
    author = 'Brian Coventry',
    author_email = 'bcov@uw.edu',
    description = 'Atomic Depth',
    packages = ['atomic_depth'],
    package_dir={'atomic_depth': 'atomic_depth'},
    ext_modules = ext_modules,
    install_requires=[
        'pybind11>=2.10.0',  # Specify compatible pybind11 versions
    ],
)