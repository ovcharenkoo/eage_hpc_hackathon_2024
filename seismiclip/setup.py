import os
from glob import glob
from setuptools import setup, find_packages

def src(pth):
    return os.path.join(os.path.dirname(__file__), pth)

# Project description
descr = 'Repository for the 2024 EAGE-NVIDIA HPC Hackathon.'

from setuptools import setup

setup(
    name="seismiclip",
    description=descr,
    long_description=open(src('README.md')).read(),
    keywords=['inverse problems',
              'generative models',
              'deep learning',
              'tomography',
              'efwi',
              'seismic'],
    author='Mohammad H. Taufik, Randy Harsuko',
    author_email='mohammad.taufik@kaust.edu.sa, mochammad.randycaesario@kaust.edu.sa',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'setuptools_scm',
    ],
)