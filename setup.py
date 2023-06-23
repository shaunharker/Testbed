# Notes on packaging
# https://packaging.python.org/tutorials/distributing-packages/#platform-wheels
#
# python setup.py sdist bdist_wheel
# twine upload dist/*
#
# note: see also https://github.com/joerick/cibuildwheel#delivering-to-pypi

from setuptools import setup, find_packages

setup(
    name='scholar',
    version='2023.05.25',
    author='Shaun Harker',
    author_email='sharker81@gmail.com',
    description='Machine Learning using PyTorch',
    long_description='',
    packages=find_packages(),  # This line is updated
    zip_safe=True,
    url='',
    download_url='',
    install_requires=['torch', 'bokeh', 'scipy', 'dill', 'numpy', 'sortedcontainers']
)
