# Notes on packaging
# https://packaging.python.org/tutorials/distributing-packages/#platform-wheels
#
# python setup.py sdist bdist_wheel
# twine upload dist/*
#
# note: see also https://github.com/joerick/cibuildwheel#delivering-to-pypi

from setuptools import setup

setup(
    name='testbed',
    version='0.0.0',
    author='Shaun Harker',
    author_email='sharker81@gmail.com',
    description='Machine Learning Testbed using PyTorch',
    long_description='',
    package_dir = {'': 'src'},
    packages=['testbed'],
    zip_safe=True,
    url = '',
    download_url = '',
    install_requires=['torch', 'bokeh', 'numpy']
)
