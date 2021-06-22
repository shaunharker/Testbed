import os
import re
import sys
import platform
import subprocess

# Notes on packaging
# https://packaging.python.org/tutorials/distributing-packages/#platform-wheels
#
# python setup.py sdist bdist_wheel
# twine upload dist/*
#
# note: see also https://github.com/joerick/cibuildwheel#delivering-to-pypi

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DUSER_INCLUDE_PATH=./src/pychomp/_chomp/include']

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='pyCHomP',
    version='0.5.0',
    author='Shaun Harker',
    author_email='shaun.harker@rutgers.edu',
    description='CHomP (Computational Homology Project) Python Extension',
    long_description='',
    package_dir = {'': 'src'},
    ext_package='pychomp',
    ext_modules=[CMakeExtension('_chomp')],
    packages=['pychomp'],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    url = 'https://github.com/shaunharker/pyCHomP',
    download_url = 'https://github.com/shaunharker/pyCHomP/archive/v0.4.tar.gz',
    install_requires=['scipy', 'matplotlib', 'numpy', 'graphviz']
)
