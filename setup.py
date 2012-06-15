# -*- encoding: utf8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension("pydeconv.utils", ["pydeconv/utils.pyx"],
    ),
]

setup(
    author="Alexis Mignon",
    author_email="alexis.mignon@gmail.com",
    name="pydeconv",
    version="0.0.1",
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    packages=["pydeconv"],
)


