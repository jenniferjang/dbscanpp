from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext


setup(
    name='CoreSetsMeanshift',
    version='1.0',
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("CoreSetsMeanshift",
                 sources=["core_sets_meanshift.pyx"],
                 language="c++",
                 include_dirs=[numpy.get_include()])],
    author='Jennifer Jang',
    author_email='j.jang42@gmail.com'

)
