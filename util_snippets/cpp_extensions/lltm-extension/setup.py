from setuptools import setup, Extension
from torch.utils import cpp_extension

# setup(name='lltm_cpp',
#       ext_modules=[cpp_extension.CppExtension('lltm_cpp',
#                                               sources=['lltm.cpp'],
#                                               extra_link_args=['-Wl,-rpath,$ORIGIN']
#                                               )
#                   ],
#       cmdclass={'build_ext': cpp_extension.BuildExtension})

setup(name='lltm_cpp',
      ext_modules=[cpp_extension.CppExtension('lltm_cpp', ['lltm.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
