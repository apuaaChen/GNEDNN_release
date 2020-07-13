from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


setup(
    name='mnorm',
    version='0.0.1',
    ext_modules=[
        CUDAExtension('mnorm',
                      ['moment_norm/src/mn.cpp', 'moment_norm/src/mn_kernel.cu'],
                      extra_compile_args={'cxx':[], 'nvcc':['-arch=sm_70']},
                      include_dirs=['/home/zdchen/envs/cub-1.8.0/'])
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch']
)