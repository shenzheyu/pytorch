from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# setup(
#     name='lltm',
#     ext_modules=[
#         CUDAExtension('lltm_cuda', [
#             'lltm_cuda.cpp',
#             'lltm_cuda_kernel.cu',
#         ])
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     })

setup(
    name='add_dropout_fuse',
    ext_modules=[
        CUDAExtension('add_dropout_fuse_cuda', [
            'add_dropout_fuse_cuda.cpp',
            'add_dropout_fuse_cuda_kernel.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                'nvcc': ['-O0', '-g', '-G', '--extended-lambda']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
