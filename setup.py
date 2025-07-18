from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, include_paths, library_paths

setup(
    name="cublaze",
    version="0.1",
    packages=find_packages(),
    description="Un pacchetto CUDA che implementa InfoNCE Loss per contrastive learning (SimCLR, etc.)",
    # long_description=open("README.md").read(),
    # long_description_content_type="text/markdown",
    author="Gianni Moretti",
    #author_email="tuo@email.com",
    install_requires=[
        'torch>=1.0.0'
    ],
    python_requires=">=3.8",  # Versione minima di Python
    ext_modules=[
        CUDAExtension(
            name='cublaze.infonce_cuda',  # Nome aggiornato per InfoNCE
            sources=[
                'cublaze/cuda/infonce_cuda.cu',
                'cublaze/cuda/infonce_cuda_wrapp.cpp'
            ],  # File CUDA aggiornati
            # include_dirs=include_paths(),
            library_dirs=library_paths(),  # Include le librerie di PyTorch
            include_dirs=include_paths(),  # Corretto: non serve racchiudere in lista
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--expt-relaxed-constexpr']  # Aggiunto flag per funzioni matematiche CUDA
            }
        )
    ],
    package_data={
        "cublaze": ["*.so"],  # Include tutti i .so nella cartella cublaze
    },
    cmdclass={
        'build_ext': BuildExtension
    }
)