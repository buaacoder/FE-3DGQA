from pathlib import Path
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

import torch

PROJECT_NAME = "FE-3DGQA"
PACKAGE_NAME = PROJECT_NAME.replace("-", "_")
DESCRIPTION = "FE-3DGQA Model"

TORCH_VERSION = [int(x) for x in torch.__version__.split(".")[:2]]
assert TORCH_VERSION >= [1, 7], "Requires PyTorch >= 1.7.1"

def get_extensions():
    _ext_src_root = "/data2/wangzhen/FE-3DGQA/FE_3DGQA/lib/pointnet2/_ext_src"
    _ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
        "{}/src/*.cu".format(_ext_src_root)
    )
    _ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext_src',
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
                "nvcc": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
            },
        )
    ]

    return ext_modules

if __name__ == "__main__":
    version = "0.1.17"

    print(f"Building {PROJECT_NAME}-{version}")

    setup(
        name=PROJECT_NAME,
        version=version,
        author="Zhen Wang",
        author_email="1440475233@qq.com",
        url=f"https://github.com/buaacoder/{PROJECT_NAME}",
        download_url=f"https://github.com/buaacoder/{PROJECT_NAME}/tags",
        description=DESCRIPTION,
        long_description=Path("README.md").read_text(),
        long_description_content_type="text/markdown",
        packages=find_packages(exclude=("tests",)),
        package_data={PACKAGE_NAME: ["*.dll", "*.so", "*.dylib", "*.txt", "*.csv", "*.npz", "*.gz", "*.h", "*.cpp", "*.cu"]},
        zip_safe=False,
        python_requires=">=3.7, <3.10",
        install_requires=[
            "torch",
            "pillow",
            "aiofiles",
            "fastapi",
            "uvicorn[standard]",
            "python-multipart",
            "plyfile",
            "opencv-python",
            "trimesh==2.35.39",
            "tensorboardX",
            "easydict",
            "tqdm",
            "h5py",
            "matplotlib",
            "numba",
            "transformers"
        ],
        ext_modules=get_extensions(),
        cmdclass={"build_ext": BuildExtension},
    )