from setuptools import find_packages, setup
from setuptools.dist import Distribution

with open("requirements.txt") as f:
    required_deps = f.read().splitlines()


class BinaryDistribution(Distribution):

    def has_ext_modules(self):
        return False

    def is_pure(self):
        return True


setup_kwargs = {}

setup(
    name='tensorrt_llm',
    version='0.1.3',
    description='TensorRT-LLM: A TensorRT Toolbox for Large Language Models',
    install_requires=required_deps,
    zip_safe=True,
    packages=find_packages(),
    package_data={
        'tensorrt_llm':
        ['libs/libth_common.so', 'libs/libnvinfer_plugin_tensorrt_llm.so']
    },
    python_requires=">=3.7, <4",
    distclass=BinaryDistribution,
    **setup_kwargs,
)
