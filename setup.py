import re
from io import open

from setuptools import find_packages, setup

try:
    filepath = "./auto_round/version.py"
    with open(filepath) as version_file:
        (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False, "Error: Could not open '%s' due %s\n" % (filepath, error)


def fetch_requirements(path):
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


if __name__ == "__main__":
    setup(
        name="auto_round",
        author="Intel AIPT Team",
        version=__version__,
        author_email="wenhua.cheng@intel.com, weiwei1.zhang@intel.com",
        description="Repository of AutoRound: Advanced Weight-Only Quantization Algorithm for LLMs",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        keywords="quantization,auto-around,LLM,SignRound",
        license="Apache 2.0",
        url="https://github.com/intel/auto-round",
        packages=find_packages(),
        include_package_data=False,
        install_requires=fetch_requirements("requirements.txt"),
        python_requires=">=3.7.0",
        classifiers=[
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "License :: OSI Approved :: Apache Software License",
        ],
    )
