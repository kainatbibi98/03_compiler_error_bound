"""Setup script for maxbound."""

from setuptools import setup, find_packages

setup(
    name="maxbound",
    version="0.1.0",
    description="Sound formal verification of model-vs-compiler discrepancy",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0,<2.0.0",
        "transformers>=4.30.0",
    ],
    python_requires=">=3.9",
)

