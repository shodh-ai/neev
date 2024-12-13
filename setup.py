import setuptools
from setuptools import setup

setup(
    name="neev",
    version="0.2.0",
    description="An end to end solution to train and deploy your own language models.",
    author="Harshit Sandilya, Tejas Anvekar, Binoy Skaria",
    author_email="harshit@shodh.ai",
    license="LICENSE",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    install_requires=[
        "protobuf>=3.20.2",
        "pytorch_lightning",
        "lightning",
        "tensorboard",
        "litdata",
        "deepspeed",
        "torch",
        "torchmetrics",
        "zstandard",
    ],
    packages=setuptools.find_packages(),
)
