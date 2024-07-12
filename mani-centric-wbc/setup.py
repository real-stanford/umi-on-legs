from distutils.core import setup

from setuptools import find_packages

setup(
    name="legged_gym",
    version="1.0.0",
    author="Huy ha",
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email="huyha@stanford.edu",
    description="Isaac Gym environments for Legged Robots",
    install_requires=["isaacgym", "matplotlib"],
)
