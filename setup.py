from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fri-cc",
    version="0.0.1",
    author="James E. T. Smith",
    author_email="james.smith9113@gmail.com",
    description="A hybrid C++/Python implementation of FRI-CC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jamesETsmith/fri-cc",
    packages=find_packages(include=("fricc")),
    python_requires=">=3.6",
    # install_requires=["pyscf", "pytest"],
    install_requires=["pytest", "emcee"],
)
