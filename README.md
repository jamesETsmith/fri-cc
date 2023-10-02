# Fast-randomized Iteration (FRI) Coupled Cluster
[![GitHub Actions](https://github.com/jamesETsmith/fri-cc/actions/workflows/cmake.yml/badge.svg)](https://github.com/jamesETsmith/fri-cc/actions)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/6bf6b7d62d3442c89a4e4017e1e6213b)](https://www.codacy.com/gh/jamesETsmith/fri-cc/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=jamesETsmith/fri-cc&amp;utm_campaign=Badge_Grade)


:warning: WARNING: PROJECT IS UNDER CONSTRUCTION AND SUBJECT TO BREAKING CHANGES :warning:

## Installation

### Requirements
- C++20 Compiler
- cmake
- Python >= 3.6
    - PySCF
    - pytest
    - matplotlib
    - h5py

### Tested Configurations

| Compiler |      Success       |
| :------: | :----------------: |
|  g++10   | :heavy_check_mark: |


### From Source
```bash
CXX=<your desired c++ compiler> cmake -B build
cmake --build build --parallel --target install
python -m pip install -e .
```

For `cmake<3.14`
```bash
mkdir build && cd build
CXX=<your desired c++ compiler> cmake ..
make install -j
cd ..
python -m pip install -e .
```

Broadly speaking, the build/install process has the following steps:

1. Compile the shared library `libfricc`
2. Using pybind11, create Python wrappers for parts of `libfricc` called `py_rccsd`
3. Package and install the compiled Python modules.

## Testing

There are two test suites, the first in C++, can be turned on by using an extra cmake flag `-DCPP_TEST=ON`:

```bash
mkdir build && cd build
CXX=<your desired c++ compiler> cmake .. -DCPP_TEST=ON
make install -j
make test
```

The second is a Python-based test suite, which you can run after following the instructions in the [From Source](#from-source) section.

## Benchmarking
Benchmarking is done from Python and scripts are contained in `fricc/benchmark`.

