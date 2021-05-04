# Fast-randomized Iteration (FRI) Coupled Cluster
[![GitHub Actions](https://github.com/jamesETsmith/fri-cc/actions/workflows/cmake.yml/badge.svg)](https://github.com/jamesETsmith/fri-cc/actions)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/6bf6b7d62d3442c89a4e4017e1e6213b)](https://www.codacy.com/gh/jamesETsmith/fri-cc/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=jamesETsmith/fri-cc&amp;utm_campaign=Badge_Grade)


:warning: WARNING: PROJECT IS UNDER CONSTRUCTION AND SUBJECT TO BREAKING CHANGES :warning:

## Installation

### Requirements
- C++20 Compiler
- cmake
- BLAS and LAPACK
- Intel TBB
- Eigen (packaged with project)
- pybind11 (packages with project)
- Python >= 3.6
    - PySCF
    - pytest

### From source

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
