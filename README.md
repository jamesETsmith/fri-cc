# Fast-randomized Iteration (FRI) Coupled Cluster

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
