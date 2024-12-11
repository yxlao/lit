# Python OptiX

## Installation

```bash
pip install cupy-cuda11x
pip install -r requirements.txt

➜ python -c "import cupy; print(cupy.__version__)"
12.2.0
➜ python -c "import pybind11; print(pybind11.__version__)"
2.11.1
➜ python -c "import numba; print(numba.__version__)"
0.57.1
➜ python -c "import pynvrtc; print(pynvrtc.__version__)"
9.2

export PYOPTIX_CMAKE_ARGS="-DOptiX_INSTALL_DIR=$HOME/bin/NVIDIA-OptiX-SDK-7.3.0-linux64-x86_64"
cd optix
pip install -e .
```
