all: install

install:

	pip uninstall -y optix
	cd optix && PYOPTIX_CMAKE_ARGS="-DOptiX_INSTALL_DIR=${HOME}/bin/NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64" pip install -e .

clean:
	rm -rf examples/path_util.py
	rm -rf optix/build
	rm -rf optix/*.egg-info
	rm -rf optix/*.so
	rm -rf optix/include
	rm -rf optix/dist

.PHONY: all clean install
