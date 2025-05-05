build:
	mkdir -p build
	nvcc -Xcompiler -fPIC -shared kernels/baseline.cu -o build/baseline.so
	nvcc -Xcompiler -fPIC -shared kernels/shared_mem.cu -o build/shared_mem.so
	nvcc -Xcompiler -fPIC -shared kernels/tiled.cu -o build/tiled.so
	nvcc -Xcompiler -fPIC -shared kernels/tiled_shared.cu -o build/tiled_shared.so
	