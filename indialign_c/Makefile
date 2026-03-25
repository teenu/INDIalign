CXX      ?= g++
NVCC     ?= nvcc
CXXFLAGS  = -O3 -ffast-math -std=c++20 -fPIC -Wall -Wextra -fopenmp
LDFLAGS   = -shared -fopenmp
CUDA_ARCH ?= native

TARGET = libindialign.so

.PHONY: all cpu gpu clean

all: gpu-try

gpu-try:
	@if command -v $(NVCC) >/dev/null 2>&1; then \
		echo "Building with CUDA support..."; \
		$(MAKE) gpu; \
	else \
		echo "nvcc not found, building CPU-only..."; \
		$(MAKE) cpu; \
	fi

cpu:
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET) indialign.cpp

gpu: gpu_kernels.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -DINDIALIGN_CUDA \
		-o $(TARGET) indialign.cpp gpu_kernels.o \
		-L$$(dirname $$(which nvcc))/../lib64 -lcudart

gpu_kernels.o: gpu_kernels.cu gpu_device.cuh search.h core.h seeds.h
	$(NVCC) -O3 --compiler-options -fPIC -std=c++20 \
		-arch=$(CUDA_ARCH) \
		-c gpu_kernels.cu -o gpu_kernels.o

clean:
	rm -f $(TARGET) gpu_kernels.o
