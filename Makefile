# Makefile for CUDA project

# Compiler
NVCC := nvcc

# Compiler flags
CFLAGS := -std=c++11

# CUDA flags
CUDAFLAGS := -arch=sm_61

# Source files
SRCS := src/main.cu src/graph.cpp src/scan.cu src/kernels.cu src/bfs_prefix_scan.cu

# Target executable
TARGET := cuda_bfs

# Build executable
$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) $^ -o $(TARGET)

# Clean rule
clean:
	rm -f $(TARGET)