# Makefile for CUDA project

# Compiler
NVCC := nvcc

# Compiler flags
CFLAGS := -std=c++11

# CUDA flags
CUDAFLAGS := -arch=sm_61

# Source files
SRCS := main.cu graph.cpp scan.cu kernels.cu bfs_prefix_scan.cu

# Target executable
TARGET := cuda_bfs

# Build executable
$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) $^ -o $(TARGET)

# Clean rule
clean:
	rm -f $(TARGET)