# Makefile for CUDA project

# Compiler
NVCC := nvcc

# Compiler flags
CFLAGS := -std=c++11

# CUDA flags
CUDAFLAGS := -arch=sm_61

# Source files
SRCS := main.cu graph.cpp scan.cu kernels.cu bfs_prefix_scan.cu bfs_layer_count.cu

# Target executable
TARGET := my_cuda_program

# Build executable
$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) $^ -o $(TARGET)

# Clean rule
clean:
	rm -f $(TARGET)
