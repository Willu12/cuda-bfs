# Makefile for CUDA project

# Compiler
NVCC := nvcc

# Compiler flags
CFLAGS := -std=c++11

# CUDA flags
CUDAFLAGS :=

# Source files
SRCS := src/main.cu src/graph.cpp src/scan.cu src/kernels.cu src/bfs_prefix_scan.cu src/bfs_layer_count.cu

# Target executable
TARGET := cuda_bfs

# Build executable
$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) $(CUDAFLAGS) $^ -o $(TARGET)

# Clean rule
clean:
	rm -f $(TARGET)