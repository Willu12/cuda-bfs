# Makefile for CUDA project

# Compiler
NVCC := nvcc

# Compiler flags
CFLAGS := -std=c++11

# CUDA flags
CUDAFLAGS := -arch=sm_61

# Source files
SRCS := main.cu graph.cpp

# Object files
OBJS := $(SRCS:.cu=.o)

# Target executable
TARGET := my_cuda_program

# Compile CUDA source files
%.o: %.cu
	$(NVCC) $(CFLAGS) $(CUDAFLAGS) -c $< -o $@

# Compile C++ source files
%.o: %.cpp
	$(NVCC) $(CFLAGS) -c $< -o $@

# Build executable
$(TARGET): $(OBJS)
	$(NVCC) $(CUDAFLAGS) $^ -o $(TARGET)

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)
