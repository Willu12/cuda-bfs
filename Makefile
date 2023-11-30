CC = nvcc
all: main
main: main.cu
	${CC} -o cuda_bfs main.cu
.PHONY: clean all