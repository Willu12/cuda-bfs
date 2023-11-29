CC = nvcc
all: main
main: main.cu
	${CC} -o cpu_bfs main.cu
.PHONY: clean all