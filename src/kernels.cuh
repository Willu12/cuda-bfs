#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void prescan_arbitrary(int *g_odata, int *g_idata, int n, int powerOfTwo);
__global__ void prescan_large(int *g_odata, int *g_idata, int n, int* sums);
__global__ void queue_from_prescan(int* queue,int* prefix,int* frontier,int n);
__global__ void add(int *output, int length, int *n1);
__global__ void add(int *output, int length, int *n1, int *n2);
__global__ void bfs_cuda_prescan_iter(int* v_adj_list,int* v_adj_begin,int* v_adj_length,int* queue, int* frontier, bool* visited,int* prev,
                                      int end, bool* stop,int offset);
__global__ void init_frontier(int* frontier, int start);