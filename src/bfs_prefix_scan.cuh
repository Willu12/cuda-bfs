#pragma once
#include "graph.hpp"
#include "cuda_runtime.h"
#include <iostream>
#include "scan.cuh"
#include "kernels.cuh"

cudaError_t cuda_init(const Graph& G, int** v_adj_list, int** v_adj_begin, int** v_adj_length,int** queue,
                      int** prev,bool** visited, int** frontier,int** prefix_scan);
void cuda_free_all(int* v_adj_list, int* v_adj_begin, int* v_adj_length,int* queue,
                   int* prev,bool* visited, int* frontier,int* prefix_scan);
cudaError_t cuda_BFS_prefix_scan(const Graph& G, int start, int end);
void cuda_prefix_queue_iter(int* v_adj_list, int* v_adj_begin, int* v_adj_length,int* queue,bool* visited,int*frontier,int* prev,int end,
                            bool* d_running,bool* h_running,int n);
inline cudaError_t cuda_calloc( void *devPtr, size_t size );
cudaError_t create_queue(int* frontier,int** prefix_scan, int** queue,int n);