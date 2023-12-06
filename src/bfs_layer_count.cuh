#pragma once
#include "graph.hpp"

__global__ void kernel_cuda_frontier_numbers(int *v_adj_list, int *v_adj_begin, int *v_adj_length,
                                             int num_vertices, int *result, int* prev, bool *still_running,
                                             int end, int iteration);

void cuda_BFS_frontier_numbers(const Graph& G, int start, int end);