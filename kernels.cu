#include "cuda_runtime.h"
#include "stdio.h"
#include "device_launch_parameters.h"
//#include <device_functions.h>

#include "kernels.cuh"

#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)

__global__ void prescan_arbitrary(int *output, int *input, int n, int powerOfTwo)
{
	extern __shared__ int temp[];// allocated on invocation
	int threadID = threadIdx.x;

	int ai = threadID;
	int bi = threadID + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	
	if (threadID < n) {
		temp[ai + bankOffsetA] = input[ai];
		temp[bi + bankOffsetB] = input[bi];
	}
	else {
		temp[ai + bankOffsetA] = 0;
		temp[bi + bankOffsetB] = 0;
	}
	

	int offset = 1;
	for (int d = powerOfTwo >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (threadID == 0) {
		temp[powerOfTwo - 1 + CONFLICT_FREE_OFFSET(powerOfTwo - 1)] = 0; // clear the last element
	}

	for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	if (threadID < n) {
		output[ai] = temp[ai + bankOffsetA];
		output[bi] = temp[bi + bankOffsetB];
	}
}

__global__ void prescan_large(int *output, int *input, int n, int *sums) {
	extern __shared__ int temp[];

	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * n;
	
	int ai = threadID;
	int bi = threadID + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	temp[ai + bankOffsetA] = input[blockOffset + ai];
	temp[bi + bankOffsetB] = input[blockOffset + bi];

	int offset = 1;
	for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	__syncthreads();


	if (threadID == 0) { 
		sums[blockID] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
		temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
	} 
	
	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	output[blockOffset + ai] = temp[ai + bankOffsetA];
	output[blockOffset + bi] = temp[bi + bankOffsetB];
}


__global__ void add(int *output, int length, int *n) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n[blockID];
}

__global__ void add(int *output, int length, int *n1, int *n2) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n1[blockID] + n2[blockID];
}

__global__ void queue_from_prescan(int* queue,int* prefix_sum,int* frontier,int n) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= n) return;
 	if (tid < n && frontier[tid]) queue[prefix_sum[tid] + 1] = tid;

 	//size of queue
 	if (tid == n - 1) queue[0] = prefix_sum[tid] + (int) frontier[tid];

}

__global__ void bfs_cuda_prescan_iter(int* v_adj_list,int* v_adj_begin,int* v_adj_length,int* queue, int* frontier, bool* visited,int *prev ,int end, bool* still_running) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int v = queue[tid + 1];
	int offset = v_adj_begin[v];
	for(int i =0; i<v_adj_length[v]; i++) {
		int u = v_adj_list[offset + i];
		if(!visited[u]) frontier[u] = 1;
		visited[u] = 1;
		prev[u] = v;
		if(u == end){
			*still_running = true;
			break;
		}
	}
}

__global__ void init_frontier(int* frontier, int start) {
    frontier[start] = 1;
}