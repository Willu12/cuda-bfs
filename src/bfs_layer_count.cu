#include "bfs_layer_count.cuh"
__global__ void kernel_cuda_frontier_numbers(int *v_adj_list, int *v_adj_begin, int *v_adj_length,
        int n, int *result, int* prev, bool *still_running, int end, int iteration) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    //if I would call not enough threads
    for (int v = 0; v < n; v += num_threads)
    {
        int vertex = v + tid;
        if (vertex < n && result[vertex] == iteration)
        {
            for (int i = 0; i < v_adj_length[vertex]; i++)
            {
                int neighbor = v_adj_list[v_adj_begin[vertex] + i];

                //check if not visited yet
                if (result[neighbor] == n + 1)
                {
                    result[neighbor] = iteration + 1;
                    prev[neighbor] = vertex;

                    if(neighbor == end) {

                        *still_running = false;
                        break;
                    }
                    *still_running = true; // we added neighbour to queue
                }

            }
        }
    }
}

void cuda_BFS_frontier_numbers(const Graph& G, int start, int end) {
    int* v_adj_list;
    int* v_adj_begin;
    int* v_adj_length;
    int* result;
    int* prev;

    int* h_result = (int*)malloc(G.n * sizeof(int));

    bool* running;
    int level = 0;

    cudaMalloc(&v_adj_list, sizeof(int) * G.m);
    cudaMalloc(&v_adj_begin, sizeof(int) * G.n);
    cudaMalloc(&v_adj_length, sizeof(int) * G.n);
    cudaMalloc(&prev, sizeof(int) * G.n);
    cudaMalloc(&result,sizeof(int) * G.n);
    cudaMalloc(&running, sizeof(bool) * 1);

    const int THREADS_PER_BLOCK = 512;
    int blocks = G.n / THREADS_PER_BLOCK;
    if(blocks == 0) blocks = 1;


    std::fill_n(h_result,G.n,G.n + 1);
    h_result[start] = 0;

    cudaMemcpy(v_adj_list, G.v_adj_list.data(), sizeof(int) * G.m, cudaMemcpyHostToDevice);
    cudaMemcpy(v_adj_begin, G.v_adj_begin.data(), sizeof(int) * G.n, cudaMemcpyHostToDevice);
    cudaMemcpy(v_adj_length, G.v_adj_length.data(), sizeof(int) * G.n, cudaMemcpyHostToDevice);
    cudaMemcpy(result, h_result, sizeof(int) * G.n, cudaMemcpyHostToDevice);
    cudaMemcpy(prev,h_result,sizeof(int) * G.n,cudaMemcpyHostToDevice);
    bool* h_running = new bool[1];

    //start measuring time
    cudaEvent_t start_time,stop_time;
    float time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);
    cudaEventRecord(start_time,0);

    do
    {
        *h_running = false;
        cudaMemcpy(running, h_running, sizeof(bool) * 1, cudaMemcpyHostToDevice);

        kernel_cuda_frontier_numbers<<<blocks, THREADS_PER_BLOCK>>>(v_adj_list,v_adj_begin,v_adj_length,
                                                      G.n,result,prev,running,
                                                      end,level);

        level++;
        cudaMemcpy(h_running, running, sizeof(bool) * 1, cudaMemcpyDeviceToHost);
    } while (*h_running);

    cudaEventRecord(stop_time,0);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&time,start_time,stop_time);
    cudaEventDestroy(start_time);
    cudaEventDestroy(stop_time);
    std::cout<<"gpu bfs with layer counter took: "<<time <<" ms\n";

    //copy prev array to cpu
    int* h_prev = (int*)malloc(G.n * sizeof(int));
    cudaMemcpy(h_prev,prev,G.n * sizeof(int),cudaMemcpyDeviceToHost);
    get_path(start,end,h_prev,G.n,"output/gpu_layer_output.txt");

    cudaFree(v_adj_list);
    cudaFree(v_adj_begin);
    cudaFree(v_adj_length);
    cudaFree(prev);
    cudaFree(result);
    cudaFree(running);


    free(h_prev);
    free(h_result);
    free(h_running);
}