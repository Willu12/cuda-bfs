#include "bfs_layer_count.cuh"

__global__ void kernel_cuda_frontier_numbers(int *v_adj_list, int *v_adj_begin, int *v_adj_length,
        int num_vertices, int *result, int* prev, bool *still_running, int end, int iteration) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int v = 0; v < num_vertices; v += num_threads)
    {
        int vertex = v + tid;
        if (vertex < num_vertices && result[vertex] == iteration)
        {
            for (int n = 0; n < v_adj_length[vertex]; n++)
            {
                int neighbor = v_adj_list[v_adj_begin[vertex] + n];
                if (result[neighbor] == num_vertices + 1)
                {
                    result[neighbor] = iteration + 1;
                    prev[neighbor] = vertex;

                    if(neighbor == end) {

                        *still_running = false;
                        break;
                    }
                    *still_running = true;
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

    int* h_result = new int[G.n];

    bool* running;
    int kernel_runs = 0;

    cudaMalloc(&v_adj_list, sizeof(int) * G.m);
    cudaMalloc(&v_adj_begin, sizeof(int) * G.n);
    cudaMalloc(&v_adj_length, sizeof(int) * G.n);
    cudaMalloc(&prev, sizeof(int) * G.n);
    cudaMalloc(&result,sizeof(int) * G.n);
    cudaMalloc(&running, sizeof(bool) * 1);


    int ELEMENTS_PER_BLOCK = 1024;
    int blocks = G.n / ELEMENTS_PER_BLOCK;
    if(blocks == 0) blocks = 1;


    std::fill_n(h_result,G.n,G.n + 1);
    h_result[start] = 0;

    cudaMemcpy(v_adj_list, G.v_adj_list.data(), sizeof(int) * G.m, cudaMemcpyHostToDevice);
    cudaMemcpy(v_adj_begin, G.v_adj_begin.data(), sizeof(int) * G.n, cudaMemcpyHostToDevice);
    cudaMemcpy(v_adj_length, G.v_adj_length.data(), sizeof(int) * G.n, cudaMemcpyHostToDevice);
    cudaMemcpy(result, h_result, sizeof(int) * G.n, cudaMemcpyHostToDevice);

    bool* h_running = new bool[1];
    double duration;
    std::clock_t start_clock = std::clock();

    do
    {
        *h_running = false;
        cudaMemcpy(running, h_running, sizeof(bool) * 1, cudaMemcpyHostToDevice);

        kernel_cuda_frontier_numbers<<<blocks, 512>>>(
                v_adj_list,
                v_adj_begin,
                v_adj_length,
                G.n,
                result,
                prev,
                running,
                end,
                kernel_runs);

        kernel_runs++;

        cudaMemcpy(h_running, running, sizeof(bool) * 1, cudaMemcpyDeviceToHost);
    } while (*h_running);

    duration = (double) (std::clock() - start_clock) /  (double) CLOCKS_PER_SEC;
    std::cout<<"gpu bfs with frontier took: "<<duration <<" seconds\n";

    //copy prev array to cpu
    int* h_prev = (int*)malloc(G.n * sizeof(int));
    cudaMemcpy(h_prev,prev,G.n * sizeof(int),cudaMemcpyDeviceToHost);
    get_path(start,end,h_prev,G.n,"gpu_output_frontier.txt");
    free(h_prev);
}