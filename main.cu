#include <iostream>
#include <queue>
#include <vector>
#include <ctime>
#include <fstream>
#include "algorithm"
//#include "kernels.cuh"
#include "bfs_prefix_scan.cuh"
//#include "graph.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>



void compute_bfs(const Graph& g, int start, int end, std::vector<int>& prev);
void get_path(int start, int end, const std::vector<int>& prev,int n, const std::string& fileName);
void cpu_BFS(const Graph& g, int start, int end);

void cuda_prefix_queue_iter(int* v_adj_list, int* v_adj_begin, int* v_adj_length,int* queue,bool* visited,int*frontier,int* prev,int end,bool* d_running,bool* h_running);
void cuda_BFS_frontier_numbers(const Graph& G, int start, int end);
int main(int argc, char** argv) {
    const char *path = "data/california.txt";
    int start = 120;
    int end = 1132332;
    if(argc == 4) {
        path = argv[1];
        start = atoi(argv[2]);
        end = atoi(argv[3]);
    }
    Graph new_graph = get_Graph_from_file(path);
    cpu_BFS(new_graph,start,end);
    cuda_BFS_prefix_scan(new_graph, start, end);
    cuda_BFS_frontier_numbers(new_graph,start,end);

    return 0;
}

void compute_bfs(const Graph& g, int start, int end, std::vector<int>& prev) {
    std::vector<bool> visited(g.n);
    std::queue<int> Q;

    Q.push(start);
    visited[start] = true;

    while(!Q.empty()) {
        int v = Q.front();
        Q.pop();

        if(visited[end]) break;

        int neighbours_count = g.v_adj_length[v];
        int neighbours_offset = g.v_adj_begin[v];
        for(int i =0; i<neighbours_count; i++) {
            int neighbour = g.v_adj_list[neighbours_offset + i];

            if(!visited[neighbour]) {
                visited[neighbour] = true;
                prev[neighbour] = v;
                Q.push(neighbour);

                if(neighbour == end) {
                    break;
                }
            }
        }
    }
}

void get_path(int start, int end, int *prev, int n,const std::string& fileName) {
    int len = 1;
    std::vector<int> path(n);
    path[0] = end;
    int v = prev[end];
    while(v != start) {
        path[len++] = v;
        v = prev[v];
    }

    std::vector<int> reversed_path(len + 1);
    reversed_path[0] = start;
    for(int i = 0; i < len ; i++) {
        reversed_path[i + 1] = path[len -1  - i];
    }

    std::ofstream output(fileName);
    for(int i =1; i <= len; i++) {
        output <<  reversed_path[i] << '\n';
    }
    output.close();    
}


void cpu_BFS(const Graph &g, int start, int end) {
    std::vector<int> prev(g.n);
    for(int v = 0; v<g.n; v++) {
        prev[v] = UINT_MAX;
    }

    double duration;
    std::clock_t start_clock = std::clock();
    //start_clock = std::clock();
    compute_bfs(g,start,end,prev);
    duration = (double) (std::clock() - start_clock) /  (double) CLOCKS_PER_SEC;

    std::cout<<"cpu bfs took: "<<duration <<" seconds\n";

    get_path(start,end,prev.data(),g.n,"cpu_output.txt");
}

cudaError_t cuda_BFS_prefix_scan(const Graph& G, int start, int end) {
    int* v_adj_list = nullptr;
    int* v_adj_begin = nullptr;
    int* v_adj_length = nullptr;
    int* queue = nullptr;
    int* prev = nullptr;
    int* prefix_scan = nullptr;
    bool* visited = nullptr;
    int* frontier = nullptr;
    cudaError_t cudaStatus;

    double duration;
    std::clock_t start_clock = std::clock();

    bool stop = false;
    bool* d_stop;
    cudaMalloc(&d_stop,sizeof(bool));
    cudaStatus = cuda_init(G,&v_adj_list,&v_adj_begin,&v_adj_length,&queue,&prev,&visited,&frontier,&prefix_scan);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda init failed");
    }
    int* host_queue = (int*)malloc(sizeof(int) * 2);
    host_queue[0] = 1;
    host_queue[1] = start;

    cudaMemcpy(queue,host_queue,2 * sizeof(int),cudaMemcpyHostToDevice);
    free(host_queue);

    //main loop
    while(!stop) {

        //iter
        cuda_prefix_queue_iter(v_adj_list,v_adj_begin,v_adj_length,queue,visited,frontier,prev,end,d_stop,&stop);
        //create queue
        create_queue(frontier,&prefix_scan,&queue,G.n);
        //clear frontier
        cudaStatus = cudaMemset(frontier, 0, G.n * sizeof(int));
        //bfs layer scan
    }

    duration = (double) (std::clock() - start_clock) /  (double) CLOCKS_PER_SEC;
    std::cout<<"gpu bfs with prefix_scan took: "<<duration <<" seconds\n";

    //copy prev array to cpu
    int* h_prev = (int*)malloc(G.n * sizeof(int));
    cudaMemcpy(h_prev,prev,G.n * sizeof(int),cudaMemcpyDeviceToHost);
    cuda_free_all(v_adj_list,v_adj_begin, v_adj_length, queue, prev, visited, frontier, prefix_scan);

    get_path(start,end,h_prev,G.n,"gpu_output.txt");
    free(h_prev);
    return cudaStatus;
}


__global__ void kernel_cuda_frontier_numbers(
        int *v_adj_list,
        int *v_adj_begin,
        int *v_adj_length,
        int num_vertices,
        int *result,
        int* prev,
        bool *still_running,
        int end,
        int iteration)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int v = 0; v < num_vertices; v += num_threads)
    {
        int vertex = v + tid;

       //printf("result[end] = %d\n",result[end]);
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

    //std::fill_n(result, num_vertices, MAX_DIST);



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