#include <iostream>
#include <queue>
#include <vector>
#include <ctime>
#include <fstream>
#include "kernels.cuh"
#include "bfs_prefix_scan.cuh"
#include "graph.hpp"
#include "cuda_runtime.h"
#include "scan.cuh"
#include "device_launch_parameters.h"
#include <stdio.h>



void compute_bfs(const Graph& g, int start, int end, std::vector<int>& prev);
void cpu_BFS(const Graph& g, int start, int end);
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

    //copy prev array to cpu
    int* h_prev = (int*)malloc(G.n * sizeof(int));
    cudaMemcpy(h_prev,prev,G.n * sizeof(int),cudaMemcpyDeviceToHost);
    cuda_free_all(v_adj_list,v_adj_begin, v_adj_length, queue, prev, visited, frontier, prefix_scan);

    duration = (double) (std::clock() - start_clock) /  (double) CLOCKS_PER_SEC;
    std::cout<<"gpu bfs with prefix_scan took: "<<duration <<" seconds\n";


    get_path(start,end,h_prev,G.n,"gpu_output.txt");
    free(h_prev);
    return cudaStatus;
}


