#include <iostream>
#include <queue>
#include "kernels.cuh"
#include "bfs_prefix_scan.cuh"
#include "graph.hpp"
#include "cuda_runtime.h"
#include "scan.cuh"
#include "device_launch_parameters.h"
#include "bfs_layer_count.cuh"



void compute_bfs(const Graph& g, int start, int end, std::vector<int>& prev);
void cpu_BFS(const Graph& g, int start, int end);
int main(int argc, char** argv) {
    const char *path = "data/wiki-topcats.txt";
    int start = 120;
    int end = 1132332;
    if(argc == 4 || argc == 5) {
        path = argv[1];
        start = atoi(argv[2]);
        end = atoi(argv[3]);
    }
    Graph new_graph = get_Graph_from_file(path);
    cpu_BFS(new_graph,start,end);

    cudaSetDevice(0);
    cuda_BFS_prefix_scan(new_graph, start, end);
    cuda_BFS_frontier_numbers(new_graph,start,end);

    //check output
    if(argc == 5) check_output(new_graph,start,end);

    return 0;
}

void compute_bfs(const Graph& g, int start, int end, std::vector<int>& prev) {
    std::vector<bool> visited(g.n);
    std::queue<int> Q;

    Q.push(start);
    visited[start] = true;

    //start measure time
    cudaEvent_t start_time,stop_time;
    float time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);
    cudaEventRecord(start_time,0);

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
    //end measure time
    cudaEventRecord(stop_time,0);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&time,start_time,stop_time);
    cudaEventDestroy(start_time);
    cudaEventDestroy(stop_time);
    std::cout<<"cpu bfs took: "<<time <<" ms\n";

}

void cpu_BFS(const Graph &g, int start, int end) {
    std::vector<int> prev(g.n);
    for(int v = 0; v<g.n; v++) {
        prev[v] = UINT_MAX;
    }
    compute_bfs(g,start,end,prev);

    get_path(start,end,prev.data(),g.n,"output/cpu_output.txt");
}