#include <iostream>
#include <queue>
#include <ctime>
#include <fstream>
#include "graph.cpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>


void compute_bfs(const Graph& g, unsigned int start, unsigned int end, std::vector<unsigned int>& prev);
void get_path(unsigned int start, unsigned int end, const std::vector<unsigned int>& prev,unsigned int n);
void cpu_BFS(const Graph& g, unsigned int start, unsigned int end);
cudaError_t cuda_BFS_prefix_scan(const Graph& G, unsigned int start, unsigned int end);

int main() {
    Graph new_graph = get_Graph_from_file("data/california.txt");
    cpu_BFS(new_graph,732,240332);

    return 0;
}

void compute_bfs(const Graph& g, unsigned int start, unsigned int end, std::vector<unsigned int>& prev) {
    std::vector<bool> visited(g.n);
    std::queue<unsigned int> Q;

    Q.push(start);
    visited[start] = true;

    while(!Q.empty()) {
        unsigned int v = Q.front();
        Q.pop();

        if(visited[end]) break;

        unsigned int neighbours_count = g.v_adj_length[v];
        unsigned int neighbours_offset = g.v_adj_begin[v];
        for(int i =0; i<neighbours_count; i++) {
            unsigned int neighbour = g.v_adj_list[neighbours_offset + i];

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

void get_path(unsigned int start, unsigned int end, const std::vector<unsigned int>& prev, unsigned int n) {
    unsigned int len = 1;
    std::vector<unsigned int> path(n);
    path[0] = end;
    unsigned int v = prev[end];
    while(v != start) {
        path[len++] = v;
        v = prev[v];
    }

    std::vector<unsigned int> reversed_path(len + 1);
    reversed_path[0] = start;
    for(unsigned int i = 0; i < len ; i++) {
        reversed_path[i + 1] = path[len -1  - i];
    }

    std::ofstream output("output.txt");
    for(unsigned int i =1; i <= len; i++) {
        output <<  reversed_path[i] << '\n';
    }
    output.close();
}


void cpu_BFS(const Graph &g, unsigned int start, unsigned int end) {
    std::vector<unsigned int> prev(g.n);
    for(unsigned int v = 0; v<g.n; v++) {
        prev[v] = UINT_MAX;
    }

    std::clock_t start_clock;
    double duration;
    start_clock = std::clock();
    compute_bfs(g,start,end,prev);
    duration = (double) (std::clock() - start_clock) /  (double) CLOCKS_PER_SEC;

    std::cout<<"cpu bfs took: "<<duration <<" seconds\n";

    get_path(start,end,prev,g.n);
}

cudaError_t cuda_BFS_prefix_scan(const Graph& G, unsigned int start, unsigned int end) {
    //tutaj trzeba zrobić wszystko 
    // inicjalizuje tabilice
}


