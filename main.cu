#include <iostream>
#include <queue>
#include <ctime>
#include <fstream>
#include "graph.hpp"
#include "cuda_runtime.h"
#include "scan.cuh"
#include "device_launch_parameters.h"
#include <stdio.h>


void compute_bfs(const Graph& g, int start, int end, std::vector<int>& prev);
void get_path(int start, int end, const std::vector<int>& prev,int n);
void cpu_BFS(const Graph& g, int start, int end);
cudaError_t cuda_init(const Graph&, int** , int** , int** ,int** ,
int** ,bool** , int** ,int** );
cudaError_t cuda_BFS_prefix_scan(const Graph& G, int start, int end);

inline cudaError_t cuda_calloc( void *devPtr, size_t size );
cudaError_t create_queue(int* frontier,int** prefix_scan, int** queue,int n);
int main() {
    Graph new_graph = get_Graph_from_file("data/california.txt");
    cpu_BFS(new_graph,732,240332);
    cuda_BFS_prefix_scan(new_graph, 732, 240332);

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

void get_path(int start, int end, const std::vector<int>& prev, int n) {
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

    std::ofstream output("output.txt");
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

    std::clock_t start_clock;
    double duration;
    start_clock = std::clock();
    compute_bfs(g,start,end,prev);
    duration = (double) (std::clock() - start_clock) /  (double) CLOCKS_PER_SEC;

    std::cout<<"cpu bfs took: "<<duration <<" seconds\n";

    get_path(start,end,prev,g.n);
}

cudaError_t cuda_BFS_prefix_scan(const Graph& G, int start, int end) {
    //tutaj trzeba zrobić wszystko 
    
    // inicjalizuje tabilice
    int* v_adj_list;
    int* v_adj_begin;
    int* v_adj_length;
    int* queue;
    int* prev;
    int* prefix_scan;
    bool* visited;
    int* frontier;
    cudaError_t cudaStatus;

    bool still_running = true;

    cudaStatus = cuda_init(G,&v_adj_list,&v_adj_begin,&v_adj_length,&queue,&prev,&visited,&frontier,&prefix_scan);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda init failed");
        goto Error;
    }

    //po przekopiowaniu danych mamy BFS
    //byśmy chcieli mieć funkcje która policzymy nam kolejke dla zadanego 
    frontier[start] = true;

    
    //main loop

    while(still_running) {
        //create queue
        create_queue(frontier,&prefix_scan,&queue,G.n);
        //memset frontier to zero
        //do BFS LAYER
        //check if finished
    }


    Error:
    //cudaFree(v_adj_list);
    //cudaFree(v_adj_begin);
    //cudaFree(v_adj_length);
    
    return cudaStatus;
}


cudaError_t cuda_init(const Graph& G, int** v_adj_list, int** v_adj_begin, int** v_adj_length,int** queue,
int** prev,bool** visited, int** frontier,int** prefix_scan) { 

    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&(*v_adj_list), G.m * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&(*v_adj_begin), G.n * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&(*v_adj_length), G.n * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cuda_calloc((void**)&(*queue), G.n * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cuda_calloc((void**)&(*prev), G.n * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cuda_calloc((void**)&(*frontier), G.n * sizeof(bool));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cuda_calloc((void**)&(*visited), G.n * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cuda_calloc((void**)&(*prefix_scan), G.n * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(*v_adj_list, G.v_adj_list.data(), G.m * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(*v_adj_begin, G.v_adj_begin.data(), G.n * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(*v_adj_length, G.v_adj_length.data(), G.n * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    //prev queue visited i frontier na 0
   // cudaM




    Error:
    cudaFree(*v_adj_list);
    cudaFree(*v_adj_begin);
    cudaFree(*v_adj_length);

    return cudaStatus;
}

inline cudaError_t cuda_calloc( void *devPtr, size_t size )
{
  cudaError_t err = cudaMalloc( (void**)devPtr, size );
  if( err == cudaSuccess ) err = cudaMemset( *(void**)devPtr, 0, size );
  return err;
}

cudaError_t prefix_scan(int* frontier, int** prefix_scan, int n) {
    
    //clear previous prefix_scan
    cudaError_t err = cudaMemset( *(void**)*prefix_scan, 0, n * sizeof(int) );
    if(err) return err;
    //create kernel
    scan(*prefix_scan,frontier,n);

    //printf scan
    for(int i =0; i<n; i++ ){
        std::cout<<prefix_scan[i];
    }

    return err;
}


cudaError_t create_queue(int* frontier,int** prefix_scan, int** queue,int n) {
    //clear previous queue
        cudaError_t err = cudaMemset( *(void**)*queue, 0, n * sizeof(int) );
    return err;
}