#include <iostream>
#include <queue>
#include <vector>
#include <ctime>
#include <fstream>
#include "kernels.cuh"
#include "graph.hpp"
#include "cuda_runtime.h"
#include "scan.cuh"
#include "device_launch_parameters.h"
#include <stdio.h>



void compute_bfs(const Graph& g, int start, int end, std::vector<int>& prev);
void get_path(int start, int end, const std::vector<int>& prev,int n);
void cpu_BFS(const Graph& g, int start, int end);
cudaError_t cuda_init(const Graph& G, int** v_adj_list, int** v_adj_begin, int** v_adj_length,int** queue,
                      int** prev,bool** visited, int** frontier,int** prefix_scan);
void cuda_free_all(int* v_adj_list, int* v_adj_begin, int* v_adj_length,int* queue,
int* prev,bool* visited, int* frontier,int* prefix_scan);
cudaError_t cuda_BFS_prefix_scan(const Graph& G, int start, int end);
void cuda_prefix_queue_iter(int* v_adj_list, int* v_adj_begin, int* v_adj_length,int* queue,bool* visited,int*frontier,int* prev,int end,bool* d_running,bool* h_running,int n);
inline cudaError_t cuda_calloc( void *devPtr, size_t size );
cudaError_t create_queue(int* frontier,int** prefix_scan, int** queue,int n);
int main() {
    Graph new_graph = get_Graph_from_file("data/simple.txt");
    cpu_BFS(new_graph,0,3);
    cuda_BFS_prefix_scan(new_graph, 0, 3);

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
    int* v_adj_list = nullptr;
    int* v_adj_begin = nullptr;
    int* v_adj_length = nullptr;
    int* queue = nullptr;
    int* prev = nullptr;
    int* prefix_scan = nullptr;
    bool* visited = nullptr;
    int* frontier = nullptr;
    cudaError_t cudaStatus;

    bool running = true;
    bool* d_running;
    cudaMalloc(&d_running,sizeof(bool));

    cudaStatus = cuda_init(G,&v_adj_list,&v_adj_begin,&v_adj_length,&queue,&prev,&visited,&frontier,&prefix_scan);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda init failed");
    }

    // cudaMalloc((void**)&(frontier), G.n * sizeof(int));
   //cuda_calloc(&frontier,G.n * sizeof(int));
    //po przekopiowaniu danych mamy BFS
    //byśmy chcieli mieć funkcje która policzymy nam kolejke dla zadanego 
    //frontier[0] = true;
    //cudaStatus = cudaMemset((void**)&frontier, 0, G.n * sizeof(int));
    init_frontier<<<1,1>>>(frontier,start);
    int* currentQueue = (int*)malloc(sizeof(int) * G.n);
    cudaMemcpy(currentQueue, queue, G.n * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout<<"start queue: [";
    for(int i =0; i<G.n; i++ ) {
        std::cout <<currentQueue[i] <<" ";
    }
    std::cout<< "]\n";

    //main loop

    while(running) {
        //create queue
        create_queue(frontier,&prefix_scan,&queue,G.n);
        cudaMemcpy(currentQueue, queue, G.n * sizeof(int), cudaMemcpyDeviceToHost);
        std::cout<<"current queue: [";
        for(int i =0; i<G.n; i++ ) {
            std::cout <<currentQueue[i] <<" ";
        }
        std::cout<< "]\n";

        //clear frontier
        cudaStatus = cudaMemset((void**)&frontier, 0, G.n * sizeof(int));

       // cuda_calloc(frontier,G.n * sizeof(int));
        
        //iterate through queue
        cuda_prefix_queue_iter(v_adj_list,v_adj_begin,v_adj_length,queue,visited,frontier,prev,end,d_running,&running,G.n);

        //debug


        //running = false;
    }


    cuda_free_all(v_adj_list,v_adj_begin, v_adj_length, queue, prev, visited, frontier, prefix_scan);

    
    //printf("krzeslo");
    
    return cudaStatus;
}


cudaError_t cuda_init(const Graph& G, int** v_adj_list, int** v_adj_begin, int** v_adj_length,int** queue,
int** prev,bool** visited, int** frontier,int** prefix_scan) {

    cudaError_t cudaStatus;
    /*
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
    */

    cudaStatus = cudaMalloc((void**)v_adj_list, G.m * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }



    cudaStatus = cudaMalloc((void**)v_adj_begin, G.n * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)v_adj_length, G.n * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cuda_calloc(queue, (G.n + 1) * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cuda_calloc((void**)prev, G.n * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    cudaStatus = cuda_calloc((void**)frontier, G.n * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
   // cudaMalloc((void**)frontier,G.n * sizeof())
    cudaStatus = cuda_calloc(visited, G.n * sizeof(bool));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cuda_calloc(prefix_scan, G.n * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(*(void**)v_adj_list, G.v_adj_list.data(), G.m * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(*(void**)v_adj_begin, G.v_adj_begin.data(), G.n * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(*(void**)v_adj_length, G.v_adj_length.data(), G.n * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    Error:
   // cuda_free_all(*v_adj_list,*v_adj_begin, *v_adj_length, *queue, *prev, *visited, *frontier, *prefix_scan);

    return cudaStatus;
}

inline cudaError_t cuda_calloc( void *devPtr, size_t size )
{
  cudaError_t err = cudaMalloc( (void**)devPtr, size );
  if( err == cudaSuccess ) err = cudaMemset( *(void**)devPtr, 0, size );
  return err;
}

cudaError_t cuda_prefix_scan(int* frontier, int** prefix_scan, int n) {
    
    //clear previous prefix_scan
    cudaError_t err = cudaMemset( *(void**)prefix_scan, 0, n * sizeof(int) );
    if(err != cudaSuccess) return err;
    scan(*prefix_scan,frontier,n);

    return err;
}

void queue_from_prefix(int* prefix_scan, int* queue,int* frontier, int n) {
    int ELEMENTS_PER_BLOCK = 1024;
    int blocks = n / ELEMENTS_PER_BLOCK;
    if(blocks == 0) blocks = 1;
	///const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(int);

	//int *d_sums, *d_incr;
	//cudaMalloc((void **)&d_sums, blocks * sizeof(int));
	//cudaMalloc((void **)&d_incr, blocks * sizeof(int));

	//prescan_large<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize>>>(d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);
	

	//const int sumsArrThreadsNeeded = (blocks + 1) / 2;
	//scanLargeDeviceArray(d_incr, d_sums, blocks);


    queue_from_prescan<<<blocks,512>>>(queue, prefix_scan, frontier,n);

    //cudaFree(d_sums);
	//cudaFree(d_incr);
}


cudaError_t create_queue(int* frontier,int** prefix_scan, int** queue,int n) {
    //clear previous queue
    cudaError_t err;
    if(cudaSuccess != (err = cudaMemset( *(void**)queue, 0, n * sizeof(int)) )) return err;
    if(cudaSuccess != (err = cuda_prefix_scan(frontier,prefix_scan,n))) return err;

    queue_from_prefix(*prefix_scan,*queue,frontier,n);

    //teraz chcemy jakby 

    return err;
}

void cuda_prefix_queue_iter(int* v_adj_list, int* v_adj_begin, int* v_adj_length,int* queue,bool* visited,int*frontier,int* prev,int end,bool* d_running,bool* h_running,int n) {
    int ELEMENTS_PER_BLOCK = 1024;
    const int blocks = n / ELEMENTS_PER_BLOCK;
    bfs_cuda_prescan_iter<<<blocks,512>>>(v_adj_list,v_adj_begin,v_adj_length,queue,frontier,visited,prev,end,d_running);
    cudaMemcpy(h_running, d_running, sizeof(bool), cudaMemcpyDeviceToHost);
}

void cuda_free_all(int* v_adj_list, int* v_adj_begin, int* v_adj_length,int* queue,
                   int* prev,bool* visited, int* frontier,int* prefix_scan) {
    cudaFree(v_adj_list);
    cudaFree(v_adj_begin);
    cudaFree(v_adj_length);
    cudaFree(queue);
    cudaFree(prev);
    cudaFree(visited);
    cudaFree(frontier);
    cudaFree(prefix_scan);
}