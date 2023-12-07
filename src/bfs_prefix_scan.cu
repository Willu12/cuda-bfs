#include "bfs_prefix_scan.cuh"

cudaError_t cuda_init(const Graph& G, int** v_adj_list, int** v_adj_begin, int** v_adj_length,int** queue,
                      int** prev,bool** visited, int** frontier,int** prefix_scan) {

    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

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

inline cudaError_t cuda_calloc( void *devPtr, size_t size ) {
    cudaError_t err = cudaMalloc( (void**)devPtr, size );
    if( err == cudaSuccess ) err = cudaMemset( *(void**)devPtr, 0, size );
    return err;
}

cudaError_t cuda_prefix_scan(int* frontier, int** prefix_scan, int n) {
    cudaError_t err = cudaMemset( *(void**)prefix_scan, 0, n * sizeof(int) );
    if(err != cudaSuccess) return err;
    scan(*prefix_scan,frontier,n);
    return err;
}

void queue_from_prefix(int* prefix_scan, int* queue,int* frontier, int n) {
    const int THREADS_PER_BLOCK = 512;
    int blocks = n / THREADS_PER_BLOCK;
    if(blocks == 0) blocks = 1;
    queue_from_prescan<<<blocks,THREADS_PER_BLOCK>>>(queue, prefix_scan, frontier,n);
}


cudaError_t create_queue(int* frontier,int** prefix_scan, int** queue,int n) {
    //clear previous queue
    cudaError_t err;

    if(cudaSuccess != (err = cudaMemset( *(void**)queue, 0, n * sizeof(int)) )) return err;

    if(cudaSuccess != (err = cuda_prefix_scan(frontier,prefix_scan,n))) return err;

    queue_from_prefix(*prefix_scan,*queue,frontier,n);
    return err;
}

void cuda_prefix_queue_iter(int* v_adj_list, int* v_adj_begin, int* v_adj_length,int* queue,bool* visited,int*frontier,int* prev,int end,
                            bool* d_stop,bool* h_stop,int n) {
    const int THREADS_PER_BLOCK = 512;
    int queue_length = 0;

    cudaMemcpy(&queue_length,queue,sizeof(int),cudaMemcpyDeviceToHost);
    if(queue_length == 0) {
        *h_stop = true;
        return;
    }
    //amount of blocks with ceil
    int blocks = queue_length / THREADS_PER_BLOCK + !!(queue_length % THREADS_PER_BLOCK);
    bfs_cuda_prescan_iter<<<blocks,THREADS_PER_BLOCK>>>(v_adj_list,v_adj_begin,v_adj_length,queue,frontier,visited,prev,end,d_stop,n);
    cudaMemcpy(h_stop, d_stop, sizeof(bool), cudaMemcpyDeviceToHost);
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

    bool stop = false;
    bool* d_stop;
    cudaMalloc(&d_stop,sizeof(bool));
    cudaMemset(d_stop,0,sizeof(bool));

    cudaStatus = cuda_init(G,&v_adj_list,&v_adj_begin,&v_adj_length,&queue,&prev,&visited,&frontier,&prefix_scan);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda init failed");
    }
    int* host_queue = (int*)malloc(sizeof(int) * 2);
    host_queue[0] = 1;
    host_queue[1] = start;

    cudaMemcpy(queue,host_queue,2 * sizeof(int),cudaMemcpyHostToDevice);
    free(host_queue);


    cudaEvent_t start_time,stop_time;
    float time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);
    cudaEventRecord(start_time,0);
    //main loop
    while(!stop) {

        //iter
        cuda_prefix_queue_iter(v_adj_list,v_adj_begin,v_adj_length,queue,visited,frontier,prev,end,d_stop,&stop,G.n);
        //create queue
        create_queue(frontier,&prefix_scan,&queue,G.n);
        //clear frontier
        cudaStatus = cudaMemset(frontier, 0, G.n * sizeof(int));
        //bfs layer scan
    }

    cudaEventRecord(stop_time,0);
    cudaEventSynchronize(stop_time);
    cudaEventElapsedTime(&time,start_time,stop_time);
    cudaEventDestroy(start_time);
    cudaEventDestroy(stop_time);
    std::cout<<"gpu bfs with prefix_scan took: "<<time <<" ms\n";


    //copy prev array to cpu
    int* h_prev = (int*)malloc(G.n * sizeof(int));
    cudaMemcpy(h_prev,prev,G.n * sizeof(int),cudaMemcpyDeviceToHost);
    cuda_free_all(v_adj_list,v_adj_begin, v_adj_length, queue, prev, visited, frontier, prefix_scan);
    cudaFree(d_stop);

    get_path(start,end,h_prev,G.n,"output/gpu_output.txt");
    free(h_prev);
    return cudaStatus;
}