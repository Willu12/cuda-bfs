#include "bfs_prefix_scan.cuh"

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
    int ELEMENTS_PER_BLOCK = 1024;
    int blocks = n / ELEMENTS_PER_BLOCK;
    if(blocks == 0) blocks = 1;
    queue_from_prescan<<<blocks,512>>>(queue, prefix_scan, frontier,n);
}


cudaError_t create_queue(int* frontier,int** prefix_scan, int** queue,int n) {
    //clear previous queue
    cudaError_t err;

    if(cudaSuccess != (err = cudaMemset( *(void**)queue, 0, n * sizeof(int)) )) return err;

    if(cudaSuccess != (err = cuda_prefix_scan(frontier,prefix_scan,n))) return err;

    queue_from_prefix(*prefix_scan,*queue,frontier,n);
    return err;
}

void cuda_prefix_queue_iter(int* v_adj_list, int* v_adj_begin, int* v_adj_length,int* queue,bool* visited,int*frontier,int* prev,int end,bool* d_stop,bool* h_stop) {
    //get amount of vertices you have to iterate
    const int ELEMENTS_PER_BLOCK = 512;
    int queue_length = 0;

    cudaMemcpy(&queue_length,queue,sizeof(int),cudaMemcpyDeviceToHost);
    if(queue_length == 0) {
        *h_stop = true;
        return;
    }
    int blocks = queue_length / ELEMENTS_PER_BLOCK;
    int remainder = queue_length - blocks * ELEMENTS_PER_BLOCK;


    bfs_cuda_prescan_iter<<<blocks,ELEMENTS_PER_BLOCK>>>(v_adj_list,v_adj_begin,v_adj_length,queue,frontier,visited,prev,end,d_stop,0);
    bfs_cuda_prescan_iter<<<1,remainder>>>(v_adj_list,v_adj_begin,v_adj_length,queue,frontier,visited,prev,end,d_stop,blocks * ELEMENTS_PER_BLOCK);
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