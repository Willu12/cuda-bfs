#include "graph.hpp"


Graph get_Graph_from_file(char const* path) {
    std::ifstream file(path,std::ios::binary);

    if (!file.is_open()) std::cout << "failed to open "  << '\n';

    int n,m = 0;
    int start_line = 0;
    int current_line = 0;
    int current_node = 0;
    int start, end;

    //first line of code is n and m
    file >> n >> m;
    std::vector<int> v_adj_list(m);
    std::vector<int> v_adj_begin(n);
    std::vector<int> v_adj_length(n);

    while (file >> start >> end)
    {
        v_adj_list[current_line] = end;

        if (start != current_node) {
            v_adj_begin[current_node] = start_line;
            v_adj_length[current_node] = current_line - start_line;
            start_line = current_line;
            current_node = start;
        }
        current_line++;
    }
    Graph G {n,m,v_adj_list,v_adj_begin,v_adj_length};
    return G;
}

void get_path(int start, int end, int *prev, int n,const std::string& fileName) {
    int len = 1;
    int* path = (int*)malloc(sizeof(int) * n);
    path[0] = end;
    int v = prev[end];
    while(v != start){
        path[len++] = v;
        v = prev[v];
    }

    int* reversed_path = (int*)malloc(sizeof(int) * (len + 1));
   // std::vector<int> reversed_path(len + 1);
    reversed_path[0] = start;
    for(int i = 0; i < len ; i++) {
        reversed_path[i + 1] = path[len -1  - i];
    }

    std::ofstream output(fileName);
    for(int i =0; i <= len; i++) {
        output <<  reversed_path[i] << '\n';
    }
    free(reversed_path);
    free(path);

    output.close();
}

void get_path_from_file(const std::string& fileName,std::vector<int>& path){
    std::ifstream file(fileName);
    int v;
    while(file >> v) {
        path.push_back(v);
    }
}

void check_output(const Graph& G, int start,int end) {

    std::vector<int> cpu_path,gpu_prefix_path,gpu_layer_path;
    get_path_from_file("output/cpu_output.txt",cpu_path);
    get_path_from_file("output/gpu_output.txt",gpu_prefix_path);
    get_path_from_file("output/gpu_output_frontier.txt",gpu_layer_path);

    if(cpu_path.size() != gpu_prefix_path.size() || cpu_path.size() != gpu_layer_path.size()) {
        std::cout<<"paths have different lengths\n";
        return;
    }
    if(check_if_path_correct(G, start, end, cpu_path) == false) {
        std::cout<<"cpu path not correct\n";
    }
    if(check_if_path_correct(G, start, end, gpu_prefix_path) == false) {
        std::cout<<"gpu_prefix path not correct\n";
    }
    if(check_if_path_correct(G, start, end, gpu_layer_path) == false) {
        std::cout<<"gpu_layer path not correct\n";
    }

}

bool check_if_path_correct(const Graph& G, int start, int end, const std::vector<int>& path) {

    if(path[0] != start) return false;
    if(path[path.size() - 1] != end) return false;

    for(int i = 1; i < path.size(); i++) {
        int u = path[i];
        int v = path[i -1];
        bool okay = false;
        //check if edge vu exists
        for(int j = 0; j<G.v_adj_length[v]; j++) {
            int neighbour = G.v_adj_list[G.v_adj_begin[v] + j];
            if(neighbour == u) okay = true;
        }
        if(!okay) {
            std::cout<<"edge"<< v << " " << u <<"doesnt exits\n";
            return false;
        }
    }

    return true;
}