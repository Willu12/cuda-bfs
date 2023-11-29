#include <iostream>
#include <queue>
#include <ctime>
#include <fstream>

struct Graph {
    unsigned int n; // |V|
    unsigned int m; // |E|
    std::vector<unsigned int> v_adj_list; // concatenation of all adj_lists for all vertices (size of m);
    std::vector<unsigned int> v_adj_begin; // size of n
    std::vector<unsigned int> v_adj_length; //size of m
};

void compute_bfs(const Graph& g, unsigned int start, unsigned int end, std::vector<unsigned int>& prev);
void get_path(unsigned int start, unsigned int end, const std::vector<unsigned int>& prev,unsigned int n);
void cpu_BFS(const Graph& g, unsigned int start, unsigned int end);
Graph get_Graph_from_file(char const* path);

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

Graph get_Graph_from_file(char const* path) {
    std::ifstream file(path,std::ios::binary);

    if (!file.is_open()) std::cout << "failed to open "  << '\n';

    unsigned int n,m = 0;
    unsigned int start_line = 0;
    unsigned int current_line = 0;
    unsigned int current_node = 0;
    unsigned int start, end;

    while (file >> start >> end) {
        m++;
    }
    n = start;
    file.clear();
    file.seekg(0, file.beg);

    std::vector<unsigned int> v_adj_list(m);
    std::vector<unsigned int> v_adj_begin(n);
    std::vector<unsigned int> v_adj_length(n);

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
    struct Graph G {.n = n, .m = m,.v_adj_list =v_adj_list,.v_adj_begin = v_adj_begin,.v_adj_length = v_adj_length};
    return G;
}