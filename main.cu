#include <iostream>
#include <queue>
#include <iterator>
#include <ctime>
#include <fstream>

struct Graph {
    unsigned long n{}; // |V|
    unsigned long m{}; // |E|
    std::vector<unsigned long> v_adj_list; // concatenation of all adj_lists for all vertices (size of m);
    std::vector<unsigned long> v_adj_begin; // size of n
    std::vector<unsigned long> v_adj_length; //size of m
};

void compute_bfs(const Graph& g, unsigned long start, unsigned long end, std::vector<unsigned long>& prev);
void get_path(unsigned long start, unsigned long end, const std::vector<unsigned long>& prev,unsigned long n);
void cpu_BFS(const Graph& g, unsigned long start, unsigned long end);
Graph get_Graph_from_file(char const* path);

int main() {

    Graph g;
    g.n = 4;
    g.m = 5;

    std::vector<unsigned long> v_adj_list {1,2,3,4,3};
    std::vector<unsigned long> v_adj_begin {0,1,2,3};
    std::vector<unsigned long> v_adj_length {1,1,1,2};

    g.v_adj_list = v_adj_list;
    g.v_adj_begin = v_adj_begin;
    g.v_adj_length = v_adj_length;

    //
    Graph new_graph = get_Graph_from_file("data/california.txt");
    cpu_BFS(new_graph,732,240332);

    // std::cout << path << std::endl;
    return 0;
}

void compute_bfs(const Graph& g, unsigned long start, unsigned long end, std::vector<unsigned long>& prev) {
    std::vector<bool> visited(g.n);
    std::queue<unsigned long> Q;

    Q.push(start);
    visited[start] = true;


    while(!Q.empty()) {
        int v = Q.front();
        Q.pop();

        if(visited[end]) break;

        unsigned long neighbours_count = g.v_adj_length[v];
        unsigned long neighbours_offset = g.v_adj_begin[v];
        for(int i =0; i<neighbours_count; i++) {
            unsigned long neighbour = g.v_adj_list[neighbours_offset + i];

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
    int p = 0;
}

void get_path(unsigned long start, unsigned long end, const std::vector<unsigned long>& prev, unsigned long n) {
    unsigned long len = 1;
    std::vector<unsigned long> path(n);
    path[0] = end;
    unsigned long v = prev[end];
    while(v != start) {
        path[len++] = v;
        v = prev[v];
    }

    std::vector<unsigned long> reversed_path(len + 1);
    reversed_path[0] = start;
    for(unsigned long i = 0; i < len ; i++) {
        reversed_path[i + 1] = path[len -1  - i];
    }

    std::ofstream output("output.txt");
    for(unsigned long i =1; i <= len; i++) {
        output <<  reversed_path[i] << '\n';
    }
    output.close();
}


void cpu_BFS(const Graph &g, unsigned long start, unsigned long end) {
    std::vector<unsigned long> prev(g.n);
    for(unsigned long v = 0; v<g.n; v++) {
        prev[v] = -1;
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

    unsigned long n,m = 0;
    unsigned long start_line = 0;
    unsigned long current_line = 0;
    unsigned long current_node = 0;
    unsigned long start, end;

    while (file >> start >> end) {
        m++;
    }
    n = start;
    file.clear();
    file.seekg(0, file.beg);

    std::vector<unsigned long> v_adj_list(m);
    std::vector<unsigned long> v_adj_begin(n);
    std::vector<unsigned long> v_adj_length(n);

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
    return Graph {n,m,v_adj_list,v_adj_begin,v_adj_length};
}