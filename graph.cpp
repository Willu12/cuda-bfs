#include "graph.hpp"


Graph get_Graph_from_file(char const* path) {
    std::ifstream file(path,std::ios::binary);

    if (!file.is_open()) std::cout << "failed to open "  << '\n';

    int n,m = 0;
    int start_line = 0;
    int current_line = 0;
    int current_node = 0;
    int start, end;

    while (file >> start >> end) {
        m++;
    }

    n = start;
    file.clear();
    file.seekg(0, file.beg);

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