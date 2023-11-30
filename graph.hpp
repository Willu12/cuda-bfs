#include <iostream>
#include <fstream>
#include <vector>

struct Graph {
    unsigned int n; // |V|
    unsigned int m; // |E|
    std::vector<unsigned int> v_adj_list; // concatenation of all adj_lists for all vertices (size of m);
    std::vector<unsigned int> v_adj_begin; // size of n
    std::vector<unsigned int> v_adj_length; //size of m
};

Graph get_Graph_from_file(char const* path);