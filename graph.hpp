#pragma once
#include <vector>
#include <iostream>
#include <fstream>

struct Graph {
    int n; // |V|
    int m; // |E|
    std::vector<int> v_adj_list; // concatenation of all adj_lists for all vertices (size of m);
    std::vector<int> v_adj_begin; // size of n
    std::vector<int> v_adj_length; //size of m
};

Graph get_Graph_from_file(char const* path);