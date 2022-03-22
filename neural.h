#ifndef NEURAL_H
#define NEURAL_H

#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <iomanip>
#include <ctime>
#include <cstring>
#include "matrix.h"


typedef struct layer_t layer_t;
struct layer_t {
    matrix_t weights;
    matrix_t bias;
};

void destroy_network(std::vector<layer_t>& network);

matrix_t predict(const matrix_t& input, std::vector<layer_t>& layers);
void add_layer(std::vector<layer_t>& network, size_t rows, size_t cols=0, double min=0, double max=1);

std::vector<layer_t> load_layers(const std::string& filename);
void save_layers(std::vector<layer_t>& network, const std::string& filename);

int correct(layer_t layer, matrix_t& input, matrix_t& predicted, matrix_t& expected, double alpha);

#endif
