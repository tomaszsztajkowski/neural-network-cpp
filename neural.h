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
#include <matrix.h>


typedef struct layer_t layer_t;
struct layer_t {
    matrix_t weights;
    matrix_t bias;
};

layer_t load_txt_layer(const std::string& filename);

matrix_t predict(const matrix_t& input, std::vector<layer_t>& layers);

void add_layer(std::vector<layer_t>& network, size_t rows, size_t cols=0, double min=0, double max=1);
std::vector<layer_t> load_layers(const std::string& filename);
void save_layers(std::vector<layer_t>& network, const std::string& filename);
void destroy_network(std::vector<layer_t>& network);
double* correct(layer_t layer, double* input, double* predicted, double* expected, double alpha);

#endif
