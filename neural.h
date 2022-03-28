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
#include <cmath>
#include "matrix.h"

enum Activation {PASS, RELU, SIGMOID};

typedef struct layer_t layer_t;
struct layer_t {
    matrix_t weights;
    matrix_t bias;
    Activation activation;
};

typedef struct network_t network_t;
struct network_t {
    std::vector<layer_t> layers;
    double learning_rate;

    layer_t& operator[](int i) {
        return layers[i];
    }
};

void destroy_network(network_t& network);

matrix_t predict(network_t& network, const matrix_t& input);
void add_layer(network_t& network, size_t rows, size_t cols=0, double min=0, double max=1, enum Activation activation=PASS);

network_t load_layers(const std::string& filename);
void save_layers(network_t& network, const std::string& filename);

int learn(network_t& network, const matrix_t& input, const matrix_t& expected);

#endif
