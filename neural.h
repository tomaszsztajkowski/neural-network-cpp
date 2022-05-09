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

enum Activation {
    PASS, RELU, SIGMOID, TANH, SOFTMAX
};
enum LayerType {
    DENSE, CONV
};

typedef struct layer_dense layer_dense;
struct layer_dense {
    matrix_t weights;
    matrix_t bias;
    Activation activation;
};

typedef struct layer_conv layer_conv;

struct layer_conv {
    uint8_t masks_count;
    matrix_t *masks;
    Activation activation;

    matrix_t &operator[](int i) const {
        return masks[i];
    }
};

typedef struct network_t network_t;

struct network_t {
    std::vector<layer_dense> layers;
    double learning_rate;

    layer_dense &operator[](int i) {
        return layers[i];
    }
};

struct layer_t {
    LayerType layer_type;
    union {
        layer_dense dense;
        layer_conv conv;
    } layer;
};


void destroy_network(network_t &network);

matrix_t predict(network_t &network, const matrix_t &input);

void add_layer(network_t &network, size_t rows, size_t cols = 0, double min = -0.1, double max = 0.1,
               enum Activation activation = PASS);

network_t load_layers(const std::string &filename);

void save_layers(network_t &network, const std::string &filename);

int fit(network_t &network, const matrix_t &input, const matrix_t &expected);

int fit_dropout(network_t &network, const matrix_t &input, const matrix_t &expected, double dropout = 0.5);

#endif
