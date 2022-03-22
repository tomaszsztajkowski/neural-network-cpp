#include "neural.h"

void destroy_network(std::vector<layer_t>& network) {
    for (layer_t layer: network) {
        free(layer.weights.values);
        free(layer.bias.values);
    }
}

matrix_t predict(const matrix_t& input, std::vector<layer_t>& layers) {
    matrix_t result = product(layers[0].weights, input);
    sumip(result, layers[0].bias);
    matrix_t temp;

    for (int i = 1; i < layers.size(); ++i) {
        temp = result;
        result = product(layers[i].weights, result);
        sumip(result, layers[i].bias);
        free(temp.values);
    }

    return result;
}

void add_layer(std::vector<layer_t> &network, size_t rows, size_t cols, double min, double max) {
    cols = cols ? cols : network.back().weights.rows;
    layer_t layer = {random(rows, cols, min, max),
                     zeros(rows, 1)};
    network.push_back(layer);
}


std::vector<layer_t> load_layers(const std::string &filename) {
    std::vector<layer_t> network = {};
    std::fstream file(filename, std::ios::in | std::ios::binary);
    int size;
    file.read((char *) &size, sizeof(int));

    for (int i = 0; i < size; i++) {
        network.push_back({});
        file.read((char *) &network[i].weights.rows, sizeof(int));
        file.read((char *) &network[i].weights.cols, sizeof(int));
        network[i].bias.rows = network[i].weights.rows;
        network[i].bias.cols = 1;
        layer_t layer = network[i];
        network[i].weights.values = (double *) malloc(layer.weights.rows * layer.weights.cols * sizeof(double));
        network[i].bias.values = (double *) malloc(layer.bias.rows * sizeof(double));
        file.read((char *) network[i].weights.values, layer.weights.rows * layer.weights.cols * sizeof(double));
        file.read((char *) network[i].bias.values, layer.bias.rows * sizeof(double));
    }

    file.close();
    return network;
}

void save_layers(std::vector<layer_t> &network, const std::string &filename) {
    std::fstream file(filename, std::ios::out | std::ios::binary);
    size_t size = network.size();
    file.write((char *) &size, sizeof(int));
    for (auto w : network) {
        file.write((char *) &w.weights.rows, sizeof(int));
        file.write((char *) &w.weights.cols, sizeof(int));
        file.write((char *) w.weights.values, w.weights.cols * w.weights.rows * sizeof(double));
        file.write((char *) w.bias.values, w.bias.rows * sizeof(double));
    }
    file.close();
}

int correct(layer_t layer, matrix_t& input, matrix_t& predicted, matrix_t& expected, double alpha) {
    matrix_t temp = diff(predicted, expected);
    sproductip(2.0/predicted.rows, temp);
    matrix_t delta = oproduct(temp, input);
    sproductip(alpha, delta);
    diffip(layer.weights, delta);

    free(temp.values);
    free(delta.values);
    return 0;
}