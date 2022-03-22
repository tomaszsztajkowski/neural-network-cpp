#include "neural.h"

//layer_t load_txt_layer(const std::string &filename) {
//    layer_t g;
//
//    std::fstream input(filename);
//    input >> g.rows >> g.cols;
//    g.values = (double *) malloc(g.rows * g.cols * sizeof(double));
//    g.bias = (double *) malloc(g.rows * sizeof(double));
//    for (int i = 0; i < g.rows * g.cols; i++)
//        input >> g.values[i];
//    for (int i = 0; i < g.rows; i++)
//        input >> g.bias[i];
//
//    input.close();
//    return g;
//}

void add_layer(std::vector<layer_t> &network, size_t rows, size_t cols, double min, double max) {
    cols = cols ? cols : network.back().weights.rows;
    layer_t layer = {random(rows, cols, min, max),
                     zeros(rows, 1)};
    network.push_back(layer);
}

//std::vector<layer_t> load_layers(const std::string &filename) {
//    std::vector<layer_t> ghostnet = {};
//    std::fstream file(filename, std::ios::in | std::ios::binary);
//    int size;
//    file.read((char *) &size, sizeof(int));
//
//    for (int i = 0; i < size; i++) {
//        ghostnet.push_back({});
//        file.read((char *) &ghostnet[i].rows, sizeof(int));
//        file.read((char *) &ghostnet[i].cols, sizeof(int));
//        layer_t layer = ghostnet[i];
//        ghostnet[i].values = (double *) malloc(layer.rows * layer.cols * sizeof(double));
//        ghostnet[i].bias = (double *) malloc(layer.rows * sizeof(double));
//        file.read((char *) ghostnet[i].values, layer.rows * layer.cols * sizeof(double));
//        file.read((char *) ghostnet[i].bias, layer.rows * sizeof(double));
//    }
//
//    file.close();
//    return ghostnet;
//}
//
//void save_layers(std::vector<layer_t> &network, const std::string &filename) {
//    std::fstream file(filename, std::ios::out | std::ios::binary);
//    int size = network.size();
//    file.write((char *) &size, sizeof(int));
//    for (int i = 0; i < network.size(); i++) {
//        layer_t w = network[i];
//        file.write((char *) &w.rows, sizeof(int));
//        file.write((char *) &w.cols, sizeof(int));
//        file.write((char *) w.values, w.cols * w.rows * sizeof(double));
//        file.write((char *) w.bias, w.rows * sizeof(double));
//    }
//    file.close();
//}
//
//void destroy_network(std::vector<layer_t> &network) {
//    for (int i = 0; i < network.size(); i++) {
//        free(network[i].values);
//        free(network[i].bias);
//    }
//}
//
//double *correct(layer_t layer, double *input, double *predicted, double *expected, double alpha) {
//    double *pred_copy = (double *) malloc(layer.rows * sizeof(double));
//    std::copy(predicted, predicted + layer.rows, pred_copy);
//
//    substraction(pred_copy, expected, layer.rows);
//    product(2.0 / (double) layer.rows, pred_copy, layer.rows);
//    double *w = outer_product(pred_copy, input, layer.rows, layer.cols);
//    product(alpha, w, layer.rows * layer.cols);
//    substraction(layer.values, w, layer.rows * layer.cols);
//
//    free(pred_copy);
//    free(w);
//
//    return layer.values;
//}