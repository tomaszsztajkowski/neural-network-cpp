#include "neural.h"

void (*activation_functions[])(matrix_t &) = {pass, relu, sigmoid, tanh, softmax};

void (*activation_derivatives[])(matrix_t &) = {pass, reluderiv, sigmoidderiv, tanhderiv, pass};

void destroy_network(network_t &network) {
    for (layer_dense layer: network.layers) {
        free(layer.weights.values);
        free(layer.bias.values);
    }
}

matrix_t predict(network_t &network, const matrix_t &input) {
    matrix_t result = product(network[0].weights, input);
    sumip(result, network[0].bias);
    activation_functions[network[0].activation](result);
    matrix_t temp;

    for (int i = 1; i < network.layers.size(); ++i) {
        temp = result;
        result = product(network[i].weights, result);
        sumip(result, network[i].bias);
        activation_functions[network[i].activation](result);
        free(temp.values);
    }

    return result;
}

void add_layer(network_t &network, size_t rows, size_t cols, double min, double max, enum Activation activation) {
    cols = cols ? cols : network.layers.back().weights.rows;
    layer_dense layer = {random(rows, cols, min, max),
                         zeros(rows, 1),
                         activation};
    network.layers.push_back(layer);
}


network_t load_layers(const std::string &filename) {
    std::vector<layer_dense> layers = {};
    std::fstream file(filename, std::ios::in | std::ios::binary);
    int size;
    double learning_rate;
    file.read((char *) &size, sizeof(int));
    file.read((char *) &learning_rate, sizeof(double));

    for (int i = 0; i < size; i++) {
        layers.push_back({});
        file.read((char *) &layers[i].weights.rows, sizeof(int));
        file.read((char *) &layers[i].weights.cols, sizeof(int));
        layers[i].bias.rows = layers[i].weights.rows;
        layers[i].bias.cols = 1;
        layer_dense layer = layers[i];
        layers[i].weights.values = (double *) malloc(layer.weights.rows * layer.weights.cols * sizeof(double));
        layers[i].bias.values = (double *) malloc(layer.bias.rows * sizeof(double));
        file.read((char *) layers[i].weights.values, layer.weights.rows * layer.weights.cols * sizeof(double));
        file.read((char *) layers[i].bias.values, layer.bias.rows * sizeof(double));
        file.read((char *) &layers[i].activation, sizeof(Activation));
    }

    file.close();
    return {layers, learning_rate};
}

void save_layers(network_t &network, const std::string &filename) {
    std::fstream file(filename, std::ios::out | std::ios::binary);
    size_t size = network.layers.size();
    file.write((char *) &size, sizeof(int));
    file.write((char *) &network.learning_rate, sizeof(double));
    for (auto w: network.layers) {
        file.write((char *) &w.weights.rows, sizeof(int));
        file.write((char *) &w.weights.cols, sizeof(int));
        file.write((char *) w.weights.values, w.weights.cols * w.weights.rows * sizeof(double));
        file.write((char *) w.bias.values, w.bias.rows * sizeof(double));
        file.write((char *) &w.activation, sizeof(Activation));
    }
    file.close();
}

int fit(network_t &network, const matrix_t &input, const matrix_t &expected) {
    std::vector<matrix_t> predictions = {input};
    for (auto layer: network.layers) {
        predictions.push_back(product(layer.weights, predictions.back()));
//        sumip(predictions.back(), layer.bias);
        activation_functions[layer.activation](predictions.back());
    }

    std::vector<matrix_t> deltas = {diff(predictions.back(), expected)};
    sproductip(2.0 / predictions.back().rows / input.cols, deltas[0]);
    std::vector<matrix_t> weighted_deltas = {oproduct(deltas.back(), predictions[network.layers.size() - 1])};

    for (size_t i = network.layers.size() - 1; i > 0; --i) {
        matrix_t transposed = transpose(network[i].weights);
        matrix_t delta = product(transposed, deltas.back());
        matrix_t deriv = copy_matrix(predictions[i]);
        activation_derivatives[network[i - 1].activation](deriv);
        ewproductip(delta, deriv);
        deltas.push_back(delta);
        weighted_deltas.insert(weighted_deltas.begin(), oproduct(delta, predictions[i - 1]));

        free(transposed.values);
        free(deriv.values);
    }

    for (int i = 0; i < deltas.size(); ++i) {
        sproductip(network.learning_rate, weighted_deltas[i]);
        diffip(network[i].weights, weighted_deltas[i]);

        free(predictions[i + 1].values);
        free(deltas[i].values);
        free(weighted_deltas[i].values);
    }

    return 0;
}

int fit_dropout(network_t &network, const matrix_t &input, const matrix_t &expected, double dropout) {
    std::vector<matrix_t> predictions = {input};
    for (auto layer: network.layers) {
        matrix_t copy = copy_matrix(layer.weights);
        matrix_t dropout_matrix = random_ones(copy.rows, dropout);
        for (int i = 0; i < copy.rows; ++i) {
            sproductip(dropout_matrix[i] / dropout, {copy.values + copy.cols * i, copy.rows, 1});
        }
        predictions.push_back(product(layer.weights, predictions.back()));
//        sumip(predictions.back(), layer.bias);
        activation_functions[layer.activation](predictions.back());
        free(copy.values);
        free(dropout_matrix.values);
    }

    std::vector<matrix_t> deltas = {diff(predictions.back(), expected)};
    sproductip(2.0 / predictions.back().rows / input.cols, deltas[0]);
    std::vector<matrix_t> weighted_deltas = {oproduct(deltas.back(), predictions[network.layers.size() - 1])};

    for (size_t i = network.layers.size() - 1; i > 0; --i) {
        matrix_t transposed = transpose(network[i].weights);
        matrix_t delta = product(transposed, deltas.back());
        matrix_t deriv = copy_matrix(predictions[i]);
        activation_derivatives[network[i - 1].activation](deriv);
        ewproductip(delta, deriv);
        deltas.push_back(delta);
        weighted_deltas.insert(weighted_deltas.begin(), oproduct(delta, predictions[i - 1]));

        free(transposed.values);
        free(deriv.values);
    }

    for (int i = 0; i < deltas.size(); ++i) {
        sproductip(network.learning_rate, weighted_deltas[i]);
        diffip(network[i].weights, weighted_deltas[i]);

        free(predictions[i + 1].values);
        free(deltas[i].values);
        free(weighted_deltas[i].values);
    }

    return 0;
}