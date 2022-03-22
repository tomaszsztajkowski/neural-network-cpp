#include "neural.h"


int main()
{
    double ivalues[] = {0.5, 0.75, 0.1};
    matrix_t input = {ivalues, 3, 1};
    double weights[] = {0.1, 0.1, -0.3,
                        0.1, 0.2, 0.0,
                        0.0, 0.7, 0.1,
                        0.2, 0.4, 0.0,
                        -0.3, 0.5, 0.1};

    layer_t layer = {{weights, 5, 3},
                      zeros(5, 1)};

    double evalues[] = {0.1, 1.0, 0.1, 0.0, -0.1};
    matrix_t expected = {evalues, 5, 1};

    std::vector<layer_t> network = {layer};

    matrix_t predicted = predict(input, network);
    correct(network[0], input, predicted, expected, 0.01);

    printmatrix(network[0].weights);

    free(predicted.values);
//    destroy_network(network);
    return 0;
}