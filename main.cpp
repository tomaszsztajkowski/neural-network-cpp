#include "neural.h"


int main()
{
    std::vector<layer_t> network = {};
    add_layer(network, 3, 3);

    printmatrix(network[0].weights);

    return 0;
}