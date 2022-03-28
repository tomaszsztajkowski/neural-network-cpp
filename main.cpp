#include "neural.h"

void zad1();

void zad2();

void zad3();

void zad4();

int main() {
    network_t network {{}, 0.01};
    add_layer(network, 40, 784, -0.1, 0.1, RELU);
    add_layer(network, 10, 40, -0.1, 0.1);

    size_t train_count = 60000;
    size_t image_size = 28 * 28;
    size_t load_count = 100;
    size_t epochs = 1;


    char labl[load_count] = {0};
    double labels[load_count] = {0};

    char imgs[load_count * image_size] = {0};
    double images[load_count][image_size] = {0};

    for (int e = 0; e < epochs; ++e) {
        std::fstream file("train-labels.idx1-ubyte", std::ios::in | std::ios::binary);
        file.read(labl, 8);

        std::fstream file2("train-images.idx1-ubyte", std::ios::in | std::ios::binary);
        file2.read(imgs, 16);

        for (int l = 0; l < train_count / load_count; ++l) {
            std::cout << l << '\n';
            file.read(labl, load_count);
            file2.read(imgs, load_count * image_size);

            for (int i = 0; i < train_count; ++i) {
                labels[i] = labl[i];
                for (int j = 0; j < image_size; ++j) {
                    images[i][j] = (double)imgs[i*image_size + j] / 255.0;
                }
                learn(network, {images[i], image_size, 1}, {labels + i, 1, 1});
            }
        }
        file.close();
        file2.close();
    }





    return 0;
}

void zad1() {
    double ivalues[][3] = {{0.5, 0.75, 0.1},
                           {0.1, 0.3,  0.7},
                           {0.2, 0.1,  0.6},
                           {0.8, 0.9,  0.2}};

    double hiddenvalues[] = {0.1, 0.1, -0.3,
                             0.1, 0.2, 0.0,
                             0.0, 0.7, 0.1,
                             0.2, 0.4, 0.0,
                             -0.3, 0.5, 0.1};

    layer_t hidden_layer = {{hiddenvalues, 5, 3},
                            zeros(5, 1),
                            RELU};

    double outvalues[] = {0.7, 0.9, -0.4, 0.8, 0.1,
                          0.8, 0.5, 0.3, 0.1, 0.0,
                          -0.3, 0.9, 0.3, 0.1, -0.2};

    layer_t out_layer = {{outvalues, 3, 5},
                         zeros(3, 1),
                         PASS};

    std::vector<layer_t> layers = {hidden_layer, out_layer};
    network_t network = {layers, 0.01};

    for (int i = 0; i < 4; ++i) {
        matrix_t result = predict(network, {ivalues[i], 3, 1});
        printmatrix(result);
        std::cout << '\n';
        free(result.values);
    }
}

void zad2() {
    double ivalues[][3] = {{0.5, 0.75, 0.1},
                           {0.1, 0.3,  0.7},
                           {0.2, 0.1,  0.6},
                           {0.8, 0.9,  0.2}};

    double hiddenvalues[] = {0.1, 0.1, -0.3,
                             0.1, 0.2, 0.0,
                             0.0, 0.7, 0.1,
                             0.2, 0.4, 0.0,
                             -0.3, 0.5, 0.1};

    layer_t hidden_layer = {{hiddenvalues, 5, 3},
                            zeros(5, 1),
                            RELU};

    double outvalues[] = {0.7, 0.9, -0.4, 0.8, 0.1,
                          0.8, 0.5, 0.3, 0.1, 0.0,
                          -0.3, 0.9, 0.3, 0.1, -0.2};

    layer_t out_layer = {{outvalues, 3, 5},
                         zeros(3, 1),
                         PASS};

    double expectedvalues[][3] = {{0.1, 1.0, 0.1},
                                  {0.5, 0.2, -0.5},
                                  {0.1, 0.3, 0.2},
                                  {0.7, 0.6, 0.2}};

    std::vector<layer_t> layers = {hidden_layer, out_layer};
    network_t network = {layers, 0.01};

    for (int j = 0; j < 51; ++j) {
        for (int i = 0; i < 4; ++i) {
            if(j == 0 || j == 49){
                matrix_t result = predict(network, {ivalues[i], 3, 1});
                printmatrix(result);
                std::cout << '\n';
                free(result.values);

            }
            learn(network, {ivalues[i], 3, 1}, {expectedvalues[i], 3, 1});
        }
    }
    free(out_layer.bias.values);
    free(hidden_layer.bias.values);
}

void zad3();

void zad4();