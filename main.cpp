#include "neural.h"

void zad1();

void zad2();

void zad3();

void zad4();

char *read_numbers(std::string filename, int offset, const long bytes) {
    char *output = (char *) malloc(bytes);
    std::fstream file(filename, std::ios::in | std::ios::binary);
    file.seekg(offset, std::fstream::beg);
    file.read(output, bytes);
    file.close();
    return output;
}

void show_number(const matrix_t images, const matrix_t labels,  size_t n) {
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            if (images[images.cols * (i * 28 + j) + n] != 0)
                std::cout << '0' << ' ';
            else
                std::cout << "  ";
        }
        std::cout << '\n';
    }

    for (int i = 0; i < 10; ++i) {
        std::cout << labels[labels.cols * i + n] << '\n';
    }
}



int main() {
    zad3();

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
            if (j == 0 || j == 49) {
                matrix_t result = predict(network, {ivalues[i], 3, 1});
                printmatrix(result);
                std::cout << '\n';
                free(result.values);

            }
            fit(network, {ivalues[i], 3, 1}, {expectedvalues[i], 3, 1});
        }
    }
    free(out_layer.bias.values);
    free(hidden_layer.bias.values);
}

void zad3() {
    network_t network{{}, 0.01};
//    network = load_layers("network_numbers.bin");
    add_layer(network, 40, 784, -0.1, 0.1, RELU);
    add_layer(network, 10, 40, -0.1, 0.1);

//    matrix_t hidden_copy = copy_matrix(network.layers[0].weights);

    const long train_count = 60 * 1000;
    const long test_count = 10 * 1000;
    const long image_size = 28 * 28;
    size_t epochs = 5;

    char *labels = read_numbers("train-labels.idx1-ubyte", 8, train_count);
    char *images = read_numbers("train-images.idx3-ubyte", 16, train_count * image_size);

    char *labels_test = read_numbers("t10k-labels.idx1-ubyte", 8, test_count);
    char *images_test = read_numbers("t10k-images.idx3-ubyte", 16, test_count * image_size);

    for (int e = 0; e < epochs; ++e) {
        for (int i = 0; i < train_count; ++i) {
            double values[image_size];
            for (int j = 0; j < image_size; ++j)
                values[j] = (double) (uint8_t) images[i * image_size + j] / 255;
            double lvalues[10] = {0};
            lvalues[labels[i]] = 1;
            matrix_t input = {values, image_size, 1};
            matrix_t label = {lvalues, 10, 1};

            fit_dropout(network, input, label);
        }
        std::cout << e << '\n';
        int correct = 0;
        for (int i = 0; i < test_count; ++i) {
            double values[image_size];
            for (int j = 0; j < image_size; ++j)
                values[j] = (double) (uint8_t) images_test[i * image_size + j] / 255;
            matrix_t input = {values, image_size, 1};
            matrix_t prediction = predict(network, input);

            int pred = 0;
            double max = prediction[0];
            for (int j = 1; j < 10; ++j) {
                if (prediction[j] > max) {
                    max = prediction[j];
                    pred = j;
                }
            }
            if (pred == labels_test[i])
                correct++;
        }

//        printmatrix({network.layers[0].weights.values, 40, 2});
//        std::cout << '\n';
//        printmatrix({hidden_copy.values, 40, 2});

        std::cout << correct << '/' << test_count << '\n';
        std::cout << (double) correct / (double) test_count << '\n';
        save_layers(network, "network_numbers.bin");
    }



    free(images);
    free(labels);
    free(images_test);
    free(labels_test);
    destroy_network(network);
}

void zad4() {
    network_t network{{}, 0.01};
    add_layer(network, 6, 3, 0, 1, RELU);
    add_layer(network, 4, 6, 0, 1);

    double train_input[109 * 3];
    int train_expected[109];

    double test_input[130 * 3];
    int test_expected[130];

    std::fstream train_file("train-colors.txt", std::ios::in);
    for (int i = 0; i < 109; ++i)
        train_file >> train_input[i * 3] >> train_input[i * 3 + 1] >> train_input[i * 3 + 2] >> train_expected[i];
    train_file.close();

    std::fstream test_file("test-colors.txt", std::ios::in);
    for (int i = 0; i < 130; ++i)
        test_file >> test_input[i * 3] >> test_input[i * 3 + 1] >> test_input[i * 3 + 2] >> test_expected[i];
    test_file.close();

    for (int e = 0; e < 30; ++e) {
        for (int i = 0; i < 109; ++i) {
            double values[3] = {train_input[i * 3], train_input[i * 3 + 1], train_input[i * 3 + 2]};
            matrix_t input = {values, 3, 1};
            double evalues[4] = {0};
            evalues[train_expected[i] - 1] = 1;
            matrix_t expected = {evalues, 4, 1};

            fit(network, input, expected);
        }

        int correct = 0;
        for (int i = 0; i < 130; ++i) {
            double values[3] = {test_input[i * 3], test_input[i * 3 + 1], test_input[i * 3 + 2]};
            matrix_t input = {values, 3, 1};
            matrix_t prediction = predict(network, input);

            int pred = 0;
            double max = prediction[0];
            for (int j = 1; j < 4; ++j) {
                if(prediction[j] > max) {
                    max = prediction[j];
                    pred = j;
                }
            }

            if (pred == test_expected[i] - 1)
                correct++;

            free(prediction.values);
        }
        std::cout << e << '\n';
        std::cout << correct << "/130\n";
        std::cout << correct / 130.0 << '\n';
    }

    destroy_network(network);
}