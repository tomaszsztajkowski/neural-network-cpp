#include "neural.h"

void zad1zad2();

char *read_numbers(std::string filename, int offset, const long bytes) {
    char *output = (char *) malloc(bytes);
    std::fstream file(filename, std::ios::in | std::ios::binary);
    file.seekg(offset, std::fstream::beg);
    file.read(output, bytes);
    file.close();
    return output;
}

void show_number(const matrix_t images, const matrix_t labels, size_t n) {
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
    const long count_train = 1 * 1000;
    const long count_test = 10 * 1000;
    const long image_size = 28 * 28;
    const long epochs = 350;
    const long batch_size = 100;
    const double learning_rate = 0.2;

    char *labels = read_numbers("train-labels.idx1-ubyte", 8, count_train);
    char *images = read_numbers("train-images.idx3-ubyte", 16, count_train * image_size);

    char *labels_test = read_numbers("t10k-labels.idx1-ubyte", 8, count_test);
    char *images_test = read_numbers("t10k-images.idx3-ubyte", 16, count_test * image_size);

    network_t network{{}, learning_rate};
    add_layer(network, 40, image_size, -0.01, 0.01, TANH);
    add_layer(network, 10, 40, -0.1, 0.1, SOFTMAX);

    for (int e = 0; e < epochs; ++e) {
        for (int b = 0; b < count_train / batch_size; ++b) {
            matrix_t input = {(double *) malloc(batch_size * image_size * sizeof(double)), image_size, batch_size};
            matrix_t expected = {(double *) calloc(10 * batch_size, sizeof(double)), 10, batch_size};
            for (int i = 0; i < batch_size; ++i)
                expected[labels[b * batch_size + i] * batch_size + i] = 1;
            for (int j = 0; j < batch_size; ++j) {
                for (int i = 0; i < image_size; ++i)
                    input[j + i * batch_size] = (uint8_t) images[b * batch_size * image_size + j * image_size + i] / 255.0;
            }

            fit_dropout(network, input, expected);
            free(input.values);
            free(expected.values);
        }

        int correct = 0;
        for (int i = 0; i < count_test; ++i) {
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
        std::cout << e << ' ' << (double) correct / (double) count_test << '\n';
        save_layers(network, "network_numbers.bin");
    }

    destroy_network(network);
    free(images);
    free(images_test);
    free(labels);
    free(labels_test);

    return 0;
}

void zad1zad2() {
    const long count_train = 1 * 1000;
    const long count_test = 10 * 1000;
    const long image_size = 28 * 28;
    const long epochs = 350;
    const long batch_size = 1;
    const double learning_rate = 0.005;

    char *labels = read_numbers("train-labels.idx1-ubyte", 8, count_train);
    char *images = read_numbers("train-images.idx3-ubyte", 16, count_train * image_size);

    char *labels_test = read_numbers("t10k-labels.idx1-ubyte", 8, count_test);
    char *images_test = read_numbers("t10k-images.idx3-ubyte", 16, count_test * image_size);

    network_t network{{}, learning_rate};
    add_layer(network, 40, image_size, -0.1, 0.1, RELU);
    add_layer(network, 10, 40, -0.1, 0.1);

    for (int e = 0; e < epochs; ++e) {
        for (int b = 0; b < count_train / batch_size; ++b) {
            matrix_t input = {(double *) malloc(batch_size * image_size * sizeof(double)), image_size, batch_size};
            matrix_t expected = {(double *) calloc(10 * batch_size, sizeof(double)), 10, batch_size};
            for (int i = 0; i < batch_size; ++i)
                expected[labels[b * batch_size + i] * batch_size + i] = 1;
            for (int j = 0; j < batch_size; ++j) {
                for (int i = 0; i < image_size; ++i)
                    input[j + i * batch_size] = (uint8_t) images[b * batch_size * image_size + j * image_size + i] / 255.0;
            }

            fit_dropout(network, input, expected);
            free(input.values);
            free(expected.values);
        }

        int correct = 0;
        for (int i = 0; i < count_test; ++i) {
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
        std::cout << e << ' ' << (double) correct / (double) count_test << '\n';
        save_layers(network, "network_numbers.bin");
    }

    destroy_network(network);
    free(images);
    free(images_test);
    free(labels);
    free(labels_test);
}