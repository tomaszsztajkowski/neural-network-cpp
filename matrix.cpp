#include "matrix.h"

std::random_device rd;
std::mt19937 rng = std::mt19937(rd());

void printmatrix(const matrix_t& matrix){
    for (int i = 0; i < matrix.rows; ++i) {
        for (int j = 0; j < matrix.cols; ++j)
            std::cout << matrix.values[i * matrix.cols + j] << ' ';
        std::cout << '\n';
    }
}

matrix_t sum(const matrix_t &left, const matrix_t &right) {
    auto *values = (double *) malloc(left.cols * left.rows * sizeof(double));

    for (int i = 0; i < left.rows * left.cols; i++)
        values[i] = left.values[i] + right.values[i];

    return {values, left.rows, left.cols};
}

matrix_t sum(double value, const matrix_t &matrix) {
    auto *values = (double *) malloc(matrix.cols * matrix.rows * sizeof(double));

    for (int i = 0; i < matrix.rows * matrix.cols; i++)
        values[i] = matrix.values[i] + value;

    return {values, matrix.rows, matrix.cols};
}

int sumip(const matrix_t &left, const matrix_t &right) {
    for (int i = 0; i < left.rows * left.cols; i++)
        left.values[i] += right.values[i];

    return 0;
}

int sumip(double value, const matrix_t &matrix) {
    for (int i = 0; i < matrix.rows * matrix.cols; i++)
        matrix.values[i] += value;

    return 0;
}

matrix_t diff(const matrix_t &left, const matrix_t &right) {
    auto *values = (double *) malloc(left.cols * left.rows * sizeof(double));

    for (int i = 0; i < left.rows * left.cols; i++)
        values[i] = left.values[i] - right.values[i];

    return {values, left.rows, left.cols};
}

matrix_t diff(double value, const matrix_t &matrix) {
    auto *values = (double *) malloc(matrix.cols * matrix.rows * sizeof(double));

    for (int i = 0; i < matrix.rows * matrix.cols; i++)
        values[i] = matrix.values[i] - value;

    return {values, matrix.rows, matrix.cols};
}

int diffip(const matrix_t &left, const matrix_t &right) {
    for (int i = 0; i < left.rows * left.cols; i++)
        left.values[i] -= right.values[i];

    return 0;
}

int diffip(double value, const matrix_t &matrix) {
    for (int i = 0; i < matrix.rows * matrix.cols; i++)
        matrix.values[i] -= value;

    return 0;
}

matrix_t product(const matrix_t& left, const matrix_t& right) {
    auto* values = (double*)calloc(left.rows * right.cols, sizeof(double));

    for(int i = 0; i < left.rows; i++) {
        for(int j = 0; j < right.cols; j++) {
            for(int k = 0; k < left.cols; k++) {
                values[i * right.cols + j] += left.values[i * left.cols + k] * right.values[k * right.cols + j];
            }
        }
    }

    return {values, left.rows, right.cols};
}
matrix_t sproduct(double multiplier, const matrix_t& matrix) {
    auto* values = (double*)malloc(matrix.rows * matrix.cols * sizeof(double));

    for(int i = 0; i < matrix.rows * matrix.cols; i++)
        values[i] = multiplier * matrix.values[i];

    return {values, matrix.rows, matrix.cols};
}
int sproductip(double multiplier, const matrix_t& matrix) {
    for(int i = 0; i < matrix.rows * matrix.cols; i++)
        matrix.values[i] *= multiplier;

    return 0;
}

matrix_t oproduct(const matrix_t& left, const matrix_t& right) {
    auto* values = (double*)malloc(left.rows * right.rows * sizeof(double));

    for(int i = 0; i < left.rows; i++) {
        for (int j = 0; j < right.rows; ++j)
            values[i * right.rows + j] = left.values[i] * right.values[j];
    }

    return {values, left.rows, right.rows};
}

double fiproduct(const matrix_t& left, const matrix_t& right) {
    double value = 0;

    for(int i = 0; i < left.rows * left.cols; i++)
        value += left.values[i] * right.values[i];

    return value;
}

matrix_t zeros(size_t rows, size_t cols) {
    auto* values = (double*)calloc(rows * cols, sizeof(double));
    return {values, rows, cols};
}
matrix_t ones(size_t rows, size_t cols) {
    auto* values = (double*)malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows * cols; ++i)
        values[i] = 1;
    return {values, rows, cols};
}
matrix_t random(size_t rows, size_t cols, double low, double high)  {
    std::uniform_real_distribution<double> dist = std::uniform_real_distribution<double>(low,high);
    auto* values = (double*)malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows * cols; ++i)
        values[i] = dist(rng);
    return {values, rows, cols};
}