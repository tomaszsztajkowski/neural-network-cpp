#include "matrix.h"

std::random_device rd;
std::mt19937 rng = std::mt19937(rd());

void printmatrix(const matrix_t &matrix) {
    for (int i = 0; i < matrix.rows; ++i) {
        for (int j = 0; j < matrix.cols; ++j)
            std::cout << matrix[i * matrix.cols + j] << ' ';
        std::cout << '\n';
    }
}

matrix_t sum(const matrix_t &left, const matrix_t &right) {
    auto *values = (double *) malloc(left.cols * left.rows * sizeof(double));

    for (int i = 0; i < left.rows * left.cols; i++)
        values[i] = left[i] + right[i];

    return {values, left.rows, left.cols};
}

matrix_t sum(double value, const matrix_t &matrix) {
    auto *values = (double *) malloc(matrix.cols * matrix.rows * sizeof(double));

    for (int i = 0; i < matrix.rows * matrix.cols; i++)
        values[i] = matrix[i] + value;

    return {values, matrix.rows, matrix.cols};
}

int sumip(const matrix_t &left, const matrix_t &right) {
    for (int i = 0; i < left.rows * left.cols; i++)
        left[i] += right[i];

    return 0;
}

int sumip(double value, const matrix_t &matrix) {
    for (int i = 0; i < matrix.rows * matrix.cols; i++)
        matrix[i] += value;

    return 0;
}

matrix_t diff(const matrix_t &left, const matrix_t &right) {
    auto *values = (double *) malloc(left.cols * left.rows * sizeof(double));

    for (int i = 0; i < left.rows * left.cols; i++)
        values[i] = left[i] - right[i];

    return {values, left.rows, left.cols};
}

matrix_t diff(double value, const matrix_t &matrix) {
    auto *values = (double *) malloc(matrix.cols * matrix.rows * sizeof(double));

    for (int i = 0; i < matrix.rows * matrix.cols; i++)
        values[i] = matrix[i] - value;

    return {values, matrix.rows, matrix.cols};
}

int diffip(const matrix_t &left, const matrix_t &right) {
    if(left.rows != right.rows || left.cols != right.cols)
        return 1;

    for (int i = 0; i < left.rows * left.cols; ++i)
        left[i] -= right[i];

    return 0;
}

int diffip(double value, const matrix_t &matrix) {
    for (int i = 0; i < matrix.rows * matrix.cols; ++i)
        matrix[i] -= value;

    return 0;
}

matrix_t product(const matrix_t &left, const matrix_t &right) {
    auto *values = (double *) calloc(left.rows * right.cols, sizeof(double));

    for (int i = 0; i < left.rows; i++) {
        for (int j = 0; j < right.cols; j++) {
            for (int k = 0; k < left.cols; k++) {
                values[i * right.cols + j] += left[i * left.cols + k] * right[k * right.cols + j];
            }
        }
    }

    return {values, left.rows, right.cols};
}

matrix_t sproduct(double multiplier, const matrix_t &matrix) {
    auto *values = (double *) malloc(matrix.rows * matrix.cols * sizeof(double));

    for (int i = 0; i < matrix.rows * matrix.cols; i++)
        values[i] = multiplier * matrix[i];

    return {values, matrix.rows, matrix.cols};
}

int sproductip(double multiplier, const matrix_t &matrix) {
    for (int i = 0; i < matrix.rows * matrix.cols; ++i)
        matrix[i] *= multiplier;

    return 0;
}

matrix_t oproduct(const matrix_t &left, const matrix_t &right) {
    matrix_t transposed = transpose(right);

    matrix_t output = product(left, transposed);
    free(transposed.values);

    return output;
}

double fiproduct(const matrix_t &left, const matrix_t &right) {
    double value = 0;

    for (int i = 0; i < left.rows * left.cols; i++)
        value += left[i] * right[i];

    return value;
}

matrix_t ewproduct(const matrix_t& left, const matrix_t& right) {
    auto* values = (double*)malloc(left.rows * left.cols * sizeof(double ));
    for(int i = 0; i < left.rows * left.cols; i++)
        values[i] = left[i] * right[i];

    return {values, left.rows, left.cols};
}
int ewproductip(const matrix_t& left, const matrix_t& right) {
    for(int i = 0; i < left.rows * left.cols; i++)
        left[i] *= right[i];
    return 0;
}

matrix_t transpose(const matrix_t& matrix) {
    auto* values = (double*)malloc(matrix.rows * matrix.cols * sizeof(double));
    #pragma omp parallel for
    for(size_t n = 0; n<matrix.rows*matrix.cols; n++) {
        size_t i = n/matrix.rows;
        size_t j = n%matrix.rows;
        values[n] = matrix[matrix.cols*j + i];
    }
    return {values, matrix.cols, matrix.rows};
}
int transposeip(matrix_t& matrix) {
    matrix_t copy = copy_matrix(matrix);
    #pragma omp parallel for
    for(size_t n = 0; n<matrix.rows*matrix.cols; n++) {
        size_t i = n/matrix.rows;
        size_t j = n%matrix.rows;
        matrix[n] = copy[matrix.cols*j + i];
    }
    free(copy.values);
    matrix.rows = copy.cols;
    matrix.cols = copy.rows;

    return 0;
}

matrix_t zeros(size_t rows, size_t cols) {
    auto *values = (double *) calloc(rows * cols, sizeof(double));
    return {values, rows, cols};
}

matrix_t ones(size_t rows, size_t cols) {
    auto *values = (double *) malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows * cols; ++i)
        values[i] = 1;
    return {values, rows, cols};
}

matrix_t random(size_t rows, size_t cols, double low, double high) {
    std::uniform_real_distribution<double> dist = std::uniform_real_distribution<double>(low, high);
    auto *values = (double *) malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows * cols; ++i)
        values[i] = dist(rng);
    return {values, rows, cols};
}

matrix_t random_ones(size_t rows, double ratio) {
    std::uniform_int_distribution<int> dist = std::uniform_int_distribution<int>(0, choose(rows, rows * ratio) - 1);
    auto *values = (double *) malloc(rows * sizeof(double));
    int ordinal = dist(rng);
    size_t ones = rows * ratio;
    for (size_t i = rows; ones > 0; --i) {
        size_t nCk = choose(i, ones);
        if(ordinal >= nCk) {
            ordinal -= nCk;
            values[i] = 1;
            --ones;
        }
    }
    return {values, rows, 1};
}

matrix_t copy_matrix(const matrix_t& matrix) {
    auto* values = (double*)malloc(matrix.rows * matrix.cols * sizeof(double));
    memcpy(values, matrix.values, matrix.rows * matrix.cols * sizeof(double ));
    return {values, matrix.rows, matrix.cols};
}

void pass(matrix_t &matrix) {}

void relu(matrix_t &matrix) {
    for (int i = 0; i < matrix.rows * matrix.cols; ++i)
        matrix[i] = matrix[i] > 0 ? matrix[i] : 0;
}

void reluderiv(matrix_t &matrix) {
    for (int i = 0; i < matrix.rows * matrix.cols; ++i)
        matrix[i] = matrix[i] > 0 ? 1 : 0;
}

void sigmoid(matrix_t &matrix) {
    for (int i = 0; i < matrix.rows * matrix.cols; ++i)
        matrix[i] = 1 / (1 + exp(-matrix[i]));
}

void sigmoidderiv(matrix_t &matrix) {
    for (int i = 0; i < matrix.rows * matrix.cols; ++i)
        matrix[i] = matrix[i] * (1 - matrix[i]);
}

void tanh(matrix_t &matrix) {
    for (int i = 0; i < matrix.rows * matrix.cols; ++i)
        matrix[i] = (exp(matrix[i]) - exp(-matrix[i])) / (exp(matrix[i]) + exp(-matrix[i]));
}

void tanhderiv(matrix_t &matrix) {
    for (int i = 0; i < matrix.rows * matrix.cols; ++i)
        matrix[i] = 1 - matrix[i] * matrix[i];
}

void softmax(matrix_t &matrix) {
    double sum = 0;
    for (int i = 0; i < matrix.rows * matrix.cols; ++i)
        sum += exp(matrix[i]);
    for (int i = 0; i < matrix.rows * matrix.cols; ++i)
        matrix[i] = exp(matrix[i]) / sum;
}

long long choose(int n, int r) {
    if(r > n) return 0;
    if(r > n - r) r = n - r; // because C(n, r) == C(n, n - r)
    long long ans = 1;
    int i;

    for(i = 1; i <= r; i++) {
        ans *= n - r + i;
        ans /= i;
    }

    return ans;
}