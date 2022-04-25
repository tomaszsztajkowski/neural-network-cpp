#ifndef MATRIX_H
#define MATRIX_H

#include <cstdlib>
#include <iostream>
#include <random>
#include <cstring>
#include <cmath>


typedef struct matrix_t matrix_t;
struct matrix_t {
    double* values;
    size_t rows;
    size_t cols;

    double& operator[](size_t i) const {
        return values[i];
    }
};

void printmatrix(const matrix_t& matrix);

matrix_t sum(const matrix_t& left, const matrix_t& right);
matrix_t sum(double value, const matrix_t& matrix);
// sum in place
int sumip(const matrix_t& left, const matrix_t& right);
int sumip(double value, const matrix_t& matrix);

matrix_t diff(const matrix_t& left, const matrix_t& right);
matrix_t diff(double value, const matrix_t& matrix);
int diffip(const matrix_t& left, const matrix_t& right);
int diffip(double value, const matrix_t& matrix);

matrix_t product(const matrix_t& left, const matrix_t& right);
matrix_t sproduct(double multiplier, const matrix_t& matrix);
int sproductip(double multiplier, const matrix_t& matrix);
matrix_t oproduct(const matrix_t& left, const matrix_t& right);
double fiproduct(const matrix_t& left, const matrix_t& right);
matrix_t ewproduct(const matrix_t& left, const matrix_t& right);
int ewproductip(const matrix_t& left, const matrix_t& right);


matrix_t transpose(const matrix_t& matrix);
int transposeip(matrix_t& matrix);



matrix_t zeros(size_t rows, size_t cols);
matrix_t ones(size_t rows, size_t cols);
matrix_t random(size_t rows, size_t cols, double low, double high);
matrix_t random_ones(size_t rows, double ratio);
matrix_t random_ones_appr(size_t rows, double ratio);
matrix_t copy_matrix(const matrix_t& matrix);

void pass(matrix_t& matrix);
void relu(matrix_t& matrix);
void reluderiv(matrix_t& matrix);
void sigmoid(matrix_t& matrix);
void sigmoidderiv(matrix_t& matrix);
void tanh(matrix_t &matrix);
void tanhderiv(matrix_t &matrix);
void softmax(matrix_t &matrix);

long long choose(int n, int r);

#endif