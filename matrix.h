#ifndef MATRIX_H
#define MATRIX_H

#include <cstdlib>
#include <iostream>
#include <random>

typedef struct matrix_t matrix_t;
struct matrix_t {
    double* values;
    size_t rows;
    size_t cols;
};

std::random_device rd;
std::mt19937 rng = std::mt19937(rd());

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

matrix_t zeros(size_t rows, size_t cols);
matrix_t ones(size_t rows, size_t cols);
matrix_t random(size_t rows, size_t cols, double low, double high);

#endif