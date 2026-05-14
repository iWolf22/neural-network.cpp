#pragma once

#include "aliases.h"
#include "matrix.h"

#include <cstdint>

/// @return $$x \in [-1, 1]$$
double random_value();

/// @brief $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
double sigmoid(double x);

Vector sigmoid(const Vector &z);

/// @brief $$\sigma'(z) = \sigma(z)\,(1 - \sigma(z))$$
double sigmoid_derivative(double z);

Vector operator*(const Matrix<double> &A, const Vector &b);

Vector operator+(const Vector &a, const Vector &b);

Vector matrix_to_vector(const Matrix<uint8_t> &matrix);

Vector label_to_vector(uint8_t label);
