#pragma once

#include "utils/aliases.h"
#include "utils/matrix.h"

#include <cstdint>

/// @return $$x \in [-1, 1]$$
double random_value();

/// @brief $$\sigma(x) = \tfrac{1}{1 + e^{-x}} = \tfrac{e^x}{e^x + 1}$$
double sigmoid(double x);

Vector sigmoid(const Vector &z);

/// @brief $$\sigma'(z) = \tfrac{e^{-z}}{(1 + e^{-z})^2} = \sigma(z)(1 - \sigma(z))$$
double sigmoid_derivative(double z);

Vector operator*(const Matrix<double> &A, const Vector &b);

Vector operator+(const Vector &a, const Vector &b);

Vector matrix_to_vector(const Matrix<uint8_t> &matrix);

Vector label_to_vector(uint8_t label);
