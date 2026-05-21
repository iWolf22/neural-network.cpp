#pragma once

#include "utils/aliases.h"
#include "utils/matrix.h"

#include <cstdint>

/// @return $$x \in [-1, 1]$$
double random_value();

/// @brief $$\sigma(x) = \tfrac{1}{1 + e^{-x}} = \tfrac{e^x}{e^x + 1}$$
double sigmoid(double x);

Vector sigmoid(const Vector &z);

Matrix<double> sigmoid(const Matrix<double> &Z);

/// @brief $$\sigma'(z) = \tfrac{e^{-z}}{(1 + e^{-z})^2} = \sigma(z)(1 - \sigma(z))$$
double sigmoid_derivative(double z);

/// @param A $$\textbf{A}^{(L)}$$
/// @return $$\sigma^\prime(\textbf{Z}^{(L)})$$
Matrix<double> sigmoid_derivative(Matrix<double> &A);

Vector operator*(const Matrix<double> &A, const Vector &b);

Vector operator*(double scalar, const Vector &v);

Matrix<double> operator*(double scalar, const Matrix<double> &A);

Matrix<double> operator*(const Matrix<double> &A, const Matrix<double> &B);

Matrix<double> operator+(const Matrix<double> &A, const Matrix<double> &B);

Matrix<double> operator+(const Matrix<double> &A, const Vector &B);

Matrix<double> hadamard(const Matrix<double> &A, const Matrix<double> &B);

Matrix<double> transpose(const Matrix<double> &A);

Vector operator+(const Vector &a, const Vector &b);

Vector matrix_to_vector(const Matrix<uint8_t> &matrix);

Vector label_to_vector(uint8_t label);
