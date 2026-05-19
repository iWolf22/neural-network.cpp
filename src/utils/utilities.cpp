#include "utils/utilities.h"

#include <cmath>
#include <random>
#include <stdexcept>

using std::exp;
using std::runtime_error;

double random_value() {
    static std::mt19937 gen(42); // sets the random seed
    static std::uniform_real_distribution<> dist(-1.0, 1.0);
    return dist(gen);
}

double sigmoid(double x) {
    if (x >= 0.0)
        return 1.0 / (1.0 + exp(-x));

    const double t = exp(x);
    return t / (t + 1.0);
}

Vector sigmoid(const Vector &z) {
    Vector result(z.size());

    for (size_t i = 0; i < z.size(); ++i)
        result[i] = sigmoid(z[i]);

    return result;
}

double sigmoid_derivative(double z) {
    const double s = sigmoid(z);
    return s * (1.0 - s);
}

Vector operator*(const Matrix<double> &A, const Vector &b) {
    if (A.cols() != b.size())
        throw runtime_error("Unmatching matrix columns to vector length");

    Vector result(A.rows());

    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j)
            result[i] += A(i, j) * b[j];

    return result;
}

Vector operator+(const Vector &a, const Vector &b) {
    if (a.size() != b.size())
        throw runtime_error("Unmatching vector lengths");

    Vector result(a.size());

    for (size_t i = 0; i < a.size(); ++i)
        result[i] = a[i] + b[i];

    return result;
}

Vector matrix_to_vector(const Matrix<uint8_t> &matrix) {
    Vector result(matrix.rows() * matrix.cols());

    for (size_t i = 0; i < matrix.rows(); ++i)
        for (size_t j = 0; j < matrix.cols(); ++j)
            result[i * matrix.cols() + j] = static_cast<double>(matrix(i, j)) / 255.0;

    return result;
}

Vector label_to_vector(uint8_t label) {
    Vector result(10);
    result.at(label) = 1;
    return result;
}
