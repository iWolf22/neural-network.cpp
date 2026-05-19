#pragma once

#include <cstddef>
#include <vector>

/// @brief Optimized matrix that uses a continuous segment of memory
/// @tparam T the datatype stored in the matrix
template <typename T> class Matrix {
    std::vector<T> matrix_;

    const size_t rows_;
    const size_t cols_;

  public:
    Matrix(size_t rows, size_t cols) : matrix_(rows * cols), rows_(rows), cols_(cols) {}

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    T &operator()(size_t row, size_t col) { return matrix_[row * cols_ + col]; }

    const T &operator()(size_t row, size_t col) const { return matrix_[row * cols_ + col]; }
};
