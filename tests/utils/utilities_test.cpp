#include "utils/matrix.h"
#include "utils/utilities.h"

#include <gtest/gtest.h>
#include <stdexcept>

namespace {

Matrix<double> make_matrix(std::initializer_list<std::initializer_list<double>> rows) {
    const size_t row_count = rows.size();
    const size_t col_count = rows.begin()->size();
    Matrix<double> m(row_count, col_count);

    size_t i = 0;
    for (const auto &row : rows) {
        size_t j = 0;
        for (double value : row)
            m(i, j++) = value;
        ++i;
    }

    return m;
}

void expect_matrix_eq(const Matrix<double> &actual, std::initializer_list<std::initializer_list<double>> expected) {
    ASSERT_EQ(actual.rows(), expected.size());
    ASSERT_EQ(actual.cols(), expected.begin()->size());

    size_t i = 0;
    for (const auto &row : expected) {
        size_t j = 0;
        for (double value : row)
            EXPECT_DOUBLE_EQ(actual(i, j++), value) << "Mismatch at (" << i << ", " << j - 1 << ")";
        ++i;
    }
}

} // namespace

TEST(MatrixMatrixMultiplyTest, TwoByTwoKnownProduct) {
    const Matrix<double> a = make_matrix({{1, 2}, {3, 4}});
    const Matrix<double> b = make_matrix({{5, 6}, {7, 8}});

    expect_matrix_eq(a * b, {{19, 22}, {43, 50}});
}

TEST(MatrixMatrixMultiplyTest, NonSquareDimensions) {
    const Matrix<double> a = make_matrix({{1, 2, 3}, {4, 5, 6}});
    const Matrix<double> b = make_matrix({{7, 8}, {9, 10}, {11, 12}});

    expect_matrix_eq(a * b, {{58, 64}, {139, 154}});
}

TEST(MatrixMatrixMultiplyTest, DimensionMismatchThrows) {
    const Matrix<double> a = make_matrix({{1, 2}});
    const Matrix<double> b = make_matrix({{1}, {2}, {3}});

    EXPECT_THROW(static_cast<void>(a * b), std::runtime_error);
}
