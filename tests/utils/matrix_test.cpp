#include "utils/matrix.h"

#include <cstddef>
#include <gtest/gtest.h>

class MatrixTest : public ::testing::Test {
  protected:
    static constexpr size_t rows_ = 3;
    static constexpr size_t cols_ = 4;

    MatrixTest() : matrix_(rows_, cols_) {}

    void SetUp() override {
        for (size_t r = 0; r < rows_; ++r)
            for (size_t c = 0; c < cols_; ++c)
                matrix_(r, c) = static_cast<int>(r * cols_ + c);
    }

    Matrix<int> matrix_;
};

TEST(MatrixConstructionTest, DimensionsAreStoredCorrectly) {
    const Matrix<int> m(5, 7);
    EXPECT_EQ(m.rows(), 5);
    EXPECT_EQ(m.cols(), 7);
}

TEST(MatrixConstructionTest, ZeroRowsAndColsIsValid) {
    const Matrix<int> m(0, 0);
    EXPECT_EQ(m.rows(), 0);
    EXPECT_EQ(m.cols(), 0);
}

TEST(MatrixConstructionTest, SingleElementMatrix) {
    const Matrix<int> m(1, 1);
    EXPECT_EQ(m.rows(), 1);
    EXPECT_EQ(m.cols(), 1);
}

TEST_F(MatrixTest, OperatorParensReadsCorrectValue) {
    EXPECT_EQ(matrix_(0, 0), 0);
    EXPECT_EQ(matrix_(1, 0), 4);
    EXPECT_EQ(matrix_(2, 3), 11);
}

TEST_F(MatrixTest, OperatorParensAllowsWrite) {
    matrix_(1, 2) = 42;
    EXPECT_EQ(matrix_(1, 2), 42);
}

TEST_F(MatrixTest, RowMajorIndexingIsCorrect) {
    for (size_t r = 0; r < rows_; ++r)
        for (size_t c = 0; c < cols_; ++c)
            EXPECT_EQ(matrix_(r, c), static_cast<int>(r * cols_ + c)) << "Mismatch at (" << r << ", " << c << ")";
}

TEST(MatrixValueTypeTest, DoubleMatrixWorks) {
    Matrix<double> m(2, 2);
    m(0, 0) = 1.5;
    m(1, 1) = 2.5;
    EXPECT_DOUBLE_EQ(m(0, 0), 1.5);
    EXPECT_DOUBLE_EQ(m(1, 1), 2.5);
}

TEST(MatrixValueTypeTest, UnsignedMatrixWorks) {
    Matrix<unsigned> m(1, 3);
    m(0, 0) = 0u;
    m(0, 1) = 1u;
    m(0, 2) = 2u;
    EXPECT_EQ(m(0, 0), 0u);
    EXPECT_EQ(m(0, 2), 2u);
}

TEST(MatrixStorageTest, RectangularMatrixHasExpectedElementCount) {
    Matrix<int> m(2, 5);

    EXPECT_EQ(m.rows(), 2u);
    EXPECT_EQ(m.cols(), 5u);

    for (size_t r = 0; r < m.rows(); ++r)
        for (size_t c = 0; c < m.cols(); ++c)
            m(r, c) = static_cast<int>(r * 10 + c);

    EXPECT_EQ(m(1, 4), 14);
    EXPECT_EQ(m(0, 0), 0);
}

TEST(MatrixStorageTest, OverwritesPreserveOtherCells) {
    Matrix<int> m(3, 3);
    m(0, 0) = 1;
    m(1, 1) = 2;
    m(2, 2) = 3;

    m(1, 1) = 99;

    EXPECT_EQ(m(0, 0), 1);
    EXPECT_EQ(m(1, 1), 99);
    EXPECT_EQ(m(2, 2), 3);
}

TEST(MatrixConstAccessTest, ConstOperatorReturnsStoredValues) {
    Matrix<double> m(2, 2);
    m(0, 0) = 1.5;
    m(1, 0) = 2.5;

    const Matrix<double> &ref = m;

    EXPECT_DOUBLE_EQ(ref(0, 0), 1.5);
    EXPECT_DOUBLE_EQ(ref(1, 0), 2.5);
}

TEST(MatrixDimensionsTest, TallSkinnyMatrix) {
    const Matrix<int> m(10, 1);
    EXPECT_EQ(m.rows(), 10u);
    EXPECT_EQ(m.cols(), 1u);
}

TEST(MatrixDimensionsTest, WideFlatMatrix) {
    const Matrix<int> m(1, 20);
    EXPECT_EQ(m.rows(), 1u);
    EXPECT_EQ(m.cols(), 20u);
}
