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

TEST(SigmoidTest, ZeroMapsToOneHalf) {
    EXPECT_DOUBLE_EQ(sigmoid(0.0), 0.5);
}

TEST(SigmoidTest, LargePositiveApproachesOne) {
    EXPECT_NEAR(sigmoid(20.0), 1.0, 1e-6);
}

TEST(SigmoidTest, LargeNegativeApproachesZero) {
    EXPECT_NEAR(sigmoid(-20.0), 0.0, 1e-6);
}

TEST(SigmoidDerivativeTest, AtZeroEqualsOneQuarter) {
    EXPECT_NEAR(sigmoid_derivative(0.0), 0.25, 1e-9);
}

TEST(SigmoidMatrixTest, AppliesElementWise) {
    const Matrix<double> z = make_matrix({{0, 1}, {-1, 2}});
    const Matrix<double> s = sigmoid(z);

    EXPECT_NEAR(s(0, 0), 0.5, 1e-9);
    EXPECT_GT(s(0, 1), 0.5);
    EXPECT_LT(s(1, 0), 0.5);
}

TEST(TransposeTest, SwapsRowsAndCols) {
    const Matrix<double> a = make_matrix({{1, 2, 3}, {4, 5, 6}});
    const Matrix<double> t = transpose(a);

    EXPECT_EQ(t.rows(), 3u);
    EXPECT_EQ(t.cols(), 2u);
    EXPECT_DOUBLE_EQ(t(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(t(2, 0), 3.0);
}

TEST(HadamardTest, MultipliesElementWise) {
    const Matrix<double> a = make_matrix({{2, 3}, {4, 5}});
    const Matrix<double> b = make_matrix({{10, 20}, {30, 40}});

    expect_matrix_eq(hadamard(a, b), {{20, 60}, {120, 200}});
}

TEST(MatrixVectorMultiplyTest, KnownProduct) {
    const Matrix<double> a = make_matrix({{1, 2}, {3, 4}});
    const Vector v{5, 6};

    const Vector result = a * v;

    ASSERT_EQ(result.size(), 2u);
    EXPECT_DOUBLE_EQ(result[0], 17.0);
    EXPECT_DOUBLE_EQ(result[1], 39.0);
}

TEST(MatrixVectorMultiplyTest, DimensionMismatchThrows) {
    const Matrix<double> a = make_matrix({{1, 2, 3}});
    const Vector v{1, 2};

    EXPECT_THROW(static_cast<void>(a * v), std::runtime_error);
}

TEST(ScalarMultiplyTest, ScalesMatrixAndVector) {
    const Matrix<double> a = make_matrix({{1, 2}, {3, 4}});
    const Vector v{1, 2};

    expect_matrix_eq(2.0 * a, {{2, 4}, {6, 8}});

    const Vector scaled = 3.0 * v;
    ASSERT_EQ(scaled.size(), 2u);
    EXPECT_DOUBLE_EQ(scaled[0], 3.0);
    EXPECT_DOUBLE_EQ(scaled[1], 6.0);
}

TEST(MatrixAdditionTest, AddsMatchingMatrices) {
    const Matrix<double> a = make_matrix({{1, 2}, {3, 4}});
    const Matrix<double> b = make_matrix({{5, 6}, {7, 8}});

    expect_matrix_eq(a + b, {{6, 8}, {10, 12}});
}

TEST(LabelToVectorTest, OneHotEncodesDigit) {
    const Vector v = label_to_vector(4);

    ASSERT_EQ(v.size(), 10u);
    for (size_t i = 0; i < v.size(); ++i)
        EXPECT_DOUBLE_EQ(v[i], i == 4 ? 1.0 : 0.0);
}

TEST(MatrixToVectorTest, FlattensAndNormalizesPixels) {
    Matrix<uint8_t> image(2, 2);
    image(0, 0) = 0;
    image(0, 1) = 255;
    image(1, 0) = 128;
    image(1, 1) = 64;

    const Vector v = matrix_to_vector(image);

    ASSERT_EQ(v.size(), 4u);
    EXPECT_DOUBLE_EQ(v[0], 0.0);
    EXPECT_DOUBLE_EQ(v[1], 1.0);
    EXPECT_NEAR(v[2], 128.0 / 255.0, 1e-9);
    EXPECT_NEAR(v[3], 64.0 / 255.0, 1e-9);
}

TEST(RandomValueTest, StaysWithinConfiguredRange) {
    for (int i = 0; i < 50; ++i) {
        const double value = random_value();
        EXPECT_GE(value, -1.0);
        EXPECT_LE(value, 1.0);
    }
}

TEST(SigmoidVectorTest, AppliesToEachElement) {
    const Vector z{0.0, 1.0, -1.0};
    const Vector s = sigmoid(z);

    ASSERT_EQ(s.size(), 3u);
    EXPECT_NEAR(s[0], 0.5, 1e-9);
    EXPECT_GT(s[1], 0.5);
    EXPECT_LT(s[2], 0.5);
}

TEST(SigmoidDerivativeMatrixTest, UsesActivationValues) {
    Matrix<double> activations = make_matrix({{0.5, 0.2}, {0.8, 0.1}});
    const Matrix<double> derivative = sigmoid_derivative(activations);

    EXPECT_NEAR(derivative(0, 0), 0.25, 1e-9);
    EXPECT_NEAR(derivative(0, 1), 0.2 * 0.8, 1e-9);
    EXPECT_NEAR(derivative(1, 0), 0.8 * 0.2, 1e-9);
}

TEST(VectorAdditionTest, AddsMatchingVectors) {
    const Vector a{1, 2, 3};
    const Vector b{4, 5, 6};

    const Vector sum = a + b;

    ASSERT_EQ(sum.size(), 3u);
    EXPECT_DOUBLE_EQ(sum[0], 5.0);
    EXPECT_DOUBLE_EQ(sum[2], 9.0);
}

TEST(VectorAdditionTest, DimensionMismatchThrows) {
    const Vector a{1, 2};
    const Vector b{3};

    EXPECT_THROW(static_cast<void>(a + b), std::runtime_error);
}

TEST(MatrixVectorAdditionTest, AddsRowBiasToEveryColumn) {
    const Matrix<double> a = make_matrix({{1, 2}, {3, 4}});
    const Vector bias{10, 20};

    expect_matrix_eq(a + bias, {{11, 12}, {23, 24}});
}

TEST(MatrixVectorAdditionTest, DimensionMismatchThrows) {
    const Matrix<double> a = make_matrix({{1, 2}, {3, 4}});
    const Vector bias{1, 2, 3};

    EXPECT_THROW(static_cast<void>(a + bias), std::runtime_error);
}

TEST(HadamardTest, DimensionMismatchThrows) {
    const Matrix<double> a = make_matrix({{1, 2}});
    const Matrix<double> b = make_matrix({{1}, {2}});

    EXPECT_THROW(hadamard(a, b), std::runtime_error);
}

TEST(MatrixAdditionTest, DimensionMismatchThrows) {
    const Matrix<double> a = make_matrix({{1, 2}});
    const Matrix<double> b = make_matrix({{1, 2, 3}});

    EXPECT_THROW(static_cast<void>(a + b), std::runtime_error);
}
