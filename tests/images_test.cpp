#include <gtest/gtest.h>

TEST(MultiFileTests, SimpleTest) { EXPECT_EQ(2, 1 + 1); }

TEST(MultiFileTests, FailedTest) { EXPECT_EQ(2, 1); }
