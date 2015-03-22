#include "../utils.h"
#include <gtest/gtest.h>
using namespace std;

int sq(int i) { return i * i; }

TEST(MyTestSuite, Dudulu)
{
    EXPECT_EQ(1+2, 3) << "cooooo";
}

TEST(MyTestSuite, Cuda)
{
}
