#include "../src/utils.h"
#include <gtest/gtest.h>
using namespace std;

TEST(MoreTest, Dummy1)
{
    EXPECT_EQ("asdf", string("as") + "df");
}

TEST(MoreTest, Dummy2)
{
    vector<int> duh {5, 4, 3, 1, 2};
    EXPECT_GT(duh.size(), 3);
}
