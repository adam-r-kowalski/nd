#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <sstream>

#include <nd/core.hh>

TEST_CASE("tensors can be indexed") {
  auto a = nd::tensor<int, 2, 3>{};
  REQUIRE(a(0, 0) == 0);
  REQUIRE(a(0, 1) == 0);
  REQUIRE(a(0, 2) == 0);
  REQUIRE(a(1, 0) == 0);
  REQUIRE(a(1, 1) == 0);
  REQUIRE(a(1, 2) == 0);

  a(0, 0) = 1;
  a(0, 1) = 2;
  a(0, 2) = 3;
  a(1, 0) = 4;
  a(1, 1) = 5;
  a(1, 2) = 6;

  REQUIRE(a(0, 0) == 1);
  REQUIRE(a(0, 1) == 2);
  REQUIRE(a(0, 2) == 3);
  REQUIRE(a(1, 0) == 4);
  REQUIRE(a(1, 1) == 5);
  REQUIRE(a(1, 2) == 6);
}
