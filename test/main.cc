#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <nd/core.hh>

TEST_CASE("tensors match the Container concept") {
  using T = nd::tensor<int, 1, 2, 3>;
  static_assert(nd::Container<T>);
}

TEST_CASE("tensors can be indexed") {
  auto t = nd::tensor<int, 1, 2, 3>{};
  REQUIRE(t(0, 0, 0) == 0);
  REQUIRE(t(0, 0, 1) == 0);
  REQUIRE(t(0, 0, 2) == 0);
  REQUIRE(t(0, 1, 0) == 0);
  REQUIRE(t(0, 1, 1) == 0);
  REQUIRE(t(0, 1, 2) == 0);

  t(0, 0, 0) = 1;
  t(0, 0, 1) = 2;
  t(0, 0, 2) = 3;
  t(0, 1, 0) = 4;
  t(0, 1, 1) = 5;
  t(0, 1, 2) = 6;

  REQUIRE(t(0, 0, 0) == 1);
  REQUIRE(t(0, 0, 1) == 2);
  REQUIRE(t(0, 0, 2) == 3);
  REQUIRE(t(0, 1, 0) == 4);
  REQUIRE(t(0, 1, 1) == 5);
  REQUIRE(t(0, 1, 2) == 6);
}

TEST_CASE("tensors have a size") {
  auto t = nd::tensor<int, 1, 2, 3>{};
  REQUIRE(t.size() == 6);
}

TEST_CASE("tensors have a max size") {
  auto t = nd::tensor<int, 1, 2, 3>{};
  REQUIRE(t.max_size() == 6);
}

TEST_CASE("tensors are not empty") {
  auto t = nd::tensor<int, 1, 2, 3>{};
  REQUIRE(!t.empty());
}

TEST_CASE("tensors can be compared for equality") {
  auto a = nd::tensor<int, 1, 2, 3>{};
  auto b = nd::tensor<int, 1, 2, 3>{};

  REQUIRE(a == b);

  a(0, 0, 0) = 1;
  a(0, 0, 1) = 2;
  a(0, 0, 2) = 3;
  a(0, 1, 0) = 4;
  a(0, 1, 1) = 5;
  a(0, 1, 2) = 6;

  REQUIRE(a != b);

  b(0, 0, 0) = 1;
  b(0, 0, 1) = 2;
  b(0, 0, 2) = 3;
  b(0, 1, 0) = 4;
  b(0, 1, 1) = 5;
  b(0, 1, 2) = 6;

  REQUIRE(a == b);
}

TEST_CASE("tensors can be swapped") {
  auto a = nd::tensor<int, 1, 2, 3>{};
  auto b = nd::tensor<int, 1, 2, 3>{};

  a(0, 0, 0) = 1;
  a(0, 0, 1) = 2;
  a(0, 0, 2) = 3;
  a(0, 1, 0) = 4;
  a(0, 1, 1) = 5;
  a(0, 1, 2) = 6;

  a.swap(b);

  REQUIRE(b(0, 0, 0) == 1);
  REQUIRE(b(0, 0, 1) == 2);
  REQUIRE(b(0, 0, 2) == 3);
  REQUIRE(b(0, 1, 0) == 4);
  REQUIRE(b(0, 1, 1) == 5);
  REQUIRE(b(0, 1, 2) == 6);

  REQUIRE(a(0, 0, 0) == 0);
  REQUIRE(a(0, 0, 1) == 0);
  REQUIRE(a(0, 0, 2) == 0);
  REQUIRE(a(0, 1, 0) == 0);
  REQUIRE(a(0, 1, 1) == 0);
  REQUIRE(a(0, 1, 2) == 0);
}
