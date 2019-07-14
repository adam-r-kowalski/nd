#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include <nd/core.hh>

TEST_CASE("arrays match the Array concept") {
  using A = nd::array<int, nd::row_major, 3, 5>;
  static_assert(nd::Array<A>);
}

TEST_CASE("arrays match the MutableArray concept") {
  using A = nd::array<int, nd::row_major, 3, 5>;
  static_assert(nd::MutableArray<A>);
}

TEST_CASE("arrays have a compile and runtime shape") {
  using A = nd::array<int, nd::row_major, 3, 5>;
  auto t = A{};
  static_assert(nd::shape_v<A> == std::array{3, 5});
  REQUIRE(t.shape() == std::array{3, 5});
}

TEST_CASE("arrays have a value type") {
  using A = nd::array<int, nd::row_major, 3, 5>;
  auto t = A{};
  static_assert(std::is_same_v<A::value_type, int>);
}

TEST_CASE("arrays have a reference type") {
  using A = nd::array<int, nd::row_major, 3, 5>;
  auto t = A{};
  static_assert(std::is_same_v<A::reference, int &>);
}

TEST_CASE("arrays have a const_reference type") {
  using A = nd::array<int, nd::row_major, 3, 5>;
  auto t = A{};
  static_assert(std::is_same_v<A::const_reference, const int &>);
}

TEST_CASE("arrays can be indexed") {
  auto t = nd::array<int, nd::row_major, 3, 5>{};
  t(1, 2) = 3;
  t(2, 4) = 5;
  REQUIRE(t(1, 2) == 3);
  REQUIRE(t(2, 4) == 5);
}

TEST_CASE("arrays default construct their elements") {
  auto t = nd::array<int, nd::row_major, 3, 5>{};
  REQUIRE(t(1, 2) == 0);
}

TEST_CASE("arrays can be constructed with column major layout") {
  using A = nd::array<int, nd::column_major, 3, 5>;
  auto t = A{};
  static_assert(nd::Array<A>);
  static_assert(nd::MutableArray<A>);
  REQUIRE(t.shape() == std::array{3, 5});
}

TEST_CASE("row major matches Layout concept") {
  using L = nd::row_major<std::array<int, 2>>;
  static_assert(nd::Layout<L>);
}

TEST_CASE("row major has a stride") {
  auto l = nd::row_major{std::array{2, 3, 4}};
  REQUIRE(l.stride() == std::array{12, 4, 1});
}

TEST_CASE("row major maps cartesian to linear index") {
  auto l = nd::row_major{std::array{2, 3, 4}};
  REQUIRE(l.linear_index({0, 0, 0}) == 0);
  REQUIRE(l.linear_index({0, 0, 1}) == 1);
  REQUIRE(l.linear_index({0, 0, 2}) == 2);
  REQUIRE(l.linear_index({0, 0, 3}) == 3);
  REQUIRE(l.linear_index({0, 1, 0}) == 4);
  REQUIRE(l.linear_index({0, 1, 1}) == 5);
  REQUIRE(l.linear_index({0, 1, 2}) == 6);
  REQUIRE(l.linear_index({0, 1, 3}) == 7);
  REQUIRE(l.linear_index({0, 2, 0}) == 8);
  REQUIRE(l.linear_index({0, 2, 1}) == 9);
  REQUIRE(l.linear_index({0, 2, 2}) == 10);
  REQUIRE(l.linear_index({0, 2, 3}) == 11);
  REQUIRE(l.linear_index({1, 0, 0}) == 12);
  REQUIRE(l.linear_index({1, 0, 1}) == 13);
  REQUIRE(l.linear_index({1, 0, 2}) == 14);
  REQUIRE(l.linear_index({1, 0, 3}) == 15);
  REQUIRE(l.linear_index({1, 1, 0}) == 16);
  REQUIRE(l.linear_index({1, 1, 1}) == 17);
  REQUIRE(l.linear_index({1, 1, 2}) == 18);
  REQUIRE(l.linear_index({1, 1, 3}) == 19);
  REQUIRE(l.linear_index({1, 2, 0}) == 20);
  REQUIRE(l.linear_index({1, 2, 1}) == 21);
  REQUIRE(l.linear_index({1, 2, 2}) == 22);
  REQUIRE(l.linear_index({1, 2, 3}) == 23);
}

TEST_CASE("column major matches Layout concept") {
  using L = nd::column_major<std::array<int, 2>>;
  static_assert(nd::Layout<L>);
}

TEST_CASE("column major has a stride") {
  auto l = nd::column_major{std::array{2, 3, 4}};
  REQUIRE(l.stride() == std::array{1, 2, 6});
}

TEST_CASE("column major maps cartesian to linear index") {
  auto l = nd::column_major{std::array{2, 3, 4}};
  REQUIRE(l.linear_index({0, 0, 0}) == 0);
  REQUIRE(l.linear_index({1, 0, 0}) == 1);
  REQUIRE(l.linear_index({0, 1, 0}) == 2);
  REQUIRE(l.linear_index({1, 1, 0}) == 3);
  REQUIRE(l.linear_index({0, 2, 0}) == 4);
  REQUIRE(l.linear_index({1, 2, 0}) == 5);
  REQUIRE(l.linear_index({0, 0, 1}) == 6);
  REQUIRE(l.linear_index({1, 0, 1}) == 7);
  REQUIRE(l.linear_index({0, 1, 1}) == 8);
  REQUIRE(l.linear_index({1, 1, 1}) == 9);
  REQUIRE(l.linear_index({0, 2, 1}) == 10);
  REQUIRE(l.linear_index({1, 2, 1}) == 11);
  REQUIRE(l.linear_index({0, 0, 2}) == 12);
  REQUIRE(l.linear_index({1, 0, 2}) == 13);
  REQUIRE(l.linear_index({0, 1, 2}) == 14);
  REQUIRE(l.linear_index({1, 1, 2}) == 15);
  REQUIRE(l.linear_index({0, 2, 2}) == 16);
  REQUIRE(l.linear_index({1, 2, 2}) == 17);
  REQUIRE(l.linear_index({0, 0, 3}) == 18);
  REQUIRE(l.linear_index({1, 0, 3}) == 19);
  REQUIRE(l.linear_index({0, 1, 3}) == 20);
  REQUIRE(l.linear_index({1, 1, 3}) == 21);
  REQUIRE(l.linear_index({0, 2, 3}) == 22);
  REQUIRE(l.linear_index({1, 2, 3}) == 23);
}
