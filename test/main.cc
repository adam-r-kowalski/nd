#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <sstream>

#include <nd/core.hh>

TEST_CASE("arrays can be constructed with default specification") {
  auto a = nd::make_array<int, 3, 5>();
  using A = decltype(a);
  static_assert(nd::Array<decltype(a)>);
  static_assert(nd::MutableArray<decltype(a)>);
  static_assert(nd::Same<A::layout_type, nd::row_major<A::shape_type>>);
}

struct spec {
  template <class Shape> using layout_type = nd::column_major<Shape>;
};

TEST_CASE("arrays can be constructed with custom specification") {
  auto a = nd::make_array<int, spec, 3, 5>();
  using A = decltype(a);
  static_assert(nd::Array<A>);
  static_assert(nd::MutableArray<A>);
  static_assert(nd::Same<A::layout_type, nd::column_major<A::shape_type>>);
}

TEST_CASE("arrays have a compile and runtime shape") {
  auto a = nd::make_array<int, 3, 5>();
  static_assert(nd::shape_v<decltype(a)> == std::array{3, 5});
  REQUIRE(a.shape() == std::array{3, 5});
}

TEST_CASE("arrays have a value type") {
  auto a = nd::make_array<int, 3, 5>();
  static_assert(nd::Same<decltype(a)::value_type, int>);
}

TEST_CASE("arrays have a reference type") {
  auto a = nd::make_array<int, 3, 5>();
  static_assert(nd::Same<decltype(a)::reference, int &>);
}

TEST_CASE("arrays have a const_reference type") {
  auto a = nd::make_array<int, 3, 5>();
  static_assert(nd::Same<decltype(a)::const_reference, const int &>);
}

TEST_CASE("arrays can be indexed") {
  auto a = nd::make_array<int, 3, 5>();
  a(1, 2) = 3;
  a(2, 4) = 5;
  REQUIRE(a(1, 2) == 3);
  REQUIRE(a(2, 4) == 5);
}

TEST_CASE("arrays default construct their elements") {
  auto a = nd::make_array<int, 3, 5>();
  REQUIRE(a(1, 2) == 0);
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

TEST_CASE("row major arrays can iterated in row order") {
  auto a = nd::make_array<int, 2, 3>();
  a(0, 0) = 1;
  a(0, 1) = 2;
  a(0, 2) = 3;
  a(1, 0) = 4;
  a(1, 1) = 5;
  a(1, 2) = 6;
  auto ss = std::stringstream{};
  std::copy(a.cbegin(), a.cend(), std::ostream_iterator<int>(ss, ""));
  REQUIRE(ss.str() == "123456");
}

TEST_CASE("column major arrays can iterated in column order") {
  auto a = nd::make_array<int, spec, 2, 3>();
  a(0, 0) = 1;
  a(0, 1) = 2;
  a(0, 2) = 3;
  a(1, 0) = 4;
  a(1, 1) = 5;
  a(1, 2) = 6;
  auto ss = std::stringstream{};
  std::copy(a.cbegin(), a.cend(), std::ostream_iterator<int>(ss, ""));
  REQUIRE(ss.str() == "142536");
}

TEST_CASE("arrays can be copy constructed") {
  auto a = nd::make_array<int, 2, 3>();
  a(0, 0) = 1;
  a(0, 1) = 2;
  a(0, 2) = 3;
  a(1, 0) = 4;
  a(1, 1) = 5;
  a(1, 2) = 6;
  auto a2 = a;

  auto ss1 = std::stringstream{};
  std::copy(a.cbegin(), a.cend(), std::ostream_iterator<int>(ss1, ""));
  REQUIRE(ss1.str() == "123456");

  auto ss2 = std::stringstream{};
  std::copy(a2.cbegin(), a2.cend(), std::ostream_iterator<int>(ss2, ""));
  REQUIRE(ss2.str() == "123456");
}

TEST_CASE("arrays can be move constructed") {
  auto a = nd::make_array<int, 2, 3>();
  a(0, 0) = 1;
  a(0, 1) = 2;
  a(0, 2) = 3;
  a(1, 0) = 4;
  a(1, 1) = 5;
  a(1, 2) = 6;
  auto a2 = std::move(a);

  auto ss1 = std::stringstream{};
  std::copy(a.cbegin(), a.cend(), std::ostream_iterator<int>(ss1, ""));
  REQUIRE(ss1.str() == "");

  auto ss2 = std::stringstream{};
  std::copy(a2.cbegin(), a2.cend(), std::ostream_iterator<int>(ss2, ""));
  REQUIRE(ss2.str() == "123456");
}

TEST_CASE("arrays can be copy assigned") {
  auto a = nd::make_array<int, 2, 3>();
  a(0, 0) = 1;
  a(0, 1) = 2;
  a(0, 2) = 3;
  a(1, 0) = 4;
  a(1, 1) = 5;
  a(1, 2) = 6;

  auto a2 = nd::make_array<int, 2, 3>();
  a2 = a;

  auto ss1 = std::stringstream{};
  std::copy(a.cbegin(), a.cend(), std::ostream_iterator<int>(ss1, ""));
  REQUIRE(ss1.str() == "123456");

  auto ss2 = std::stringstream{};
  std::copy(a2.cbegin(), a2.cend(), std::ostream_iterator<int>(ss2, ""));
  REQUIRE(ss2.str() == "123456");
}

TEST_CASE("arrays can be move assigned") {
  auto a = nd::make_array<int, 2, 3>();
  a(0, 0) = 1;
  a(0, 1) = 2;
  a(0, 2) = 3;
  a(1, 0) = 4;
  a(1, 1) = 5;
  a(1, 2) = 6;

  auto a2 = nd::make_array<int, 2, 3>();
  a2 = std::move(a);

  auto ss1 = std::stringstream{};
  std::copy(a.cbegin(), a.cend(), std::ostream_iterator<int>(ss1, ""));
  REQUIRE(ss1.str() == "");

  auto ss2 = std::stringstream{};
  std::copy(a2.cbegin(), a2.cend(), std::ostream_iterator<int>(ss2, ""));
  REQUIRE(ss2.str() == "123456");
}

TEST_CASE("arrays can be compared for equality") {
  auto a = nd::make_array<int, 2, 3>();
  a(0, 0) = 1;
  a(0, 1) = 2;
  a(0, 2) = 3;
  a(1, 0) = 4;
  a(1, 1) = 5;
  a(1, 2) = 6;

  auto a2 = nd::make_array<int, 2, 3>();
  a2(0, 0) = 1;
  a2(0, 1) = 2;
  a2(0, 2) = 3;
  a2(1, 0) = 4;
  a2(1, 1) = 5;
  a2(1, 2) = 6;

  REQUIRE(a == a2);
}

TEST_CASE("arrays can be compared for inequality") {
  auto a = nd::make_array<int, 2, 3>();
  a(0, 0) = 1;
  a(0, 1) = 2;
  a(0, 2) = 3;
  a(1, 0) = 4;
  a(1, 1) = 5;
  a(1, 2) = 6;

  auto a2 = nd::make_array<int, 2, 3>();
  a2(0, 0) = 3;
  a2(0, 1) = 3;
  a2(0, 2) = 3;
  a2(1, 0) = 3;
  a2(1, 1) = 3;
  a2(1, 2) = 3;

  REQUIRE(a != a2);
}

TEST_CASE("arrays can be negated") {
  auto a = nd::make_array<int, 2, 3>();
  a(0, 0) = 1;
  a(0, 1) = 2;
  a(0, 2) = 3;
  a(1, 0) = 4;
  a(1, 1) = 5;
  a(1, 2) = 6;

  auto a2 = nd::make_array<int, 2, 3>();
  a2(0, 0) = -1;
  a2(0, 1) = -2;
  a2(0, 2) = -3;
  a2(1, 0) = -4;
  a2(1, 1) = -5;
  a2(1, 2) = -6;

  REQUIRE(-a == a2);
}

TEST_CASE("arrays can be added") {
  auto a = nd::make_array<int, 2, 3>();
  a(0, 0) = 1;
  a(0, 1) = 2;
  a(0, 2) = 3;
  a(1, 0) = 4;
  a(1, 1) = 5;
  a(1, 2) = 6;

  auto a2 = a;

  auto a3 = nd::make_array<int, 2, 3>();
  a3(0, 0) = 2;
  a3(0, 1) = 4;
  a3(0, 2) = 6;
  a3(1, 0) = 8;
  a3(1, 1) = 10;
  a3(1, 2) = 12;

  REQUIRE(a + a2 == a3);
}

TEST_CASE("arrays can be subtracted") {
  auto a = nd::make_array<int, 2, 3>();
  a(0, 0) = 2;
  a(0, 1) = 4;
  a(0, 2) = 6;
  a(1, 0) = 8;
  a(1, 1) = 10;
  a(1, 2) = 12;

  auto a2 = nd::make_array<int, 2, 3>();;
  a2(0, 0) = 1;
  a2(0, 1) = 2;
  a2(0, 2) = 3;
  a2(1, 0) = 4;
  a2(1, 1) = 5;
  a2(1, 2) = 6;

  auto a3 = nd::make_array<int, 2, 3>();
  a3(0, 0) = 1;
  a3(0, 1) = 2;
  a3(0, 2) = 3;
  a3(1, 0) = 4;
  a3(1, 1) = 5;
  a3(1, 2) = 6;

  REQUIRE(a - a2 == a3);
}

