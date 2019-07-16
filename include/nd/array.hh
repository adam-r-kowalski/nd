#pragma once

#include <algorithm>
#include <array>
#include <numeric>
#include <type_traits>
#include <vector>

#include <nd/traits.hh>

namespace nd {

inline namespace v0 {

// clang-format off
template <class A,
          class shape_t = traits::shape_t<A>,
          class const_reference_t = traits::const_reference_t<A>,
          class const_iterator_t = traits::const_iterator_t<A>>
concept Array = requires(const A const_array, shape_t cartesian_index) {
  { shape(const_array) } -> const shape_t &;
  { std::apply(const_array, cartesian_index) } -> const_reference_t;
  { std::begin(const_array) } -> const_iterator_t;
  { std::end(const_array) } -> const_iterator_t;
  { std::rbegin(const_array) } -> const_iterator_t;
  { std::rend(const_array) } -> const_iterator_t;
  { std::cbegin(const_array) } -> const_iterator_t;
  { std::cend(const_array) } -> const_iterator_t;
  { std::crbegin(const_array) } -> const_iterator_t;
  { std::crend(const_array) } -> const_iterator_t;
};
// clang-format on

// clang-format off
template <class A,
          class shape_t = traits::shape_t<A>,
          class reference_t = traits::reference_t<A>,
          class iterator_t = traits::iterator_t<A>>
concept MutableArray = requires(A array, shape_t cartesian_index) {
  requires Array<A>;
  { std::apply(array, cartesian_index) } -> reference_t;
  { std::begin(array) } -> iterator_t;
  { std::end(array) } -> iterator_t;
  { std::rbegin(array) } -> iterator_t;
  { std::rend(array) } -> iterator_t;
};
// clang-format on

// clang-format off
template <class L,
          class shape_t = traits::shape_t<L>,
          class value_t = traits::value_t<shape_t>>
concept Layout = requires(L layout, shape_t array_size, shape_t cartesian_index) {
  { L{array_size} };
  { layout.linear_index(cartesian_index) } -> value_t;
};
// clang-format on

template <class Shape> struct row_major {
  using shape_t = Shape;
  using value_t = traits::value_t<shape_t>;

  explicit row_major(const shape_t &shape) {
    stride_[shape.size() - 1] = 1;
    std::partial_sum(rbegin(shape), rend(shape) - 1, rbegin(stride_) + 1,
                     std::multiplies{});
  }

  auto linear_index(const shape_t &cartesian_index) const -> value_t {
    return std::transform_reduce(begin(stride_), end(stride_),
                                 begin(cartesian_index), 0);
  }

  auto stride() const -> const shape_t & { return stride_; }

private:
  shape_t stride_;
};

template <class Shape> struct column_major {
  using shape_t = Shape;
  using value_t = traits::value_t<shape_t>;

  explicit column_major(const shape_t &shape) {
    stride_[0] = 1;
    std::partial_sum(begin(shape), end(shape) - 1, begin(stride_) + 1,
                     std::multiplies{});
  }

  auto linear_index(const shape_t &cartesian_index) const -> value_t {
    return std::transform_reduce(begin(stride_), end(stride_),
                                 begin(cartesian_index), 0);
  }

  auto stride() const -> const shape_t & { return stride_; }

private:
  shape_t stride_;
};

// clang-format off
template <class D>
concept Dimensions = requires() {
  { D::shape } -> typename D::shape_t;
  { D::size } -> int;
  { D::rank } -> int;
};
// clang-format on

template <int... Ds> struct d {
  using shape_t = std::array<int, sizeof...(Ds)>;
  constexpr static shape_t shape = shape_t{Ds...};
  constexpr static int size = (Ds * ...);
  constexpr static int rank = sizeof...(Ds);
};

// clang-format off
template <class S, class shape_t = std::array<int, 2>>
concept Specification = Layout<typename S::template layout_type<shape_t>>;
// clang-format on

struct default_specification {
  template <class shape_t> using layout_type = row_major<shape_t>;
};

template <template <class> class... Policies> struct p {
  template <class shape_t> using layout_type = column_major<shape_t>;
};

template <class T,
          Dimensions D,
          Specification S = default_specification>
struct array {
  using value_t = T;
  using reference_t = value_t &;
  using const_reference_t = const value_t &;
  using storage_t = std::vector<value_t>;
  using shape_t = typename D::shape_t;
  using iterator_t = typename storage_t::iterator;
  using const_iterator_t = typename storage_t::const_iterator;
  using layout_t = typename S::template layout_type<shape_t>;

  array()
      : storage_{storage_t(D::size)}, shape_{D::shape}, layout_{shape_} {}

  auto shape() const -> const shape_t & { return shape_; }

  template <class... Indices>
  auto operator()(Indices... indices) const -> const_reference_t {
    static_assert(sizeof...(Indices) == D::rank);
    return storage_[layout_.linear_index(std::array{indices...})];
  }

  template <class... Indices> auto operator()(Indices... indices) -> reference_t {
    static_assert(sizeof...(Indices) == D::rank);
    return storage_[layout_.linear_index(std::array{indices...})];
  }

  auto begin() -> iterator_t { return storage_.begin(); }
  auto begin() const -> const_iterator_t { return storage_.begin(); }
  auto end() -> iterator_t { return storage_.end(); }
  auto end() const -> const_iterator_t { return storage_.end(); }
  auto rbegin() -> iterator_t { return storage_.rbegin(); }
  auto rbegin() const -> const_iterator_t { return storage_.rbegin(); }
  auto rend() -> iterator_t { return storage_.rend(); }
  auto rend() const -> const_iterator_t { return storage_.rend(); }
  auto cbegin() const -> const_iterator_t { return storage_.cbegin(); }
  auto cend() const -> const_iterator_t { return storage_.cend(); }
  auto crbegin() const -> const_iterator_t { return storage_.crbegin(); }
  auto crend() const -> const_iterator_t { return storage_.crend(); }

private:
  storage_t storage_;
  shape_t shape_;
  layout_t layout_;
};

template <class A> auto shape(const A &a) -> typename A::shape_t {
  return a.shape();
}

template <Array A> auto operator-(const A &a) -> A {
  auto a2 = A{};
  std::transform(a.cbegin(), a.cend(), a2.begin(), std::negate<>{});
  return a2;
}

template <Array A> auto operator==(const A &a, const A &a2) -> bool {
  return std::equal(a.cbegin(), a.cend(), a2.cbegin());
}

template <Array A> auto operator!=(const A &a, const A &a2) -> bool {
  return !std::equal(a.cbegin(), a.cend(), a2.cbegin());
}

template <Array A> auto operator+(const A &a, const A &a2) -> A {
  auto a3 = A{};
  std::transform(a.cbegin(), a.cend(), a2.cbegin(), a3.begin(), std::plus<>{});
  return a3;
}

template <Array A> auto operator-(const A &a, const A &a2) -> A {
  auto a3 = A{};
  std::transform(a.cbegin(), a.cend(), a2.cbegin(), a3.begin(), std::minus<>{});
  return a3;
}

} // namespace v0

} // namespace nd
