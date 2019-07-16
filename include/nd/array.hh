#pragma once

#include <algorithm>
#include <array>
#include <numeric>
#include <type_traits>
#include <vector>

#include <iostream>

namespace nd {

inline namespace v0 {

template <class T> struct shape;

template <class T>
constexpr static typename T::shape_type shape_v = shape<T>::value;

// clang-format off
template <class A,
          class Shape = typename A::shape_type,
          class Iterator = typename A::const_iterator>
concept Array = requires(const A const_array, Shape shape) {
  { const_array.shape() } -> const Shape &;
  { shape_v<A> } -> Shape;
  { std::apply(const_array, shape) } -> typename A::const_reference;
  { const_array.begin() } -> Iterator;
  { const_array.end() } -> Iterator;
  { const_array.rbegin() } -> Iterator;
  { const_array.rend() } -> Iterator;
  { const_array.cbegin() } -> Iterator;
  { const_array.cend() } -> Iterator;
  { const_array.crbegin() } -> Iterator;
  { const_array.crend() } -> Iterator;
};
// clang-format on

// clang-format off
template <class A,
          class Shape = typename A::shape_type,
          class Iterator = typename A::iterator>
concept MutableArray = requires(A array, Shape shape) {
  requires Array<A>;
  { std::apply(array, shape) } -> typename A::reference;
  { array.begin() } -> Iterator;
  { array.end() } -> Iterator;
  { array.rbegin() } -> Iterator;
  { array.rend() } -> Iterator;
};
// clang-format on

// clang-format off
template <class L, class Shape = typename L::shape_type>
concept Layout = requires(L layout, Shape shape) {
  { L{shape} };
  { layout.linear_index(shape) } -> typename Shape::value_type;
};
// clang-format on

template <class Shape> struct row_major {
  using value_type = typename Shape::value_type;
  using shape_type = Shape;

  explicit row_major(const Shape &shape) {
    stride_[shape.size() - 1] = 1;
    std::partial_sum(rbegin(shape), rend(shape) - 1, rbegin(stride_) + 1,
                     std::multiplies{});
  }

  auto linear_index(const Shape &cartesian_index) const -> value_type {
    return std::transform_reduce(begin(stride_), end(stride_),
                                 begin(cartesian_index), 0);
  }

  auto stride() const -> const Shape & { return stride_; }

private:
  Shape stride_;
};

template <class Shape> struct column_major {
  using value_type = typename Shape::value_type;
  using shape_type = Shape;

  explicit column_major(const Shape &shape) {
    stride_[0] = 1;
    std::partial_sum(begin(shape), end(shape) - 1, begin(stride_) + 1,
                     std::multiplies{});
  }

  auto linear_index(const Shape &cartesian_index) const -> value_type {
    return std::transform_reduce(begin(stride_), end(stride_),
                                 begin(cartesian_index), 0);
  }

  auto stride() const -> const Shape & { return stride_; }

private:
  Shape stride_;
};

// clang-format off
template <class D>
concept Dimensions = requires() {
  { D::shape } -> typename D::shape_type;
  { D::size } -> int;
  { D::rank } -> int;
};
// clang-format on

template <int... Ds> struct d {
  using shape_type = std::array<int, sizeof...(Ds)>;
  constexpr static shape_type shape = shape_type{Ds...};
  constexpr static int size = (Ds * ...);
  constexpr static int rank = sizeof...(Ds);
};

// clang-format off
template <class S, class Shape = std::array<int, 2>>
concept Specification = Layout<typename S::template layout_type<Shape>>;
// clang-format on

struct default_specification {
  template <class Size> using layout_type = row_major<Size>;
};

template <template <class> class... Policies>
struct p {
  template <class Size> using layout_type = column_major<Size>;
};

template <class T, Dimensions D, Specification S = default_specification>
struct array {
  using value_type = T;
  using reference = value_type &;
  using const_reference = const value_type &;
  using storage_type = std::vector<value_type>;
  using shape_type = typename D::shape_type;
  using iterator = typename storage_type::iterator;
  using const_iterator = typename storage_type::const_iterator;
  using layout_type = typename S::template layout_type<shape_type>;

  array()
      : storage_{storage_type(D::size)}, shape_{D::shape}, layout_{shape_} {}

  auto shape() const -> const shape_type & { return shape_; }

  template <class... Indices>
  auto operator()(Indices... indices) const -> const_reference {
    static_assert(sizeof...(Indices) == D::rank);
    return storage_[layout_.linear_index(std::array{indices...})];
  }

  template <class... Indices> auto operator()(Indices... indices) -> reference {
    static_assert(sizeof...(Indices) == D::rank);
    return storage_[layout_.linear_index(std::array{indices...})];
  }

  auto begin() -> iterator { return storage_.begin(); }
  auto begin() const -> const_iterator { return storage_.begin(); }
  auto end() -> iterator { return storage_.end(); }
  auto end() const -> const_iterator { return storage_.end(); }
  auto rbegin() -> iterator { return iterator{}; }
  auto rbegin() const -> const_iterator { return storage_.rbegin(); }
  auto rend() -> iterator { return iterator{}; }
  auto rend() const -> const_iterator { return storage_.rend(); }
  auto cbegin() const -> const_iterator { return storage_.cbegin(); }
  auto cend() const -> const_iterator { return storage_.cend(); }
  auto crbegin() const -> const_iterator { return storage_.crbegin(); }
  auto crend() const -> const_iterator { return storage_.crend(); }

private:
  storage_type storage_;
  shape_type shape_;
  layout_type layout_;
};

template <class T, Dimensions D, Specification S> struct shape<array<T, D, S>> {
  constexpr static typename D::shape_type value = D::shape;
};

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
