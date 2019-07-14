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
template <class A, class Shape = typename A::shape_type>
concept Array = requires(const A const_array, Shape shape) {
  { const_array.shape() } -> const Shape &;
  { shape_v<A> } -> Shape;
  { std::apply(const_array, shape) } -> typename A::const_reference;
};
// clang-format on

// clang-format off
template <class A, class Shape = typename A::shape_type>
concept MutableArray = requires(A array, Shape shape) {
  requires Array<A>;
  { std::apply(array, shape) } -> typename A::reference;
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

template <class T, template <class> class L, int... Dimensions> struct array {
  using value_type = T;
  using reference = value_type &;
  using const_reference = const value_type &;
  using shape_type = std::array<int, sizeof...(Dimensions)>;

  static_assert(Layout<L<shape_type>>);

  array()
      : data_{std::vector<value_type>((Dimensions * ...))},
        shape_{Dimensions...}, layout_{shape_} {}

  auto shape() const -> const shape_type & { return shape_; }

  template <class... Indices>
  auto operator()(Indices... indices) const -> const_reference {
    static_assert(sizeof...(Indices) == sizeof...(Dimensions));
    return data_[layout_.linear_index(std::array{indices...})];
  }

  template <class... Indices> auto operator()(Indices... indices) -> reference {
    static_assert(sizeof...(Indices) == sizeof...(Dimensions));
    return data_[layout_.linear_index(std::array{indices...})];
  }

private:
  std::vector<value_type> data_;
  shape_type shape_;
  L<shape_type> layout_;
};

template <class T, template <class> class Layout, int... Dimensions>
struct shape<array<T, Layout, Dimensions...>> {
  using shape_type = typename array<T, Layout, Dimensions...>::shape_type;
  constexpr static shape_type value = shape_type{Dimensions...};
};

} // namespace v0

} // namespace nd