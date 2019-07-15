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

template <class T, Layout L, int... Dimensions> struct array {
  using value_type = T;
  using reference = value_type &;
  using const_reference = const value_type &;
  using data_type = std::vector<value_type>;
  using shape_type = std::array<int, sizeof...(Dimensions)>;
  using iterator = typename data_type::iterator;
  using const_iterator = typename data_type::const_iterator;
  using layout_type = L;

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

  auto begin() -> iterator { return data_.begin(); }
  auto begin() const -> const_iterator { return data_.begin(); }
  auto end() -> iterator { return data_.end(); }
  auto end() const -> const_iterator { return data_.end(); }
  auto rbegin() -> iterator { return iterator{}; }
  auto rbegin() const -> const_iterator { return data_.rbegin(); }
  auto rend() -> iterator { return iterator{}; }
  auto rend() const -> const_iterator { return data_.rend(); }
  auto cbegin() const -> const_iterator { return data_.cbegin(); }
  auto cend() const -> const_iterator { return data_.cend(); }
  auto crbegin() const -> const_iterator { return data_.crbegin(); }
  auto crend() const -> const_iterator { return data_.crend(); }

private:
  data_type data_;
  shape_type shape_;
  L layout_;
};

template <class T, class Specification, int... Dimensions, class... Arguments>
auto make_array(Arguments &&... arguments) {
  using layout_type = typename Specification::template layout_type<
      std::array<int, sizeof...(Dimensions)>>;
  return array<T, layout_type, Dimensions...>{
      std::forward<Arguments>(arguments)...};
}

template <class T, int... Dimensions, class... Arguments>
auto make_array(Arguments &&... arguments) {
  using layout_type = row_major<std::array<int, sizeof...(Dimensions)>>;
  return array<T, layout_type, Dimensions...>{
      std::forward<Arguments>(arguments)...};
}

template <class T, Layout L, int... Dimensions>
struct shape<array<T, L, Dimensions...>> {
  using shape_type = typename array<T, L, Dimensions...>::shape_type;
  constexpr static shape_type value = shape_type{Dimensions...};
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
