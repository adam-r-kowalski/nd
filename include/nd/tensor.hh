#pragma once

#include <algorithm>
#include <array>
#include <numeric>
#include <type_traits>
#include <vector>

namespace nd {

inline namespace v0 {

template <class T, size_t... Dimensions> struct tensor {
  tensor() : storage_{std::vector<T>(size)}, shape_{Dimensions...} {
    stride_[rank - 1] = 1;
    std::partial_sum(rbegin(shape_), rend(shape_) - 1, rbegin(stride_) + 1,
                     std::multiplies{});
  }

  template <class... Indices>
  auto operator()(Indices... indicies) const -> const T & {
    static_assert(sizeof...(indicies) == rank);
    auto cartesian_index = std::array{indicies...};
    auto linear_index = std::transform_reduce(begin(stride_), end(stride_),
                                              begin(cartesian_index), 0);
    return storage_[linear_index];
  }

  template <class... Indices> auto operator()(Indices... indicies) -> T & {
    static_assert(sizeof...(indicies) == rank);
    auto cartesian_index = std::array{indicies...};
    auto linear_index = std::transform_reduce(begin(stride_), end(stride_),
                                              begin(cartesian_index), 0);
    return storage_[linear_index];
  }

private:
  constexpr static int size = (Dimensions * ...);
  constexpr static int rank = sizeof...(Dimensions);
  using shape = std::array<int, rank>;

  std::vector<T> storage_;
  shape shape_;
  shape stride_;
};

} // namespace v0

} // namespace nd
