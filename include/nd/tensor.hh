#pragma once

#include <algorithm>
#include <array>
#include <iterator>
#include <numeric>
#include <vector>

namespace nd {

inline namespace v0 {

template <class T, int... Ns> struct basic_tensor {
  using storage_type = std::vector<T>;
  using value_type = typename storage_type::value_type;
  using reference = typename storage_type::reference;
  using const_reference = typename storage_type::const_reference;
  using iterator = typename storage_type::iterator;
  using const_iterator = typename storage_type::const_iterator;
  using reverse_iterator = typename storage_type::reverse_iterator;
  using const_reverse_iterator = typename storage_type::const_reverse_iterator;
  using difference_type = typename storage_type::difference_type;
  using size_type = typename storage_type::size_type;
  constexpr static int rank = sizeof...(Ns);
  using shape_type = std::array<int, rank>;

  basic_tensor() : size_((Ns * ...)), storage_{storage_type(size_)}, shape_{Ns...} {
    stride_[shape_.size() - 1] = 1;
    std::partial_sum(shape_.crbegin(), shape_.crend() - 1, stride_.rbegin() + 1,
                     std::multiplies{});
  }

  auto begin() -> iterator { return storage_.begin(); }
  auto begin() const -> const_iterator { return storage_.begin(); }
  auto end() -> iterator { return storage_.end(); }
  auto end() const -> const_iterator { return storage_.end(); }
  auto cbegin() const -> const_iterator { return storage_.cbegin(); }
  auto cend() const -> const_iterator { return storage_.cend(); }
  auto rbegin() -> reverse_iterator { return storage_.rbegin(); }
  auto rbegin() const -> const_reverse_iterator { return storage_.rbegin(); }
  auto rend() -> reverse_iterator { return storage_.rend(); }
  auto rend() const -> const_reverse_iterator { return storage_.rend(); }
  auto crbegin() const -> const_reverse_iterator { return storage_.crbegin(); }
  auto crend() const -> const_reverse_iterator { return storage_.crend(); }
  auto swap(basic_tensor &other) -> void { storage_.swap(other.storage_); }
  auto size() const -> size_type { return size_; }
  auto max_size() const -> size_type { return size_; }
  auto empty() const -> bool { return begin() == end(); }

  template <class... Indices>
  auto operator()(Indices... indices) const -> const_reference {
    return storage_[linear_index(std::forward<Indices>(indices)...)];
  }

  template <class... Indices> auto operator()(Indices... indices) -> reference {
    return storage_[linear_index(std::forward<Indices>(indices)...)];
  }

private:
  template <class... Indices>
  auto linear_index(Indices... indices) const -> int {
    static_assert(sizeof...(indices) == rank);
    auto cartesian_index = std::array{indices...};
    return std::transform_reduce(stride_.begin(), stride_.end(),
                                 cartesian_index.begin(), 0);
  }

  size_type size_;
  storage_type storage_;
  shape_type shape_;
  shape_type stride_;
};

template <class T, int... Ns>
auto operator==(const basic_tensor<T, Ns...> &a, const basic_tensor<T, Ns...> &b) {
  return std::equal(a.cbegin(), a.cend(), b.cbegin());
}

template <class T, int... Ns>
auto operator!=(const basic_tensor<T, Ns...> &a, const basic_tensor<T, Ns...> &b) {
  return !(a == b);
}

template <class T, int... Ns>
auto swap(basic_tensor<T, Ns...> &a, basic_tensor<T, Ns...> &b) -> void {
  a.swap(b);
}

} // namespace v0

} // namespace nd
