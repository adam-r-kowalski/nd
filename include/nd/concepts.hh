#pragma once

#include <algorithm>
#include <iterator>
#include <type_traits>

#include <std/concepts.hh>

namespace nd {

inline namespace v0 {

// clang-format off
template <class C,
          class iterator = typename C::iterator,
          class const_iterator = typename C::const_iterator,
          class size_type = typename C::size_type>
concept container =
  std::forward_iterator<iterator> &&
  std::forward_iterator<const_iterator> &&
  requires(C a, const C ca, C b) {
    { C() } -> std::same_as<C>;
    { C(a) } -> std::same_as<C>;
    { a = b } -> std::same_as<C&>;
    { a.~C() } -> std::same_as<void>;
    { a.begin() } -> std::same_as<iterator>;
    { ca.begin() } -> std::same_as<const_iterator>;
    { a.end() } -> std::same_as<iterator>;
    { ca.end() } -> std::same_as<const_iterator>;
    { a.cbegin() } -> std::same_as<const_iterator>;
    { a.cend() } -> std::same_as<const_iterator>;
    { a == b } -> std::convertible_to<bool>;
    { a != b } -> std::convertible_to<bool>;
    { a.swap(b) };
    { swap(a, b) };
    { a.size() } -> std::same_as<size_type>;
    { a.max_size() } -> std::same_as<size_type>;
    { a.empty() } -> std::convertible_to<bool>;
  };
// clang-format on

// clang-format off
template <class C,
          class reverse_iterator = typename C::reverse_iterator,
          class const_reverse_iterator = typename C::const_reverse_iterator>
concept reversible_container =
  container<C> &&
  requires(C a, const C ca) {
    { a.rbegin() } -> std::same_as<reverse_iterator>; 
    { ca.rbegin() } -> std::same_as<const_reverse_iterator>; 
    { a.rend() } -> std::same_as<reverse_iterator>; 
    { ca.rend() } -> std::same_as<const_reverse_iterator>; 
    { a.crbegin() } -> std::same_as<const_reverse_iterator>; 
    { a.crend() } -> std::same_as<const_reverse_iterator>; 
  };
// clang-format on

// clang-format off
template <class T>
concept tensor =
  reversible_container<T> &&
  requires(T t, const T ct, std::array<int, T::rank> cartesian_index) {
    { std::apply(t, cartesian_index) } -> std::same_as<typename T::reference>;
    { std::apply(ct, cartesian_index) } -> std::same_as<typename T::const_reference>;
  };
// clang-format on

} // namespace v0

} // namespace nd
