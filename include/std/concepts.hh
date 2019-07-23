#pragma once

#include <algorithm>
#include <iterator>
#include <type_traits>

namespace std {

inline namespace v0 {

// clang-format off
template <class T, class U>
concept same_as = std::is_same_v<T, U> && std::is_same_v<U, T>;
// clang-format on

template <class From, class To>
concept convertible_to = std::is_convertible_v<From, To>;

template <class T> concept lvalue_reference = std::is_lvalue_reference_v<T>;

template <class T> concept destructible = std::is_nothrow_destructible_v<T>;

// clang-format off
template <class T, class... Args>
concept constructible = 
  destructible<T> && std::is_constructible_v<T, Args...>;
// clang-format on

// clang-format off
template <class T>
concept move_constructible = constructible<T, T> && convertible_to<T, T>;
// clang-format on

// clang-format off
template <class T>
concept copy_constructible =
  move_constructible<T> &&
  constructible<T, T&> &&
  convertible_to<T, T> &&
  constructible<T, const T&> &&
  convertible_to<const T&, T> &&
  constructible<T, const T> &&
  convertible_to<const T, T>;
// clang-format on

template <class T> concept object = std::is_object_v<T>;

// clang-format off
template <class LHS, class RHS>
concept assignable =
  lvalue_reference<LHS> &&
  requires(LHS lhs, RHS&& rhs) {
    { lhs = std::forward<RHS>(rhs) } -> same_as<LHS>;
  };
// clang-format on

// clang-format off
template <class T>
concept swappable =
  requires(T& a, T& b) {
    std::swap(a, b);
  };
// clang-format on

// clang-format off
template <class T>
concept movable =
  object<T> &&
  move_constructible<T> &&
  assignable<T&, T> &&
  swappable<T>;
// clang-format on

// clang-format off
template <class T>
concept copyable =
  copy_constructible<T> &&
  movable<T> &&
  assignable<T&, const T&>;
// clang-format on

// clang-format off
template <class I>
concept iterator_type =
  copyable<I> && requires(I i) {
    { *i } -> same_as<typename std::iterator_traits<I>::reference>;
    { ++i } -> convertible_to<I&>;
    { *i++ } -> convertible_to<typename std::iterator_traits<I>::value_type>;
  };
// clang-format on

// clang-format off
template <class T>
concept integral = std::is_integral_v<T>;
// clang-format on

// clang-format off
template <class T>
concept signed_integral = integral<T> && std::is_signed_v<T>;
// clang-format on

// clang-format off
template <class I>
concept input_iterator =
  iterator_type<I> &&
  signed_integral<typename std::iterator_traits<I>::difference_type> &&
  requires(I i, I j) {
    { i != j } -> convertible_to<bool>;
  };
// clang-format on

// clang-format off
template <class I,
          class reference = typename std::iterator_traits<I>::reference>
concept forward_iterator =
  input_iterator<I> &&
  constructible<I> &&
  lvalue_reference<reference> &&
  same_as<std::remove_cvref_t<reference>, typename std::iterator_traits<I>::value_type> &&
  requires(I i) {
    { i++ } -> convertible_to<const I&>;
    { *i++ } -> same_as<reference>;
  };
// clang-format on

// clang-format off
template <class I>
concept bidirectional_iterator =
  forward_iterator<I> &&
  requires(I i) {
    { --i } -> same_as<I&>;
    { i-- } -> convertible_to<const I&>;
    { *i-- } -> same_as<typename std::iterator_traits<I>::reference>;
  };
// clang-format on

// clang-format off
template <class T>
concept totally_ordered =
  // EqualityComparable<T> &&
  requires(const std::remove_reference_t<T>& a,
           const std::remove_reference_t<T>& b) {
    { a < b } -> convertible_to<bool>;
    { a > b } -> convertible_to<bool>;
    { a <= b } -> convertible_to<bool>;
    { a >= b } -> convertible_to<bool>;
  };
// clang-format on

// clang-format off
template <class I,
          class difference_type = typename std::iterator_traits<I>::difference_type>
concept random_access_iterator =
  bidirectional_iterator<I> &&
  totally_ordered<I> &&
  requires(I i, difference_type n) {
    { i += n } -> same_as<I&>;
    { i -= n } -> same_as<I&>;
    { i + n } -> same_as<I>;
    { n + i } -> same_as<I>;
    { i - n } -> same_as<I>;
    { i - i } -> same_as<difference_type>;
    { i[n] } -> convertible_to<typename std::iterator_traits<I>::reference>;
  };
// clang-format on

} // namespace v0

} // namespace std
