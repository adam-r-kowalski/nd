#pragma once

#include <algorithm>
#include <iterator>
#include <type_traits>

namespace nd {

inline namespace v0 {

// clang-format off
template <class T, class U>
concept Same = std::is_same_v<T, U> && std::is_same_v<U, T>;
// clang-format on

template <class From, class To>
concept ConvertibleTo = std::is_convertible_v<From, To>;

template <class T> concept LvalueReference = std::is_lvalue_reference_v<T>;

template <class T> concept Destructible = std::is_nothrow_destructible_v<T>;

// clang-format off
template <class T, class... Args>
concept Constructible = 
  Destructible<T> && std::is_constructible_v<T, Args...>;
// clang-format on

// clang-format off
template <class T>
concept Copyable = true;
/*
  CopyConstructible<T> &&
  Movable<T> &&
  Assignable<T&, const T&>;
*/
// clang-format on

// clang-format off
template <class I>
concept Iterator =
  Copyable<I> && requires(I i) {
    { *i };
    { ++i } -> Same<I&>;
    { *i++ };
  };
// clang-format on

// clang-format off
template <class T>
concept Integral = std::is_integral_v<T>;
// clang-format on

// clang-format off
template <class T>
concept SignedIntegral = Integral<T> && std::is_signed_v<T>;
// clang-format on

// clang-format off
template <class I>
concept InputIterator =
  Iterator<I> &&
  SignedIntegral<typename std::iterator_traits<I>::difference_type> &&
  requires(I i, I j) {
    { i != j } -> ConvertibleTo<bool>;
    { *i } -> Same<typename std::iterator_traits<I>::reference>;
    { ++i } -> ConvertibleTo<I&>;
    { *i++ } -> ConvertibleTo<typename std::iterator_traits<I>::value_type>;
  };
// clang-format on

// clang-format off
template <class I,
          class reference = typename std::iterator_traits<I>::reference>
concept ForwardIterator =
  InputIterator<I> &&
  Constructible<I> &&
  LvalueReference<reference> &&
  Same<std::remove_cvref_t<reference>, typename std::iterator_traits<I>::value_type> &&
  requires(I i) {
    { i++ } -> ConvertibleTo<const I&>;
    { *i++ } -> Same<reference>;
  };
// clang-format on

// clang-format off
template <class C,
          class iterator = typename C::iterator,
          class const_iterator = typename C::const_iterator,
          class size_type = typename C::size_type>
concept Container =
  ForwardIterator<iterator> &&
  ForwardIterator<const_iterator> &&
  requires(C a, const C ca, C b) {
    { C() } -> Same<C>;
    { C(a) } -> Same<C>;
    { a = b } -> Same<C&>;
    { a.~C() } -> Same<void>;
    { a.begin() } -> Same<iterator>;
    { ca.begin() } -> Same<const_iterator>;
    { a.end() } -> Same<iterator>;
    { ca.end() } -> Same<const_iterator>;
    { a.cbegin() } -> Same<const_iterator>;
    { a.cend() } -> Same<const_iterator>;
    { a == b } -> ConvertibleTo<bool>;
    { a != b } -> ConvertibleTo<bool>;
    { a.swap(b) } -> Same<void>;
    { swap(a, b) } -> Same<void>;
    { a.size() } -> Same<size_type>;
    { a.max_size() } -> Same<size_type>;
    { a.empty() } -> ConvertibleTo<bool>;
  };
// clang-format on

// clang-format off
template <class T, class Shape = std::array<int, T::rank>>
concept Tensor =
  Container<T> &&
  requires(T t, const T ct, Shape cartesian_index) {
    { std::apply(t, cartesian_index) } -> Same<typename T::reference>;
    { std::apply(ct, cartesian_index) } -> Same<typename T::const_reference>;
  };
// clang-format on

} // namespace v0

} // namespace nd
