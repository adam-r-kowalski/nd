#pragma once

namespace nd {

inline namespace v0 {

// clang-format off
template <class C>
concept Container = requires(C a, const C ca, C b) {
  { C() } -> C;
  { C(a) } -> C;
  { a = b } -> C&;
  { a.~C() } -> void;
  { a.begin() } -> typename C::iterator;
  { ca.begin() } -> typename C::const_iterator;
  { a.end() } -> typename C::iterator;
  { ca.end() } -> typename C::const_iterator;
  { a.cbegin() } -> typename C::const_iterator;
  { a.cend() } -> typename C::const_iterator;
  { a == b } -> bool;
  { a != b } -> bool;
  { a.swap(b) } -> void;
  { swap(a, b) } -> void;
  { a.size() } -> typename C::size_type;
  { a.max_size() } -> typename C::size_type;
  { a.empty() } -> bool;
};
// clang-format on

} // namespace v0

} // namespace nd
