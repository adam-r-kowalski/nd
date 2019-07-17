#pragma once

#include <array>
#include <type_traits>

namespace nd {

namespace traits {

inline namespace v0 {

template <class T, class U> concept Same = std::is_same_v<T, U>;

template <class T> struct shape { using type = typename T::shape_t; };
template <class T> using shape_t = typename shape<T>::type;

template <class T> struct value { using type = typename T::value_t; };
template <class T, size_t N> struct value<std::array<T, N>> {
  using type = typename std::array<T, N>::value_type;
};
template <class T> using value_t = typename value<T>::type;

template <class T> struct reference { using type = typename T::reference_t; };
template <class T> using reference_t = typename reference<T>::type;

template <class T> struct const_reference {
  using type = typename T::const_reference_t;
};
template <class T> using const_reference_t = typename const_reference<T>::type;

template <class T> struct iterator { using type = typename T::iterator_t; };
template <class T> using iterator_t = typename iterator<T>::type;

template <class T> struct const_iterator {
  using type = typename T::const_iterator_t;
};
template <class T> using const_iterator_t = typename const_iterator<T>::type;

} // namespace v0

} // namespace traits

} // namespace nd
