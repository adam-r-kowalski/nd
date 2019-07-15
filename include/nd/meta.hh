#pragma once

namespace nd {

inline namespace v0 {

template <class T, class U>
concept Same = std::is_same_v<T, U>;

}

}
