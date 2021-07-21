#pragma once

#include <array>
#include <cstring>
#include <cmath>

#include <arbor/simd/implbase.hpp>

namespace arb {
namespace simd {
namespace detail {

template <typename T, unsigned N>
struct generic;

template <typename T, unsigned N>
struct simd_traits<generic<T, N>> {
    static constexpr unsigned width = N;
    using scalar_type = T;
    using vector_type = std::array<T, N>;
    using mask_impl = generic<bool, N>;
    static constexpr unsigned min_align = alignof(vector_type);
};

template <typename T, unsigned N>
struct generic: implbase<generic<T, N>> {
    using array = std::array<T, N>;

    static array copy_from(const T* p) {
        array result;
        std::memcpy(&result, p, sizeof(result));
        return result;
    }

    static void copy_to(const array& v, T* p) {
        std::memcpy(p, &v, sizeof(v));
    }

    static void mask_copy_to(const array& v, bool* w) {
        std::copy(v.begin(), v.end(), w);
    }

    static array mask_copy_from(const bool* y) {
        array v;
        std::copy(y, y+N, v.data());
        return v;
    }

    static bool mask_element(const array& v, int i) {
        return static_cast<bool>(v[i]);
    }

    static void mask_set_element(array& v, int i, bool x) {
        v[i] = x;
    }
};

} // namespace detail

namespace simd_abi {

    template <typename T, unsigned N> struct generic {
        using type = detail::generic<T, N>;
    };

} // namespace simd_abi

} // namespace simd
} // namespace arb
