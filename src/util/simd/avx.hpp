#pragma once

// AVX/AVX2 SIMD intrinsics implementation.

#ifdef __AVX__

#include <cstdint>
#include <immintrin.h>

namespace arb {
namespace simd_detail {

struct avx_double4 {
    using scalar_type = double;
    using vector_type = __m256d;

    // Masks use the same representation:
    using mask_impl = avx_double4;
    using mask_type = typename mask_impl::vector_type;

    using int64 = std::int64_t;

    static constexpr int cmp_gt_oq = 30; // _CMP_GT_OQ
    static constexpr int cmp_ge_oq = 29; // _CMP_GE_OQ
    static constexpr int cmp_le_oq = 18; // _CMP_LE_OQ
    static constexpr int cmp_lt_oq = 17; // _CMP_LT_OQ
    static constexpr int cmp_eq_oq =  0; // _CMP_EQ_OQ
    static constexpr int cmp_neq_uq = 4; // _CMP_NEQ_UQ

    constexpr static unsigned width = 4;

    static vector_type broadcast(double v) {
        return _mm256_set1_pd(v);
    }

    static vector_type immediate(double v0, double v1, double v2, double v3) {
        return _mm256_setr_pd(v0, v1, v2, v3);
    }

    static vector_type immediate(bool b0, bool b1, bool b2, bool b3) {
        return _mm256_castsi256_pd(_mm256_setr_epi64x(-b0, -b1, -b2, -b3));
    }

    static void copy_to(const vector_type& v, scalar_type* p) {
        _mm256_storeu_pd(p, v);
    }

    static vector_type copy_from(const scalar_type* p) {
        return _mm256_loadu_pd(p);
    }

    static vector_type add(const vector_type& a, const vector_type& b) {
        return _mm256_add_pd(a, b);
    }

    static vector_type sub(const vector_type& a, const vector_type& b) {
        return _mm256_sub_pd(a, b);
    }

    static vector_type mul(const vector_type& a, const vector_type& b) {
        return _mm256_mul_pd(a, b);
    }

    static vector_type div(const vector_type& a, const vector_type& b) {
        return _mm256_div_pd(a, b);
    }

    static vector_type fma(const vector_type& a, const vector_type& b, const vector_type& c) {
        vector_type r;
        for (unsigned i = 0; i<width; ++i) {
           r[i] = std::fma(a[i], b[i], c[i]);
        }
        return r;
    }

    static vector_type logical_not(const vector_type& a) {
        return _mm256_castsi256_pd(~_mm256_castpd_si256(a));
    }

    static vector_type logical_and(const vector_type& a, const vector_type& b) {
        return _mm256_and_pd(a, b);
    }

    static vector_type logical_or(const vector_type& a, const vector_type& b) {
        return _mm256_or_pd(a, b);
    }

    static mask_type cmp_eq(const vector_type& a, const vector_type& b) {
        return _mm256_cmp_pd(a, b, cmp_eq_oq);
    }

    static mask_type cmp_neq(const vector_type& a, const vector_type& b) {
        return _mm256_cmp_pd(a, b, cmp_neq_uq);
    }

    static mask_type cmp_gt(const vector_type& a, const vector_type& b) {
        return _mm256_cmp_pd(a, b, cmp_gt_oq);
    }

    static mask_type cmp_geq(const vector_type& a, const vector_type& b) {
        return _mm256_cmp_pd(a, b, cmp_ge_oq);
    }

    static mask_type cmp_lt(const vector_type& a, const vector_type& b) {
        return _mm256_cmp_pd(a, b, cmp_lt_oq);
    }

    static mask_type cmp_leq(const vector_type& a, const vector_type& b) {
        return _mm256_cmp_pd(a, b, cmp_le_oq);
    }

    static vector_type select(const mask_type& m, const vector_type& u, const vector_type& v) {
        return _mm256_blendv_pd(u, v, m);
    }

    static scalar_type element(const vector_type& u, int i) {
        return u[i];
    }

    static void set_element(vector_type& u, int i, scalar_type x) {
        u[i] = x;
    }

    static vector_type mask_broadcast(bool b) {
        return _mm256_castsi256_pd(_mm256_set1_epi64x(-(int64)b));
    }

    static bool mask_element(const vector_type& u, int i) {
        return static_cast<bool>(u[i]);
    }

    static void mask_set_element(vector_type& u, int i, bool b) {
        __m256i ui = _mm256_castpd_si256(u);
        ui[i] = -(int64)b;
        u = _mm256_castsi256_pd(ui);
    }

    static void mask_copy_to(const vector_type& m, bool* y) {
        __m128i zero = _mm_setzero_si128();

        // Split into upper and lower 128-bits (two mask values
        // in each), translate 0xffffffffffffffff to 0x0000000000000001.

        __m128i ml = _mm_castpd_si128(_mm256_castpd256_pd128(m));
        ml = _mm_sub_epi64(zero, ml);

        __m128i mu = _mm_castpd_si128(_mm256_castpd256_pd128(_mm256_permute2f128_pd(m, m, 1)));
        mu = _mm_sub_epi64(zero, mu);

        // Move bytes with bool value to bytes 0 and 1 in lower half,
        // bytes 2 and 3 in upper half, and merge with bitwise-or.

        __m128i sl = _mm_setr_epi32(0x80800800ul,0,0,0);
        ml = _mm_shuffle_epi8(ml, sl);

        __m128i su = _mm_setr_epi32(0x08008080ul,0,0,0);
        mu = _mm_shuffle_epi8(mu, su);

        __m128i r = _mm_or_si128(mu, ml);
        std::memcpy(y, &r, 4);
    }

    static vector_type mask_copy_from(const bool* w) {
        __m128i zero = _mm_setzero_si128();

        __m128i r;
        std::memcpy(&r, w, 4);

        // Move bytes:
        //   rl: byte 0 to byte 0, byte 1 to byte 8, zero elsewhere.
        //   ru: byte 2 to byte 0, byte 3 to byte 8, zero elsewhere.
        //
        // Subtract from zero to translate
        // 0x0000000000000001 to 0xffffffffffffffff.

        __m128i sl = _mm_setr_epi32(0x80808000ul, 0x80808080ul, 0x80808001ul, 0x80808080ul);
        __m128i rl = _mm_sub_epi64(zero, _mm_shuffle_epi8(r, sl));

        __m128i su = _mm_setr_epi32(0x80808002ul, 0x80808080ul, 0x80808003ul, 0x80808080ul);
        __m128i ru = _mm_sub_epi64(zero, _mm_shuffle_epi8(r, su));

        return _mm256_castsi256_pd(combine_m128i(ru, rl));
    }

protected:
    static __m256i combine_m128i(__m128i hi, __m128i lo) {
        return _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1);
    }
};

} // namespace simd_detail

namespace simd_abi {

    template <typename T, unsigned N> struct avx;
    template <> struct avx<double, 4> { using type = simd_detail::avx_double4; };

} // namespace simd_abi;


// AVX2 extends AVX operations, with the same data representation.

#ifdef __AVX2__

namespace simd_detail {

struct avx2_double4: avx_double4 {
    // Masks use the same representation:
    using mask_impl = avx2_double4;
    using mask_type = typename mask_impl::vector_type;

    static vector_type fma(const vector_type& a, const vector_type& b, const vector_type& c) {
        return _mm256_fmadd_pd(a, b, c);
    }

    static void mask_copy_to(const vector_type& m, bool* y) {
        __m256i zero = _mm256_setzero_si256();

        // Translate 0xffffffffffffffff scalars to 0x0000000000000001.

        __m256i x = _mm256_castpd_si256(m);
        x = _mm256_sub_epi64(zero, x);

        // Move lower 32-bits of each field to lower 128-bit half of x.

        __m256i s1 = _mm256_setr_epi32(0,2,4,8,0,0,0,0);
        x = _mm256_permutevar8x32_epi32(x, s1);

        // Move the lowest byte from each 32-bit field to bottom bytes.

        __m128i s2 = _mm_setr_epi32(0x0c080400ul,0,0,0);
        __m128i r = _mm_shuffle_epi8(_mm256_castsi256_si128(x), s2);
        std::memcpy(y, &r, 4);
    }

    static vector_type mask_copy_from(const bool* w) {
        __m256i zero = _mm256_setzero_si256();

        __m256i r;
        std::memcpy(&r, w, 4);
        return _mm256_sub_epi64(zero, _mm256_cvtepi8_epi64(r));
    }
};

} // namespace simd_detail

namespace simd_abi {

    template <typename T, unsigned N> struct avx2;
    template <> struct avx2<double, 4> { using type = simd_detail::avx2_double4; };

} // namespace simd_abi;

#endif // def __AVX2__

} // namespace arb

#endif // def __AVX__
