#pragma once

// AVX512F SIMD intrinsics implementation.

#ifdef __AVX512F__

#include <cmath>
#include <cstdint>
#include <immintrin.h>

#include <arbor/simd/approx.hpp>
#include <arbor/simd/implbase.hpp>

namespace arb {
namespace simd {
namespace detail {

struct avx512_double8;
struct avx512_int8;
struct avx512_mask8;

static constexpr unsigned avx512_min_align = 16;

template <>
struct simd_traits<avx512_mask8> {
    static constexpr unsigned width = 8;
    static constexpr unsigned min_align = avx512_min_align;
    using scalar_type = bool;
    using vector_type = __mmask8;
    using mask_impl = avx512_mask8;
};

template <>
struct simd_traits<avx512_double8> {
    static constexpr unsigned width = 8;
    static constexpr unsigned min_align = avx512_min_align;
    using scalar_type = double;
    using vector_type = __m512d;
    using mask_impl = avx512_mask8;
};

template <>
struct simd_traits<avx512_int8> {
    static constexpr unsigned width = 8;
    static constexpr unsigned min_align = avx512_min_align;
    using scalar_type = std::int32_t;
    using vector_type = __m512i;
    using mask_impl = avx512_mask8;
};

struct avx512_mask8: implbase<avx512_mask8> {
    using implbase<avx512_mask8>::gather;
    using implbase<avx512_mask8>::scatter;
    using implbase<avx512_mask8>::cast_from;

    static __mmask8 broadcast(bool b) {
        return _mm512_int2mask(-b);
    }

    static void copy_to(const __mmask8& k, bool* b) {
        __m256i a = _mm256_setzero_si256();
        a = _mm512_castsi512_si256(_mm512_mask_set1_epi32(_mm512_castsi256_si512(a), k, 1));

        __m256i s = _mm256_set1_epi32(0x0c080400);
        a = _mm256_shuffle_epi8(a, s);

        s = _mm256_setr_epi32(0, 4, 0, 0, 0, 0, 0, 0);
        a = _mm256_permutevar8x32_epi32(a, s);

        std::memcpy(b, &a, 8);
    }

    static __mmask8 copy_from(const bool* p) {
        __m256i a = _mm256_setzero_si256();
        std::memcpy(&a, p, 8);
        a = _mm256_sub_epi8(_mm256_setzero_si256(), a);
        return _mm512_int2mask(_mm256_movemask_epi8(a));
    }

    // Note: fall back to implbase implementations of copy_to_masked and copy_from_masked;
    // could be improved with the use of AVX512BW instructions on supported platforms.

    static __mmask8 logical_not(const __mmask8& k) {
        return _mm512_knot(k);
    }

    static __mmask8 logical_and(const __mmask8& a, const __mmask8& b) {
        return _mm512_kand(a, b);
    }

    static __mmask8 logical_or(const __mmask8& a, const __mmask8& b) {
        return _mm512_kor(a, b);
    }

    // Arithmetic operations not necessarily appropriate for
    // packed bit mask, but implemented for completeness/testing,
    // with Z modulo 2 semantics:
    //     a + b   is equivalent to   a ^ b
    //     a * b                      a & b
    //     a / b                      a
    //     a - b                      a ^ b
    //     -a                         a
    //     max(a, b)                  a | b
    //     min(a, b)                  a & b

    static __mmask8 neg(const __mmask8& a) {
        return a;
    }

    static __mmask8 add(const __mmask8& a, const __mmask8& b) {
        return _mm512_kxor(a, b);
    }

    static __mmask8 sub(const __mmask8& a, const __mmask8& b) {
        return _mm512_kxor(a, b);
    }

    static __mmask8 mul(const __mmask8& a, const __mmask8& b) {
        return _mm512_kand(a, b);
    }

    static __mmask8 div(const __mmask8& a, const __mmask8& b) {
        return a;
    }

    static __mmask8 fma(const __mmask8& a, const __mmask8& b, const __mmask8& c) {
        return add(mul(a, b), c);
    }

    static __mmask8 max(const __mmask8& a, const __mmask8& b) {
        return _mm512_kor(a, b);
    }

    static __mmask8 min(const __mmask8& a, const __mmask8& b) {
        return _mm512_kand(a, b);
    }

    // Comparison operators are also taken as operating on Z modulo 2,
    // with 1 > 0:
    //
    //     a > b    is equivalent to  a & ~b
    //     a >= b                     a | ~b,  ~(~a & b)
    //     a < b                      ~a & b
    //     a <= b                     ~a | b,  ~(a & ~b)
    //     a == b                     ~(a ^ b)
    //     a != b                     a ^ b

    static __mmask8 cmp_eq(const __mmask8& a, const __mmask8& b) {
        return _mm512_kxnor(a, b);
    }

    static __mmask8 cmp_neq(const __mmask8& a, const __mmask8& b) {
        return _mm512_kxor(a, b);
    }

    static __mmask8 cmp_lt(const __mmask8& a, const __mmask8& b) {
        return _mm512_kandn(a, b);
    }

    static __mmask8 cmp_gt(const __mmask8& a, const __mmask8& b) {
        return cmp_lt(b, a);
    }

    static __mmask8 cmp_geq(const __mmask8& a, const __mmask8& b) {
        return logical_not(cmp_lt(a, b));
    }

    static __mmask8 cmp_leq(const __mmask8& a, const __mmask8& b) {
        return logical_not(cmp_gt(a, b));
    }

    static __mmask8 ifelse(const __mmask8& m, const __mmask8& u, const __mmask8& v) {
        return _mm512_kor(_mm512_kandn(m, u), _mm512_kand(m, v));
    }

    static bool element(const __mmask8& k, int i) {
        return _mm512_mask2int(k)&(1<<i);
    }

    static void set_element(__mmask8& k, int i, bool b) {
        int n = _mm512_mask2int(k);
        k = _mm512_int2mask((n&~(1<<i))|(b<<i));
    }

    static __mmask8 mask_broadcast(bool b) {
        return broadcast(b);
    }

    static __mmask8 mask_unpack(unsigned long long p) {
        return _mm512_int2mask(p);
    }

    static bool mask_element(const __mmask8& u, int i) {
        return element(u, i);
    }

    static void mask_set_element(__mmask8& u, int i, bool b) {
        set_element(u, i, b);
    }

    static void mask_copy_to(const __mmask8& m, bool* y) {
        copy_to(m, y);
    }

    static __mmask8 mask_copy_from(const bool* y) {
        return copy_from(y);
    }
};

struct avx512_int8: implbase<avx512_int8> {
    // Use default implementations for:
    //     element, set_element.
    //
    // Consider changing mask representation to __mmask16
    // and be explicit about comparison masks: restrictions
    // to __mmask8 seem to produce a lot of ultimately unnecessary
    // operations. 

    using implbase<avx512_int8>::gather;
    using implbase<avx512_int8>::scatter;
    using implbase<avx512_int8>::cast_from;

    using int32 = std::int32_t;

    static __mmask8 lo() {
        return _mm512_int2mask(0xff);
    }

    static __m512i broadcast(int32 v) {
        return _mm512_set1_epi32(v);
    }

    static void copy_to(const __m512i& v, int32* p) {
        _mm512_mask_storeu_epi32(p, lo(), v);
    }

    static void copy_to_masked(const __m512i& v, int32* p, const __mmask8& mask) {
        _mm512_mask_storeu_epi32(p, mask, v);
    }

    static __m512i copy_from(const int32* p) {
        return _mm512_maskz_loadu_epi32(lo(), p);
    }

    static __m512i copy_from_masked(const int32* p, const __mmask8& mask) {
        return _mm512_maskz_loadu_epi32(mask, p);
    }

    static __m512i copy_from_masked(const __m512i& v, const int32* p, const __mmask8& mask) {
        return _mm512_mask_loadu_epi32(v, mask, p);
    }

    static int element0(const __m512i& a) {
        return _mm_cvtsi128_si32(_mm512_castsi512_si128(a));
    }

    static __m512i neg(const __m512i& a) {
        return sub(_mm512_setzero_epi32(), a);
    }

    static __m512i add(const __m512i& a, const __m512i& b) {
        return _mm512_add_epi32(a, b);
    }

    static __m512i sub(const __m512i& a, const __m512i& b) {
        return _mm512_sub_epi32(a, b);
    }

    static __m512i mul(const __m512i& a, const __m512i& b) {
        // Can represent 32-bit exactly in double, and technically overflow is
        // undefined behaviour, so we can do this in doubles.
        constexpr int32 rtz = _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC;
        auto da = _mm512_cvtepi32_pd(_mm512_castsi512_si256(a));
        auto db = _mm512_cvtepi32_pd(_mm512_castsi512_si256(b));
        auto fpmul = _mm512_mul_round_pd(da, db, rtz);
        return _mm512_castsi256_si512(_mm512_cvt_roundpd_epi32(fpmul, rtz));
    }

    static __m512i div(const __m512i& a, const __m512i& b) {
        // Can represent 32-bit exactly in double, so do a fp division with fixed rounding.
        constexpr int32 rtz = _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC;
        auto da = _mm512_cvtepi32_pd(_mm512_castsi512_si256(a));
        auto db = _mm512_cvtepi32_pd(_mm512_castsi512_si256(b));
        auto fpdiv = _mm512_div_round_pd(da, db, rtz);
        return _mm512_castsi256_si512(_mm512_cvt_roundpd_epi32(fpdiv, rtz));
    }

    static __m512i fma(const __m512i& a, const __m512i& b, const __m512i& c) {
        return add(mul(a, b), c);
    }

    static __mmask8 cmp_eq(const __m512i& a, const __m512i& b) {
        return _mm512_cmpeq_epi32_mask(a, b);
    }

    static __mmask8 cmp_neq(const __m512i& a, const __m512i& b) {
        return _mm512_cmpneq_epi32_mask(a, b);
    }

    static __mmask8 cmp_gt(const __m512i& a, const __m512i& b) {
        return _mm512_cmpgt_epi32_mask(a, b);
    }

    static __mmask8 cmp_geq(const __m512i& a, const __m512i& b) {
        return _mm512_cmpge_epi32_mask(a, b);
    }

    static __mmask8 cmp_lt(const __m512i& a, const __m512i& b) {
        return _mm512_cmplt_epi32_mask(a, b);
    }

    static __mmask8 cmp_leq(const __m512i& a, const __m512i& b) {
        return _mm512_cmple_epi32_mask(a, b);
    }

    static __m512i ifelse(const __mmask8& m, const __m512i& u, const __m512i& v) {
        return _mm512_mask_blend_epi32(m, v, u);
    }

    static __m512i max(const __m512i& a, const __m512i& b) {
        return _mm512_max_epi32(a, b);
    }

    static __m512i min(const __m512i& a, const __m512i& b) {
        return _mm512_min_epi32(a, b);
    }

    static __m512i abs(const __m512i& a) {
        return _mm512_abs_epi32(a);
    }

    static int reduce_add(const __m512i& a) {
        // Add [...|a7|a6|a5|a4|a3|a2|a1|a0] to [...|a3|a2|a1|a0|a7|a6|a5|a4]
        //__m512i b = add(a, _mm512_shuffle_i32x4(a, a, 0xb1));
        __m512i b = add(a, _mm512_shuffle_i32x4(a, a, _MM_PERM_CDAB)); // 0xb1
        // Add [...|b7|b6|b5|b4|b3|b2|b1|b0] to [...|b6|b7|b4|b5|b2|b3|b0|b1]
        //__m512i c = add(b, _mm512_shuffle_epi32(b, 0xb1));
        __m512i c = add(b, _mm512_shuffle_epi32(b, _MM_PERM_CDAB)); // 0xb1
        // Add [...|c7|c6|c5|c4|c3|c2|c1|c0] to [...|c5|c4|c7|c6|c1|c0|c3|c2]
        //__m512i d = add(c, _mm512_shuffle_epi32(c, 0x4e));
        __m512i d = add(c, _mm512_shuffle_epi32(c, _MM_PERM_BADC)); // 0x4e

        return element0(d);
    }

    // Generic 8-wide int solutions for gather and scatter.

    template <typename Impl>
    using is_int8_simd = std::integral_constant<bool, std::is_same<int, typename Impl::scalar_type>::value && Impl::width==8>;

    template <typename ImplIndex,
              typename = std::enable_if_t<is_int8_simd<ImplIndex>::value>>
    static __m512i gather(tag<ImplIndex>, const int32* p, const typename ImplIndex::vector_type& index) {
        int32 o[16];
        ImplIndex::copy_to(index, o);
        auto op = reinterpret_cast<const __m512i*>(o);
        return _mm512_mask_i32gather_epi32(_mm512_setzero_epi32(), lo(), _mm512_loadu_si512(op), p, 4);
    }

    template <typename ImplIndex,
              typename = std::enable_if_t<is_int8_simd<ImplIndex>::value>>
    static __m512i gather(tag<ImplIndex>, const __m512i& a, const int32* p, const typename ImplIndex::vector_type& index, const __mmask8& mask) {
        int32 o[16];
        ImplIndex::copy_to(index, o);
        auto op = reinterpret_cast<const __m512i*>(o);
        return _mm512_mask_i32gather_epi32(a, mask, _mm512_loadu_si512(op), p, 4);
    }

    template <typename ImplIndex,
              typename = std::enable_if_t<is_int8_simd<ImplIndex>::value>>
    static void scatter(tag<ImplIndex>, const __m512i& s, int32* p, const typename ImplIndex::vector_type& index) {
        int32 o[16];
        ImplIndex::copy_to(index, o);
        auto op = reinterpret_cast<const __m512i*>(o);
        _mm512_mask_i32scatter_epi32(p, lo(), _mm512_loadu_si512(op), s, 4);
    }

    template <typename ImplIndex,
              typename = std::enable_if_t<is_int8_simd<ImplIndex>::value>>
    static void scatter(tag<ImplIndex>, const __m512i& s, int32* p, const typename ImplIndex::vector_type& index, const __mmask8& mask) {
        int32 o[16];
        ImplIndex::copy_to(index, o);
        auto op = reinterpret_cast<const __m512i*>(o);
        _mm512_mask_i32scatter_epi32(p, mask, _mm512_loadu_si512(op), s, 4);
    }

    // Specialized 8-wide gather and scatter for avx512_int8 implementation.

    static __m512i gather(tag<avx512_int8>, const int32* p, const __m512i& index) {
        return _mm512_mask_i32gather_epi32(_mm512_setzero_epi32(), lo(), index, p, 4);
    }

    static __m512i gather(tag<avx512_int8>, __m512i a, const int32* p, const __m512i& index, const __mmask8& mask) {
        return _mm512_mask_i32gather_epi32(a, mask, index, p, 4);
    }

    static void scatter(tag<avx512_int8>, const __m512i& s, int32* p, const __m512i& index) {
        _mm512_mask_i32scatter_epi32(p, lo(), index, s, 4);
    }

    static void scatter(tag<avx512_int8>, const __m512i& s, int32* p, const __m512i& index, const __mmask8& mask) {
        _mm512_mask_i32scatter_epi32(p, mask, index, s, 4);
    }
};

struct avx512_double8: implbase<avx512_double8> {
    // Use default implementations for:
    //     element, set_element.

    using implbase<avx512_double8>::gather;
    using implbase<avx512_double8>::scatter;
    using implbase<avx512_double8>::cast_from;

    // CMPPD predicates:
    static constexpr int cmp_eq_oq =    0;
    static constexpr int cmp_unord_q =  3;
    static constexpr int cmp_neq_uq =   4;
    static constexpr int cmp_true_uq = 15;
    static constexpr int cmp_lt_oq =   17;
    static constexpr int cmp_le_oq =   18;
    static constexpr int cmp_nge_uq =  25;
    static constexpr int cmp_ge_oq =   29;
    static constexpr int cmp_gt_oq =   30;

    static __m512d broadcast(double v) {
        return _mm512_set1_pd(v);
    }

    static void copy_to(const __m512d& v, double* p) {
        _mm512_storeu_pd(p, v);
    }

    static void copy_to_masked(const __m512d& v, double* p, const __mmask8& mask) {
        _mm512_mask_storeu_pd(p, mask, v);
    }

    static __m512d copy_from(const double* p) {
        return _mm512_loadu_pd(p);
    }

    static __m512d copy_from_masked(const double* p, const __mmask8& mask) {
        return _mm512_maskz_loadu_pd(mask, p);
    }

    static __m512d copy_from_masked(const __m512d& v, const double* p, const __mmask8& mask) {
        return _mm512_mask_loadu_pd(v, mask, p);
    }

    static double element0(const __m512d& a) {
        return _mm_cvtsd_f64(_mm512_castpd512_pd128(a));
    }

    static __m512d neg(const __m512d& a) {
        return _mm512_sub_pd(_mm512_setzero_pd(), a);
    }

    static __m512d add(const __m512d& a, const __m512d& b) {
        return _mm512_add_pd(a, b);
    }

    static __m512d sub(const __m512d& a, const __m512d& b) {
        return _mm512_sub_pd(a, b);
    }

    static __m512d mul(const __m512d& a, const __m512d& b) {
        return _mm512_mul_pd(a, b);
    }

    static __m512d div(const __m512d& a, const __m512d& b) {
        return _mm512_div_pd(a, b);
    }

    static __m512d fma(const __m512d& a, const __m512d& b, const __m512d& c) {
        return _mm512_fmadd_pd(a, b, c);
    }

    static __mmask8 cmp_eq(const __m512d& a, const __m512d& b) {
        return _mm512_cmp_pd_mask(a, b, cmp_eq_oq);
    }

    static __mmask8 cmp_neq(const __m512d& a, const __m512d& b) {
        return _mm512_cmp_pd_mask(a, b, cmp_neq_uq);
    }

    static __mmask8 cmp_gt(const __m512d& a, const __m512d& b) {
        return _mm512_cmp_pd_mask(a, b, cmp_gt_oq);
    }

    static __mmask8 cmp_geq(const __m512d& a, const __m512d& b) {
        return _mm512_cmp_pd_mask(a, b, cmp_ge_oq);
    }

    static __mmask8 cmp_lt(const __m512d& a, const __m512d& b) {
        return _mm512_cmp_pd_mask(a, b, cmp_lt_oq);
    }

    static __mmask8 cmp_leq(const __m512d& a, const __m512d& b) {
        return _mm512_cmp_pd_mask(a, b, cmp_le_oq);
    }

    static __m512d ifelse(const __mmask8& m, const __m512d& u, const __m512d& v) {
        return _mm512_mask_blend_pd(m, v, u);
    }

    static __m512d max(const __m512d& a, const __m512d& b) {
        return _mm512_max_pd(a, b);
    }

    static __m512d min(const __m512d& a, const __m512d& b) {
        return _mm512_min_pd(a, b);
    }

    static __m512d abs(const __m512d& x) {
        // (NB: _mm512_abs_pd() wrapper buggy on gcc, so use AND directly.)
        __m512i m = _mm512_set1_epi64(0x7fffffffffffffff);
        return _mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(x), m));
    }

    static double reduce_add(const __m512d& a) {
        // add [a7|a6|a5|a4|a3|a2|a1|a0] to [a3|a2|a1|a0|a7|a6|a5|a4]
        __m512d b = add(a, _mm512_shuffle_f64x2(a, a, 0x4e));
        // add [b7|b6|b5|b4|b3|b2|b1|b0] to [b5|b4|b7|b6|b1|b0|b3|b2]
        __m512d c = add(b, _mm512_permutex_pd(b, 0x4e));
        // add [c7|c6|c5|c4|c3|c2|c1|c0] to [c6|c7|c4|c5|c2|c3|c0|c1]
        __m512d d = add(c, _mm512_permute_pd(c, 0x55));

        return element0(d);
    }

    // Generic 8-wide int solutions for gather and scatter.

    template <typename Impl>
    using is_int8_simd = std::integral_constant<bool, std::is_same<int, typename Impl::scalar_type>::value && Impl::width==8>;

    template <typename ImplIndex, typename = std::enable_if_t<is_int8_simd<ImplIndex>::value>>
    static __m512d gather(tag<ImplIndex>, const double* p, const typename ImplIndex::vector_type& index) {
        int o[8];
        ImplIndex::copy_to(index, o);
        auto op = reinterpret_cast<const __m256i*>(o);
        return _mm512_i32gather_pd(_mm256_loadu_si256(op), p, 8);
    }

    template <typename ImplIndex, typename = std::enable_if_t<is_int8_simd<ImplIndex>::value>>
    static __m512d gather(tag<ImplIndex>, const __m512d& a, const double* p, const typename ImplIndex::vector_type& index, const __mmask8& mask) {
        int o[8];
        ImplIndex::copy_to(index, o);
        auto op = reinterpret_cast<const __m256i*>(o);
        return _mm512_mask_i32gather_pd(a, mask, _mm256_loadu_si256(op), p, 8);
    }

    template <typename ImplIndex, typename = std::enable_if_t<is_int8_simd<ImplIndex>::value>>
    static void scatter(tag<ImplIndex>, const __m512d& s, double* p, const typename ImplIndex::vector_type& index) {
        int o[8];
        ImplIndex::copy_to(index, o);
        auto op = reinterpret_cast<const __m256i*>(o);
        _mm512_i32scatter_pd(p, _mm256_loadu_si256(op), s, 8);
    }

    template <typename ImplIndex, typename = std::enable_if_t<is_int8_simd<ImplIndex>::value>>
    static void scatter(tag<ImplIndex>, const __m512d& s, double* p, const typename ImplIndex::vector_type& index, const __mmask8& mask) {
        int o[8];
        ImplIndex::copy_to(index, o);
        auto op = reinterpret_cast<const __m256i*>(o);
        _mm512_mask_i32scatter_pd(p, mask, _mm256_loadu_si256(op), s, 8);
    }

    // Specialized 8-wide gather and scatter for avx512_int8 implementation.

    static __m512d gather(tag<avx512_int8>, const double* p, const __m512i& index) {
        return _mm512_i32gather_pd(_mm512_castsi512_si256(index), p, 8);
    }

    static __m512d gather(tag<avx512_int8>, __m512d a, const double* p, const __m512i& index, const __mmask8& mask) {
        return _mm512_mask_i32gather_pd(a, mask, _mm512_castsi512_si256(index), p, 8);
    }

    static void scatter(tag<avx512_int8>, const __m512d& s, double* p, const __m512i& index) {
        _mm512_i32scatter_pd(p, _mm512_castsi512_si256(index), s, 8);
    }

    static void scatter(tag<avx512_int8>, const __m512d& s, double* p, const __m512i& index, const __mmask8& mask) {
        _mm512_mask_i32scatter_pd(p, mask, _mm512_castsi512_si256(index), s, 8);
    }

    // Use SVML for exp and log if compiling with icpc, else use ratpoly
    // approximations.

#if defined(__INTEL_COMPILER)
    static  __m512d exp(const __m512d& x) {
        return _mm512_exp_pd(x);
    }

    static  __m512d expm1(const __m512d& x) {
        return _mm512_expm1_pd(x);
    }

    static __m512d log(const __m512d& x) {
        return _mm512_log_pd(x);
    }

    static __m512d pow(const __m512d& x, const __m512d& y) {
        return _mm512_pow_pd(x, y);
    }
#else

    // Refer to avx/avx2 code for details of the exponential and log
    // implementations.

    static  __m512d exp(const __m512d& x) {
        // Masks for exceptional cases.

        auto is_large = cmp_gt(x, broadcast(exp_maxarg));
        auto is_small = cmp_lt(x, broadcast(exp_minarg));

        // Compute n and g.

        auto n = _mm512_floor_pd(add(mul(broadcast(ln2inv), x), broadcast(0.5)));

        auto g = fma(n, broadcast(-ln2C1), x);
        g = fma(n, broadcast(-ln2C2), g);

        auto gg = mul(g, g);

        // Compute the g*P(g^2) and Q(g^2).
auto odd = mul(g, horner(gg, P0exp, P1exp, P2exp));
        auto even = horner(gg, Q0exp, Q1exp, Q2exp, Q3exp);

        // Compute R(g)/R(-g) = 1 + 2*g*P(g^2) / (Q(g^2)-g*P(g^2))

        auto expg = fma(broadcast(2), div(odd, sub(even, odd)), broadcast(1));

        // Scale by 2^n, propogating NANs.

        auto result = _mm512_scalef_pd(expg, n);

        return
            ifelse(is_large, broadcast(HUGE_VAL),
            ifelse(is_small, broadcast(0),
                   result));
    }

    static  __m512d expm1(const __m512d& x) {
        auto is_large = cmp_gt(x, broadcast(exp_maxarg));
        auto is_small = cmp_lt(x, broadcast(expm1_minarg));

        auto half = broadcast(0.5);
        auto one = broadcast(1.);

        auto nnz = cmp_gt(abs(x), half);
        auto n = _mm512_maskz_roundscale_round_pd(
                    nnz,
                    mul(broadcast(ln2inv), x),
                    0,
                    _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);

        auto g = fma(n, broadcast(-ln2C1), x);
        g = fma(n, broadcast(-ln2C2), g);

        auto gg = mul(g, g);

        auto odd = mul(g, horner(gg, P0exp, P1exp, P2exp));
        auto even = horner(gg, Q0exp, Q1exp, Q2exp, Q3exp);

        // Compute R(g)/R(-g) -1 = 2*g*P(g^2) / (Q(g^2)-g*P(g^2))

        auto expgm1 = div(mul(broadcast(2), odd), sub(even, odd));

        // For small x (n zero), bypass scaling step to avoid underflow.
        // Otherwise, compute result 2^n * expgm1 + (2^n-1) by:
        //     result = 2 * ( 2^(n-1)*expgm1 + (2^(n-1)+0.5) )
        // to avoid overflow when n=1024.

        auto nm1 = sub(n, one);

        auto result =
            _mm512_scalef_pd(
                add(sub(_mm512_scalef_pd(one, nm1), half),
                    _mm512_scalef_pd(expgm1, nm1)),
                one);

        return
            ifelse(is_large, broadcast(HUGE_VAL),
            ifelse(is_small, broadcast(-1),
            ifelse(nnz, result, expgm1)));
    }

    static __m512d log(const __m512d& x) {
        // Masks for exceptional cases.

        auto is_large = cmp_geq(x, broadcast(HUGE_VAL));
        auto is_small = cmp_lt(x, broadcast(log_minarg));
        is_small = avx512_mask8::logical_and(is_small, cmp_geq(x, broadcast(0)));

        __m512d g = _mm512_getexp_pd(x);
        __m512d u = _mm512_getmant_pd(x, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_nan);

        __m512d one = broadcast(1.);
        __m512d half = broadcast(0.5);
        auto gtsqrt2 = cmp_geq(u, broadcast(sqrt2));
        g = ifelse(gtsqrt2, add(g, one), g);
        u = ifelse(gtsqrt2, mul(u, half), u);

        auto z = sub(u, one);
        auto pz = horner(z, P0log, P1log, P2log, P3log, P4log, P5log);
        auto qz = horner1(z, Q0log, Q1log, Q2log, Q3log, Q4log);

        auto z2 = mul(z, z);
        auto z3 = mul(z2, z);

        auto r = div(mul(z3, pz), qz);
        r = fma(g,  broadcast(ln2C4), r);
        r = fms(z2, half, r);
        r = sub(z, r);
        r = fma(g,  broadcast(ln2C3), r);

        // r is alrady NaN if x is NaN or negative, otherwise
        // return  +inf if x is +inf, or -inf if zero or (positive) denormal.

        return
            ifelse(is_large, broadcast(HUGE_VAL),
            ifelse(is_small, broadcast(-HUGE_VAL),
                r));
    }
#endif

protected:
    static inline __m512d horner1(__m512d x, double a0) {
        return add(x, broadcast(a0));
    }

    static inline __m512d horner(__m512d x, double a0) {
        return broadcast(a0);
    }

    template <typename... T>
    static __m512d horner(__m512d x, double a0, T... tail) {
        return fma(x, horner(x, tail...), broadcast(a0));
    }

    template <typename... T>
    static __m512d horner1(__m512d x, double a0, T... tail) {
        return fma(x, horner1(x, tail...), broadcast(a0));
    }

    static __m512d fms(const __m512d& a, const __m512d& b, const __m512d& c) {
        return _mm512_fmsub_pd(a, b, c);
    }
};

} // namespace detail

namespace simd_abi {
    template <typename T, unsigned N> struct avx512;
    template <> struct avx512<double, 8> { using type = detail::avx512_double8; };
    template <> struct avx512<int, 8> { using type = detail::avx512_int8; };
} // namespace simd_abi

} // namespace simd
} // namespace arb

#endif // def __AVX512F__
