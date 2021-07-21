#pragma once

// AVX/AVX2 SIMD intrinsics implementation.

#ifdef __AVX__

#include <cmath>
#include <cstdint>
#include <immintrin.h>

#include <arbor/simd/approx.hpp>
#include <arbor/simd/implbase.hpp>

namespace arb {
namespace simd {
namespace detail {

struct avx_int4;
struct avx_double4;

static constexpr unsigned avx_min_align = 16;

template <>
struct simd_traits<avx_int4> {
    static constexpr unsigned width = 4;
    static constexpr unsigned min_align = avx_min_align;
    using scalar_type = std::int32_t;
    using vector_type = __m128i;
    using mask_impl = avx_int4;
};

template <>
struct simd_traits<avx_double4> {
    static constexpr unsigned width = 4;
    static constexpr unsigned min_align = avx_min_align;
    using scalar_type = double;
    using vector_type = __m256d;
    using mask_impl = avx_double4;
};

struct avx_int4: implbase<avx_int4> {
    // Use default implementations for: element, set_element, div.

    using implbase<avx_int4>::cast_from;

    using int32 = std::int32_t;

    static __m128i broadcast(int32 v) {
        return _mm_set1_epi32(v);
    }

    static void copy_to(const __m128i& v, int32* p) {
        _mm_storeu_si128((__m128i*)p, v);
    }

    static void copy_to_masked(const __m128i& v, int32* p, const __m128i& mask) {
        _mm_maskstore_ps(reinterpret_cast<float*>(p), mask, _mm_castsi128_ps(v));
    }

    static __m128i copy_from(const int32* p) {
        return _mm_loadu_si128((const __m128i*)p);
    }

    static __m128i copy_from_masked(const int32* p, const __m128i& mask) {
        return _mm_castps_si128(_mm_maskload_ps(reinterpret_cast<const float*>(p), mask));
    }

    static __m128i copy_from_masked(const __m128i& v, const int32* p, const __m128i& mask) {
        __m128 d = _mm_maskload_ps(reinterpret_cast<const float*>(p), mask);
        return ifelse(mask, _mm_castps_si128(d), v);
    }

    static __m128i cast_from(tag<avx_double4>, const __m256d& v) {
        return _mm256_cvttpd_epi32(v);
    }

    static int element0(const __m128i& a) {
        return _mm_cvtsi128_si32(a);
    }

    static __m128i neg(const __m128i& a) {
        __m128i zero = _mm_setzero_si128();
        return _mm_sub_epi32(zero, a);
    }

    static __m128i add(const __m128i& a, const __m128i& b) {
        return _mm_add_epi32(a, b);
    }

    static __m128i sub(const __m128i& a, const __m128i& b) {
        return _mm_sub_epi32(a, b);
    }

    static __m128i mul(const __m128i& a, const __m128i& b) {
        return _mm_mullo_epi32(a, b);
    }

    static __m128i fma(const __m128i& a, const __m128i& b, const __m128i& c) {
        return _mm_add_epi32(_mm_mullo_epi32(a, b), c);
    }

    static __m128i logical_not(const __m128i& a) {
        __m128i ones = {};
        return _mm_xor_si128(a, _mm_cmpeq_epi32(ones, ones));
    }

    static __m128i logical_and(const __m128i& a, const __m128i& b) {
        return _mm_and_si128(a, b);
    }

    static __m128i logical_or(const __m128i& a, const __m128i& b) {
        return _mm_or_si128(a, b);
    }

    static __m128i cmp_eq(const __m128i& a, const __m128i& b) {
        return _mm_cmpeq_epi32(a, b);
    }

    static __m128i cmp_neq(const __m128i& a, const __m128i& b) {
        return logical_not(cmp_eq(a, b));
    }

    static __m128i cmp_gt(const __m128i& a, const __m128i& b) {
        return _mm_cmpgt_epi32(a, b);
    }

    static __m128i cmp_geq(const __m128i& a, const __m128i& b) {
        return logical_not(cmp_gt(b, a));
    }

    static __m128i cmp_lt(const __m128i& a, const __m128i& b) {
        return cmp_gt(b, a);
    }

    static __m128i cmp_leq(const __m128i& a, const __m128i& b) {
        return logical_not(cmp_gt(a, b));
    }

    static __m128i ifelse(const __m128i& m, const __m128i& u, const __m128i& v) {
        return _mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(v), _mm_castsi128_ps(u), _mm_castsi128_ps(m)));
    }

    static __m128i mask_broadcast(bool b) {
        return _mm_set1_epi32(-(int32)b);
    }

    static __m128i mask_unpack(unsigned long long k) {
        // Only care about bottom four bits of k.
        __m128i b = _mm_set1_epi8((char)k);
        b = logical_or(b, _mm_setr_epi32(0xfefefefe,0xfdfdfdfd,0xfbfbfbfb,0xf7f7f7f7));

        __m128i ones = {};
        ones = _mm_cmpeq_epi32(ones, ones);
        return _mm_cmpeq_epi32(b, ones);
    }

    static bool mask_element(const __m128i& u, int i) {
        return static_cast<bool>(element(u, i));
    }

    static void mask_set_element(__m128i& u, int i, bool b) {
        set_element(u, i, -(int32)b);
    }

    static void mask_copy_to(const __m128i& m, bool* y) {
        // Negate (convert 0xffffffff to 0x00000001) and move low bytes to
        // bottom 4 bytes.

        __m128i s = _mm_setr_epi32(0x0c080400ul,0,0,0);
        __m128i p = _mm_shuffle_epi8(neg(m), s);
        std::memcpy(y, &p, 4);
    }

    static __m128i mask_copy_from(const bool* w) {
        __m128i r;
        std::memcpy(&r, w, 4);

        __m128i s = _mm_setr_epi32(0x80808000ul, 0x80808001ul, 0x80808002ul, 0x80808003ul);
        return neg(_mm_shuffle_epi8(r, s));
    }

    static __m128i max(const __m128i& a, const __m128i& b) {
        return _mm_max_epi32(a, b);
    }

    static __m128i min(const __m128i& a, const __m128i& b) {
        return _mm_min_epi32(a, b);
    }

    static int reduce_add(const __m128i& a) {
        // Add [a3|a2|a1|a0] to [a2|a3|a0|a1]
        __m128i b = add(a, _mm_shuffle_epi32(a, 0xb1));
        // Add [b3|b2|b1|b0] to [b1|b0|b3|b2]
        __m128i c = add(b, _mm_shuffle_epi32(b, 0x4e));

        return element0(c);
    }
};

struct avx_double4: implbase<avx_double4> {
    // Use default implementations for:
    //     element, set_element, fma.

    using implbase<avx_double4>::cast_from;

    using int64 = std::int64_t;

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

    static __m256d broadcast(double v) {
        return _mm256_set1_pd(v);
    }

    static void copy_to(const __m256d& v, double* p) {
        _mm256_storeu_pd(p, v);
    }

    static void copy_to_masked(const __m256d& v, double* p, const __m256d& mask) {
        _mm256_maskstore_pd(p, _mm256_castpd_si256(mask), v);
    }

    static __m256d copy_from(const double* p) {
        return _mm256_loadu_pd(p);
    }

    static __m256d copy_from_masked(const double* p, const __m256d& mask) {
        return _mm256_maskload_pd(p, _mm256_castpd_si256(mask));
    }

    static __m256d copy_from_masked(const __m256d& v, const double* p, const __m256d& mask) {
        __m256d d = _mm256_maskload_pd(p, _mm256_castpd_si256(mask));
        return ifelse(mask, d, v);
    }

    static __m256d cast_from(tag<avx_int4>, const __m128i& v) {
        return _mm256_cvtepi32_pd(v);
    }

    static double element0(const __m256d& a) {
        return _mm_cvtsd_f64(_mm256_castpd256_pd128(a));
    }

    static __m256d neg(const __m256d& a) {
        return _mm256_sub_pd(zero(), a);
    }

    static __m256d add(const __m256d& a, const __m256d& b) {
        return _mm256_add_pd(a, b);
    }

    static __m256d sub(const __m256d& a, const __m256d& b) {
        return _mm256_sub_pd(a, b);
    }

    static __m256d mul(const __m256d& a, const __m256d& b) {
        return _mm256_mul_pd(a, b);
    }

    static __m256d div(const __m256d& a, const __m256d& b) {
        return _mm256_div_pd(a, b);
    }

    static __m256d logical_not(const __m256d& a) {
        __m256d ones = {};
        return _mm256_xor_pd(a, _mm256_cmp_pd(ones, ones, 15));
    }

    static __m256d logical_and(const __m256d& a, const __m256d& b) {
        return _mm256_and_pd(a, b);
    }

    static __m256d logical_or(const __m256d& a, const __m256d& b) {
        return _mm256_or_pd(a, b);
    }

    static __m256d cmp_eq(const __m256d& a, const __m256d& b) {
        return _mm256_cmp_pd(a, b, cmp_eq_oq);
    }

    static __m256d cmp_neq(const __m256d& a, const __m256d& b) {
        return _mm256_cmp_pd(a, b, cmp_neq_uq);
    }

    static __m256d cmp_gt(const __m256d& a, const __m256d& b) {
        return _mm256_cmp_pd(a, b, cmp_gt_oq);
    }

    static __m256d cmp_geq(const __m256d& a, const __m256d& b) {
        return _mm256_cmp_pd(a, b, cmp_ge_oq);
    }

    static __m256d cmp_lt(const __m256d& a, const __m256d& b) {
        return _mm256_cmp_pd(a, b, cmp_lt_oq);
    }

    static __m256d cmp_leq(const __m256d& a, const __m256d& b) {
        return _mm256_cmp_pd(a, b, cmp_le_oq);
    }

    static __m256d ifelse(const __m256d& m, const __m256d& u, const __m256d& v) {
        return _mm256_blendv_pd(v, u, m);
    }

    static __m256d mask_broadcast(bool b) {
        return _mm256_castsi256_pd(_mm256_set1_epi64x(-(int64)b));
    }

    static bool mask_element(const __m256d& u, int i) {
        return static_cast<bool>(element(u, i));
    }

    static __m256d mask_unpack(unsigned long long k) {
        // Only care about bottom four bits of k.
        __m128i b = _mm_set1_epi8((char)k);
        // (Note: there is no _mm_setr_epi64x (!))
        __m128i bl = _mm_or_si128(b, _mm_set_epi64x(0xfdfdfdfdfdfdfdfd, 0xfefefefefefefefe));
        __m128i bu = _mm_or_si128(b, _mm_set_epi64x(0xf7f7f7f7f7f7f7f7, 0xfbfbfbfbfbfbfbfb));

        __m128i ones = {};
        ones = _mm_cmpeq_epi32(ones, ones);
        bl = _mm_cmpeq_epi64(bl, ones);
        bu = _mm_cmpeq_epi64(bu, ones);
        return _mm256_castsi256_pd(combine_m128i(bu, bl));
    }

    static void mask_set_element(__m256d& u, int i, bool b) {
        char data[256];
        _mm256_storeu_pd((double*)data, u);
        ((int64*)data)[i] = -(int64)b;
        u = _mm256_loadu_pd((double*)data);
    }

    static void mask_copy_to(const __m256d& m, bool* y) {
        // Convert to 32-bit wide mask values, and delegate to
        // avx2_int4.

        avx_int4::mask_copy_to(lo_epi32(_mm256_castpd_si256(m)), y);
    }

    static __m256d mask_copy_from(const bool* w) {
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

    static __m256d max(const __m256d& a, const __m256d& b) {
        return _mm256_max_pd(a, b);
    }

    static __m256d min(const __m256d& a, const __m256d& b) {
        return _mm256_min_pd(a, b);
    }

    static __m256d abs(const __m256d& x) {
        __m256i m = _mm256_set1_epi64x(0x7fffffffffffffffll);
        return _mm256_and_pd(x, _mm256_castsi256_pd(m));
    }

    static double reduce_add(const __m256d& a) {
        // add [a3|a2|a1|a0] to [a1|a0|a3|a2]
        __m256d b = add(a, _mm256_permute2f128_pd(a, a, 0x01));
        // add [b3|b2|b1|b0] to [b2|b3|b0|b1]
        __m256d c = add(b, _mm256_permute_pd(b, 0x05));

        return element0(c);
    }

    // Exponential is calculated as follows:
    //
    //     e^x = e^g · 2^n,
    //
    // where g in [-0.5, 0.5) and n is an integer. 2^n can be
    // calculated via bit manipulation or specialized scaling intrinsics,
    // whereas e^g is approximated using the order-6 rational
    // approximation:
    //
    //     e^g = R(g)/R(-g)
    //
    // with R(x) split into even and odd terms:
    //
    //     R(x) = Q(x^2) + x·P(x^2)
    //
    // so that the ratio can be computed as:
    //
    //     e^g = 1 + 2·g·P(g^2) / (Q(g^2)-g·P(g^2)).
    //
    // Note that the coefficients for R are close to but not the same as those
    // from the 6,6 Padé approximant to the exponential. 
    //
    // The exponents n and g are calculated by:
    //
    //     n = floor(x/ln(2) + 0.5)
    //     g = x - n·ln(2)
    //
    // so that x = g + n·ln(2). We have:
    //
    //     |g| = |x - n·ln(2)|
    //         = |x - x + α·ln(2)|
    //  
    // for some fraction |α| ≤ 0.5, and thus |g| ≤ 0.5ln(2) ≈ 0.347.
    //
    // Tne subtraction x - n·ln(2) is performed in two parts, with
    // ln(2) = C1 + C2, in order to compensate for the possible loss of precision
    // attributable to catastrophic rounding. C1 comprises the first
    // 32-bits of mantissa, C2 the remainder.

    static  __m256d exp(const __m256d& x) {
        // Masks for exceptional cases.

        auto is_large = cmp_gt(x, broadcast(exp_maxarg));
        auto is_small = cmp_lt(x, broadcast(exp_minarg));
        auto is_nan = _mm256_cmp_pd(x, x, cmp_unord_q);

        // Compute n and g.

        auto n = _mm256_floor_pd(add(mul(broadcast(ln2inv), x), broadcast(0.5)));

        auto g = sub(x, mul(n, broadcast(ln2C1)));
        g = sub(g, mul(n, broadcast(ln2C2)));

        auto gg = mul(g, g);

        // Compute the g*P(g^2) and Q(g^2).

        auto odd = mul(g, horner(gg, P0exp, P1exp, P2exp));
        auto even = horner(gg, Q0exp, Q1exp, Q2exp, Q3exp);

        // Compute R(g)/R(-g) = 1 + 2*g*P(g^2) / (Q(g^2)-g*P(g^2))

        auto expg = add(broadcast(1), mul(broadcast(2),
            div(odd, sub(even, odd))));

        // Finally, compute product with 2^n.
        // Note: can only achieve full range using the ldexp implementation,
        // rather than multiplying by 2^n directly.

        auto result = ldexp_positive(expg, _mm256_cvtpd_epi32(n));

        return
            ifelse(is_large, broadcast(HUGE_VAL),
            ifelse(is_small, broadcast(0),
            ifelse(is_nan, broadcast(NAN),
                   result)));
    }

    // Use same rational polynomial expansion as for exp(x), without
    // the unit term.
    //
    // For |x|<=0.5, take n to be zero. Otherwise, set n as above,
    // and scale the answer by:
    //     expm1(x) = 2^n * expm1(g) + (2^n - 1).

    static  __m256d expm1(const __m256d& x) {
        auto is_large = cmp_gt(x, broadcast(exp_maxarg));
        auto is_small = cmp_lt(x, broadcast(expm1_minarg));
        auto is_nan = _mm256_cmp_pd(x, x, cmp_unord_q);

        auto half = broadcast(0.5);
        auto one = broadcast(1.);
        auto two = add(one, one);

        auto nzero = cmp_leq(abs(x), half);
        auto n = _mm256_floor_pd(add(mul(broadcast(ln2inv), x), half));
        n = ifelse(nzero, zero(), n);

        auto g = sub(x, mul(n, broadcast(ln2C1)));
        g = sub(g, mul(n, broadcast(ln2C2)));

        auto gg = mul(g, g);

        auto odd = mul(g, horner(gg, P0exp, P1exp, P2exp));
        auto even = horner(gg, Q0exp, Q1exp, Q2exp, Q3exp);

        // Note: multiply by two, *then* divide: avoids a subnormal
        // intermediate that will get truncated to zero with default
        // icpc options.
        auto expgm1 = div(mul(two, odd), sub(even, odd));

        // For small x (n zero), bypass scaling step to avoid underflow.
        // Otherwise, compute result 2^n * expgm1 + (2^n-1) by:
        //     result = 2 * ( 2^(n-1)*expgm1 + (2^(n-1)+0.5) )
        // to avoid overflow when n=1024.

        auto nm1 = _mm256_cvtpd_epi32(sub(n, one));
        auto scaled = mul(add(sub(exp2int(nm1), half), ldexp_normal(expgm1, nm1)), two);

        return
            ifelse(is_large, broadcast(HUGE_VAL),
            ifelse(is_small, broadcast(-1),
            ifelse(is_nan, broadcast(NAN),
            ifelse(nzero, expgm1,
                   scaled))));
    }

    // Natural logarithm:
    //
    // Decompose x = 2^g * u such that g is an integer and
    // u is in the interval [sqrt(2)/2, sqrt(2)].
    //
    // Then ln(x) is computed as R(u-1) + g*ln(2) where
    // R(z) is a rational polynomial approximating ln(z+1)
    // for small z:
    //
    //     R(z) = z - z^2/2 + z^3 * P(z)/Q(z)
    //
    // where P and Q are degree 5 polynomials, Q monic.
    //
    // In order to avoid cancellation error, ln(2) is represented
    // as C3 + C4, with the C4 correction term approx. -2.1e-4.
    // The summation order for R(z)+2^g is:
    //
    //     z^3*P(z)/Q(z) + g*C4 - z^2/2 + z + g*C3

    static __m256d log(const __m256d& x) {
        // Masks for exceptional cases.

        auto is_large = cmp_geq(x, broadcast(HUGE_VAL));
        auto is_small = cmp_lt(x, broadcast(log_minarg));
        auto is_domainerr = _mm256_cmp_pd(x, broadcast(0), cmp_nge_uq);

        __m256d g = _mm256_cvtepi32_pd(logb_normal(x));
        __m256d u = fraction_normal(x);

        __m256d one = broadcast(1.);
        __m256d half = broadcast(0.5);
        auto gtsqrt2 = cmp_geq(u, broadcast(sqrt2));
        g = ifelse(gtsqrt2, add(g, one), g);
        u = ifelse(gtsqrt2, mul(u, half), u);

        auto z = sub(u, one);
        auto pz = horner(z, P0log, P1log, P2log, P3log, P4log, P5log);
        auto qz = horner1(z, Q0log, Q1log, Q2log, Q3log, Q4log);

        auto z2 = mul(z, z);
        auto z3 = mul(z2, z);

        auto r = div(mul(z3, pz), qz);
        r = add(r, mul(g,  broadcast(ln2C4)));
        r = sub(r, mul(z2, half));
        r = add(r, z);
        r = add(r, mul(g,  broadcast(ln2C3)));

        // Return NaN if x is NaN or negarive, +inf if x is +inf,
        // or -inf if zero or (positive) denormal.

        return
            ifelse(is_domainerr, broadcast(NAN),
            ifelse(is_large, broadcast(HUGE_VAL),
            ifelse(is_small, broadcast(-HUGE_VAL),
                r)));
    }

protected:
    static __m256d zero() {
        return _mm256_setzero_pd();
    }

    static __m128i hi_epi32(__m256i x) {
        __m128i xl = _mm256_castsi256_si128(x);
        __m128i xh = _mm256_extractf128_si256(x, 1);
        return _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(xl), _mm_castsi128_ps(xh), 0xddu));
    }

    static __m128i lo_epi32(__m256i x) {
        __m128i xl = _mm256_castsi256_si128(x);
        __m128i xh = _mm256_extractf128_si256(x, 1);
        return _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(xl), _mm_castsi128_ps(xh), 0x88u));
    }

    static __m256i combine_m128i(__m128i hi, __m128i lo) {
        return _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1);
    }

    // horner(x, a0, ..., an) computes the degree n polynomial A(x) with coefficients
    // a0, ..., an by a0+x·(a1+x·(a2+...+x·an)...).

    static inline __m256d horner(__m256d x, double a0) {
        return broadcast(a0);
    }

    template <typename... T>
    static __m256d horner(__m256d x, double a0, T... tail) {
        return add(mul(x, horner(x, tail...)), broadcast(a0));
    }

    // horner1(x, a0, ..., an) computes the degree n+1 monic polynomial A(x) with coefficients
    // a0, ..., an, 1 by by a0+x·(a1+x·(a2+...+x·(an+x)...).

    static inline __m256d horner1(__m256d x, double a0) {
        return add(x, broadcast(a0));
    }

    template <typename... T>
    static __m256d horner1(__m256d x, double a0, T... tail) {
        return add(mul(x, horner1(x, tail...)), broadcast(a0));
    }

    // Compute 2.0^n.
    static __m256d exp2int(__m128i n) {
        n = _mm_slli_epi32(n, 20);
        n = _mm_add_epi32(n, _mm_set1_epi32(1023<<20));

        auto nl = _mm_shuffle_epi32(n, 0x50);
        auto nh = _mm_shuffle_epi32(n, 0xfa);
        __m256i nhnl = combine_m128i(nh, nl);

        return _mm256_castps_pd(
            _mm256_blend_ps(_mm256_set1_ps(0),
            _mm256_castsi256_ps(nhnl),0xaa));
    }

    // Compute n and f such that x = 2^n·f, with |f| ∈ [1,2), given x is finite and normal.
    static __m128i logb_normal(const __m256d& x) {
        __m128i xw = hi_epi32(_mm256_castpd_si256(x));
        __m128i emask = _mm_set1_epi32(0x7ff00000);
        __m128i ebiased = _mm_srli_epi32(_mm_and_si128(xw, emask), 20);

        return _mm_sub_epi32(ebiased, _mm_set1_epi32(1023));
    }

    static __m256d fraction_normal(const __m256d& x) {
        // 0x800fffffffffffff (intrinsic takes signed parameter)
        __m256d emask = _mm256_castsi256_pd(_mm256_set1_epi64x(-0x7ff0000000000001));
        __m256d bias = _mm256_castsi256_pd(_mm256_set1_epi64x(0x3ff0000000000000));
        return _mm256_or_pd(bias, _mm256_and_pd(emask, x));
    }

    // Compute 2^n·x when both x and 2^n·x are normal, finite and strictly positive doubles.
    static __m256d ldexp_positive(__m256d x, __m128i n) {
        n = _mm_slli_epi32(n, 20);
        auto zero = _mm_set1_epi32(0);
        auto nl = _mm_unpacklo_epi32(zero, n);
        auto nh = _mm_unpackhi_epi32(zero, n);

        __m128d xl = _mm256_castpd256_pd128(x);
        __m128d xh = _mm256_extractf128_pd(x, 1);

        __m128i suml = _mm_add_epi64(nl, _mm_castpd_si128(xl));
        __m128i sumh = _mm_add_epi64(nh, _mm_castpd_si128(xh));
        __m256i sumhl = combine_m128i(sumh, suml);

        return _mm256_castsi256_pd(sumhl);
    }

    // Compute 2^n·x when both x and 2^n·x are normal and finite.
    static __m256d ldexp_normal(__m256d x, __m128i n) {
        __m256d smask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fffffffffffffffll));
        __m256d sbits = _mm256_andnot_pd(smask, x);

        n = _mm_slli_epi32(n, 20);
        auto zi = _mm_set1_epi32(0);
        auto nl = _mm_unpacklo_epi32(zi, n);
        auto nh = _mm_unpackhi_epi32(zi, n);

        __m128d xl = _mm256_castpd256_pd128(x);
        __m128d xh = _mm256_extractf128_pd(x, 1);

        __m128i suml = _mm_add_epi64(nl, _mm_castpd_si128(xl));
        __m128i sumh = _mm_add_epi64(nh, _mm_castpd_si128(xh));
        __m256i sumhl = combine_m128i(sumh, suml);

        auto nzans = _mm256_or_pd(_mm256_and_pd(_mm256_castsi256_pd(sumhl), smask), sbits);
        return ifelse(cmp_eq(x, zero()), zero(), nzans);
    }
};


#if defined(__AVX2__) && defined(__FMA__)

struct avx2_int4;
struct avx2_double4;

template <>
struct simd_traits<avx2_int4> {
    static constexpr unsigned width = 4;
    static constexpr unsigned min_align = avx_min_align;
    using scalar_type = std::int32_t;
    using vector_type = __m128i;
    using mask_impl = avx_int4;
};

template <>
struct simd_traits<avx2_double4> {
    static constexpr unsigned width = 4;
    static constexpr unsigned min_align = avx_min_align;
    using scalar_type = double;
    using vector_type = __m256d;
    using mask_impl = avx2_double4;
};

// Note: we derive from avx_int4 only as an implementation shortcut.
// Because `avx2_int4` does not derive from `implbase<avx2_int4>`,
// any fallback methods in `implbase` will use the `avx_int4`
// functions rather than the `avx2_int4` functions.

struct avx2_int4: avx_int4 {
    using implbase<avx_int4>::cast_from;

    // Need to provide a cast overload for avx2_double4 tag:
    static __m128i cast_from(tag<avx2_double4>, const __m256d& v) {
        return _mm256_cvttpd_epi32(v);
    }
};

// Note: we derive from avx_double4 only as an implementation shortcut.
// Because `avx2_double4` does not derive from `implbase<avx2_double4>`,
// any fallback methods in `implbase` will use the `avx_double4`
// functions rather than the `avx2_double4` functions.

struct avx2_double4: avx_double4 {
    using implbase<avx_double4>::cast_from;
    using implbase<avx_double4>::gather;

    // Need to provide a cast overload for avx2_int4 tag:
    static __m256d cast_from(tag<avx2_int4>, const __m128i& v) {
        return _mm256_cvtepi32_pd(v);
    }

    static __m256d fma(const __m256d& a, const __m256d& b, const __m256d& c) {
        return _mm256_fmadd_pd(a, b, c);
    }

    static vector_type logical_not(const vector_type& a) {
        __m256i ones = {};
        return _mm256_xor_pd(a, _mm256_castsi256_pd(_mm256_cmpeq_epi32(ones, ones)));
    }

    static __m256d mask_unpack(unsigned long long k) {
        // Only care about bottom four bits of k.
        __m256i b = _mm256_set1_epi8((char)k);
        b = _mm256_or_si256(b, _mm256_setr_epi64x(
                0xfefefefefefefefe, 0xfdfdfdfdfdfdfdfd,
                0xfbfbfbfbfbfbfbfb, 0xf7f7f7f7f7f7f7f7));

        __m256i ones = {};
        ones = _mm256_cmpeq_epi64(ones, ones);
        return _mm256_castsi256_pd(_mm256_cmpeq_epi64(ones, b));
    }

    static void mask_copy_to(const __m256d& m, bool* y) {
        // Convert to 32-bit wide mask values, and delegate to
        // avx2_int4.

        avx_int4::mask_copy_to(lo_epi32(_mm256_castpd_si256(m)), y);
    }

    static __m256d mask_copy_from(const bool* w) {
        __m256i zero = _mm256_setzero_si256();

        __m128i r;
        std::memcpy(&r, w, 4);
        return _mm256_castsi256_pd(_mm256_sub_epi64(zero, _mm256_cvtepi8_epi64(r)));
    }

    static __m256d gather(tag<avx2_int4>, const double* p, const __m128i& index) {
        return _mm256_i32gather_pd(p, index, 8);
    }

    static __m256d gather(tag<avx2_int4>, __m256d a, const double* p, const __m128i& index, const __m256d& mask) {
        return  _mm256_mask_i32gather_pd(a, p, index, mask, 8);
    };

    // avx4_double4 versions of log, exp, and expm1 use the same algorithms as for avx_double4,
    // but use AVX2-specialized bit manipulation and FMA.

    static  __m256d exp(const __m256d& x) {
        // Masks for exceptional cases.

        auto is_large = cmp_gt(x, broadcast(exp_maxarg));
        auto is_small = cmp_lt(x, broadcast(exp_minarg));
        auto is_nan = _mm256_cmp_pd(x, x, cmp_unord_q);

        // Compute n and g.

        auto n = _mm256_floor_pd(fma(broadcast(ln2inv), x, broadcast(0.5)));

        auto g = fma(n, broadcast(-ln2C1), x);
        g = fma(n, broadcast(-ln2C2), g);

        auto gg = mul(g, g);

        // Compute the g*P(g^2) and Q(g^2).

        auto odd = mul(g, horner(gg, P0exp, P1exp, P2exp));
        auto even = horner(gg, Q0exp, Q1exp, Q2exp, Q3exp);

        // Compute R(g)/R(-g) = 1 + 2*g*P(g^2) / (Q(g^2)-g*P(g^2))

        auto expg = fma(broadcast(2), div(odd, sub(even, odd)), broadcast(1));

        // Finally, compute product with 2^n.
        // Note: can only achieve full range using the ldexp implementation,
        // rather than multiplying by 2^n directly.

        auto result = ldexp_positive(expg, _mm256_cvtpd_epi32(n));

        return
            ifelse(is_large, broadcast(HUGE_VAL),
            ifelse(is_small, broadcast(0),
            ifelse(is_nan, broadcast(NAN),
                   result)));
    }

    static  __m256d expm1(const __m256d& x) {
        auto is_large = cmp_gt(x, broadcast(exp_maxarg));
        auto is_small = cmp_lt(x, broadcast(expm1_minarg));
        auto is_nan = _mm256_cmp_pd(x, x, cmp_unord_q);

        auto half = broadcast(0.5);
        auto one = broadcast(1.);
        auto two = add(one, one);

        auto smallx = cmp_leq(abs(x), half);
        auto n = _mm256_floor_pd(fma(broadcast(ln2inv), x, half));
        n = ifelse(smallx, zero(), n);

        auto g = fma(n, broadcast(-ln2C1), x);
        g = fma(n, broadcast(-ln2C2), g);

        auto gg = mul(g, g);

        auto odd = mul(g, horner(gg, P0exp, P1exp, P2exp));
        auto even = horner(gg, Q0exp, Q1exp, Q2exp, Q3exp);

        auto expgm1 = div(mul(two, odd), sub(even, odd));

        auto nm1 = _mm256_cvtpd_epi32(sub(n, one));
        auto scaled = mul(add(sub(exp2int(nm1), half), ldexp_normal(expgm1, nm1)), two);

        return
            ifelse(is_large, broadcast(HUGE_VAL),
            ifelse(is_small, broadcast(-1),
            ifelse(is_nan, broadcast(NAN),
            ifelse(smallx, expgm1,
                   scaled))));
    }

    static __m256d log(const __m256d& x) {
        // Masks for exceptional cases.

        auto is_large = cmp_geq(x, broadcast(HUGE_VAL));
        auto is_small = cmp_lt(x, broadcast(log_minarg));
        auto is_domainerr = _mm256_cmp_pd(x, broadcast(0), cmp_nge_uq);

        __m256d g = _mm256_cvtepi32_pd(logb_normal(x));
        __m256d u = fraction_normal(x);

        __m256d one = broadcast(1.);
        __m256d half = broadcast(0.5);
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

        // Return NaN if x is NaN or negative, +inf if x is +inf,
        // or -inf if zero or (positive) denormal.

        return
            ifelse(is_domainerr, broadcast(NAN),
            ifelse(is_large, broadcast(HUGE_VAL),
            ifelse(is_small, broadcast(-HUGE_VAL),
                r)));
    }

protected:
    static __m128i lo_epi32(__m256i a) {
        a = _mm256_shuffle_epi32(a, 0x08);
        a = _mm256_permute4x64_epi64(a, 0x08);
        return _mm256_castsi256_si128(a);
    }

    static  __m128i hi_epi32(__m256i a) {
        a = _mm256_shuffle_epi32(a, 0x0d);
        a = _mm256_permute4x64_epi64(a, 0x08);
        return _mm256_castsi256_si128(a);
    }

    static inline __m256d horner(__m256d x, double a0) {
        return broadcast(a0);
    }

    template <typename... T>
    static __m256d horner(__m256d x, double a0, T... tail) {
        return fma(x, horner(x, tail...), broadcast(a0));
    }

    static inline __m256d horner1(__m256d x, double a0) {
        return add(x, broadcast(a0));
    }

    template <typename... T>
    static __m256d horner1(__m256d x, double a0, T... tail) {
        return fma(x, horner1(x, tail...), broadcast(a0));
    }

    static __m256d fms(const __m256d& a, const __m256d& b, const __m256d& c) {
        return _mm256_fmsub_pd(a, b, c);
    }

    // Compute 2.0^n.
    // Overrides avx_double4::exp2int.
    static __m256d exp2int(__m128i n) {
        __m256d x = broadcast(1);
        __m256i nshift = _mm256_slli_epi64(_mm256_cvtepi32_epi64(n), 52);
        __m256i sum = _mm256_add_epi64(nshift, _mm256_castpd_si256(x));
        return _mm256_castsi256_pd(sum);
    }

    // Compute 2^n*x when both x and 2^n*x are normal, finite and strictly positive doubles.
    // Overrides avx_double4::ldexp_positive.
    static __m256d ldexp_positive(__m256d x, __m128i n) {
        __m256i nshift = _mm256_slli_epi64(_mm256_cvtepi32_epi64(n), 52);
        __m256i sum = _mm256_add_epi64(nshift, _mm256_castpd_si256(x));
        return _mm256_castsi256_pd(sum);
    }

    // Compute 2^n*x when both x and 2^n*x are normal and finite.
    // Overrides avx_double4::ldexp_normal.
    static __m256d ldexp_normal(__m256d x, __m128i n) {
        __m256d smask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fffffffffffffffll));
        __m256d sbits = _mm256_andnot_pd(smask, x);
        __m256i nshift = _mm256_slli_epi64(_mm256_cvtepi32_epi64(n), 52);
        __m256i sum = _mm256_add_epi64(nshift, _mm256_castpd_si256(x));

        auto nzans = _mm256_or_pd(_mm256_and_pd(_mm256_castsi256_pd(sum), smask), sbits);
        return ifelse(cmp_eq(x, zero()), zero(), nzans);
    }

    // Override avx_double4::logb_normal so as to use avx2 version of hi_epi32.
    static __m128i logb_normal(const __m256d& x) {
        __m128i xw = hi_epi32(_mm256_castpd_si256(x));
        __m128i emask = _mm_set1_epi32(0x7ff00000);
        __m128i ebiased = _mm_srli_epi32(_mm_and_si128(xw, emask), 20);

        return _mm_sub_epi32(ebiased, _mm_set1_epi32(1023));
    }
};
#endif // defined(__AVX2__) && defined(__FMA__)

} // namespace detail

namespace simd_abi {
    template <typename T, unsigned N> struct avx;

    template <> struct avx<int, 4> { using type = detail::avx_int4; };
    template <> struct avx<double, 4> { using type = detail::avx_double4; };

#if defined(__AVX2__) && defined(__FMA__)
    template <typename T, unsigned N> struct avx2;

    template <> struct avx2<int, 4> { using type = detail::avx2_int4; };
    template <> struct avx2<double, 4> { using type = detail::avx2_double4; };
#endif
} // namespace simd_abi

} // namespace simd
} // namespace arb

#endif // def __AVX__
