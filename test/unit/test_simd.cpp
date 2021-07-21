#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <random>
#include <unordered_set>

#include <arbor/simd/avx.hpp>
#include <arbor/simd/neon.hpp>
#include <arbor/simd/sve.hpp>
#include <arbor/simd/simd.hpp>
#include <arbor/util/compat.hpp>

#include "common.hpp"

using namespace arb::simd;
using index_constraint = arb::simd::index_constraint;

namespace {
    // Use different distributions in `fill_random`, based on the value type in question:
    //
    //     * floating point type => uniform_real_distribution, default interval [-1, 1).
    //     * bool                => uniform_int_distribution, default interval [0, 1).
    //     * other integral type => uniform_int_distribution, default interval [L, U]
    //                              such that L^2+L and U^2+U fit within the integer range.

    template <typename V, typename = std::enable_if_t<std::is_floating_point<V>::value>>
    std::uniform_real_distribution<V> make_udist(V lb = -1., V ub = 1.) {
        return std::uniform_real_distribution<V>(lb, ub);
    }

    template <typename V, typename = std::enable_if_t<std::is_integral<V>::value && !std::is_same<V, bool>::value>>
    std::uniform_int_distribution<V> make_udist(
            V lb = std::numeric_limits<V>::lowest() / (2 << std::numeric_limits<V>::digits/2),
            V ub = std::numeric_limits<V>::max() >> (1+std::numeric_limits<V>::digits/2))
    {
        return std::uniform_int_distribution<V>(lb, ub);
    }

    template <typename V, typename = std::enable_if_t<std::is_same<V, bool>::value>>
    std::uniform_int_distribution<> make_udist(V lb = 0, V ub = 1) {
        return std::uniform_int_distribution<>(0, 1);
    }

    template <typename Seq, typename Rng>
    void fill_random(Seq&& seq, Rng& rng) {
        using V = std::decay_t<decltype(*std::begin(seq))>;

        auto u = make_udist<V>();
        for (auto& x: seq) { x = u(rng); }
    }

    template <typename Seq, typename Rng, typename B1, typename B2>
    void fill_random(Seq&& seq, Rng& rng, B1 lb, B2 ub) {
        using V = std::decay_t<decltype(*std::begin(seq))>;

        auto u = make_udist<V>(lb, ub);
        for (auto& x: seq) { x = u(rng); }
    }

    template <typename Simd, typename Rng, typename B1, typename B2, typename = std::enable_if_t<is_simd<Simd>::value>>
    void fill_random(Simd& s, Rng& rng, B1 lb, B2 ub) {
        using V = typename Simd::scalar_type;
        constexpr unsigned N = Simd::width;

        V v[N];
        fill_random(v, rng, lb, ub);
        s.copy_from(v);
    }

    template <typename Simd, typename Rng, typename = std::enable_if_t<is_simd<Simd>::value>>
    void fill_random(Simd& s, Rng& rng) {
        using V = typename Simd::scalar_type;
        constexpr unsigned N = Simd::width;

        V v[N];
        fill_random(v, rng);
        s.copy_from(v);
    }

    template <typename Simd>
    ::testing::AssertionResult simd_eq(Simd a, Simd b) {
        constexpr unsigned N = Simd::width;
        using V = typename Simd::scalar_type;

        V as[N], bs[N];
        a.copy_to(as);
        b.copy_to(bs);

        return ::testing::seq_eq(as, bs);
    }

    constexpr unsigned nrounds = 20u;
}

template <typename S>
struct simd_value: public ::testing::Test {};

TYPED_TEST_CASE_P(simd_value);

// Test agreement between simd::width(), simd::min_align() and corresponding type attributes.
TYPED_TEST_P(simd_value, meta) {
    using simd = TypeParam;
    using scalar = typename simd::scalar_type;

    ASSERT_EQ((int)simd::width, ::arb::simd::width(simd{}));
    ASSERT_EQ(simd::min_align, ::arb::simd::min_align(simd{}));

    EXPECT_LE(alignof(scalar), simd::min_align);
}

// Initialization and element access.
TYPED_TEST_P(simd_value, elements) {
    using simd = TypeParam;
    using scalar = typename simd::scalar_type;
    constexpr unsigned N = simd::width;

    std::minstd_rand rng(1001);

    // broadcast:
    simd a(2);
    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(2., a[i]);
    }

    // scalar assignment:
    a = 3;
    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(3, a[i]);
    }

    scalar bv[N], cv[N], dv[N];

    fill_random(bv, rng);
    fill_random(cv, rng);
    fill_random(dv, rng);

    // array initialization:
    simd b(bv);
    EXPECT_TRUE(testing::indexed_eq_n(N, bv, b));

    // array rvalue initialization:
    auto cv_copy = cv;
    simd c(std::move(cv));
    EXPECT_TRUE(testing::indexed_eq_n(N, cv_copy, c));

    // pointer initialization:
    simd d(&dv[0]);
    EXPECT_TRUE(testing::indexed_eq_n(N, dv, d));

    // copy construction:
    simd e(d);
    EXPECT_TRUE(testing::indexed_eq_n(N, dv, e));

    // copy assignment:
    b = d;
    EXPECT_TRUE(testing::indexed_eq_n(N, dv, b));
}

TYPED_TEST_P(simd_value, element_lvalue) {
    using simd = TypeParam;
    constexpr unsigned N = simd::width;

    simd a(3);
    ASSERT_GT(N, 1u);
    a[N-2] = 5;

    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(i==N-2? 5: 3, a[i]);
    }
}

TYPED_TEST_P(simd_value, copy_to_from) {
    using simd = TypeParam;
    using scalar = typename simd::scalar_type;
    constexpr unsigned N = simd::width;

    std::minstd_rand rng(1010);

    scalar buf1[N], buf2[N];
    fill_random(buf1, rng);
    fill_random(buf2, rng);

    simd s;
    s.copy_from(buf1);
    s.copy_to(buf2);

    EXPECT_TRUE(testing::indexed_eq_n(N, buf1, s));
    EXPECT_TRUE(testing::seq_eq(buf1, buf2));
}

TYPED_TEST_P(simd_value, copy_to_from_masked) {
    using simd = TypeParam;
    using mask = typename simd::simd_mask;
    using scalar = typename simd::scalar_type;
    constexpr unsigned N = simd::width;

    std::minstd_rand rng(1031);

    for (unsigned i = 0; i<nrounds; ++i) {
        scalar buf1[N], buf2[N], buf3[N], buf4[N];
        fill_random(buf1, rng);
        fill_random(buf2, rng);
        fill_random(buf3, rng);
        fill_random(buf4, rng);

        bool mbuf1[N], mbuf2[N];
        fill_random(mbuf1, rng);
        fill_random(mbuf2, rng);
        mask m1(mbuf1);
        mask m2(mbuf2);

        scalar expected[N];
        for (unsigned i = 0; i<N; ++i) {
            expected[i] = mbuf1[i]? buf2[i]: buf1[i];
        }

        simd s(buf1);
        where(m1, s) = indirect(buf2, N);
        EXPECT_TRUE(testing::indexed_eq_n(N, expected, s));

        for (unsigned i = 0; i<N; ++i) {
            if (!mbuf2[i]) expected[i] = buf3[i];
        }

        indirect(buf3, N) = where(m2, s);
        EXPECT_TRUE(testing::indexed_eq_n(N, expected, buf3));

        for (unsigned i = 0; i<N; ++i) {
            expected[i] = mbuf2[i]? buf1[i]: buf4[i];
        }

        simd b(buf1);
        indirect(buf4, N) = where(m2, b);
        EXPECT_TRUE(testing::indexed_eq_n(N, expected, buf4));
    }
}

TYPED_TEST_P(simd_value, construct_masked) {
    using simd = TypeParam;
    using mask = typename simd::simd_mask;
    using scalar = typename simd::scalar_type;
    constexpr unsigned N = simd::width;

    std::minstd_rand rng(1031);

    for (unsigned i = 0; i<nrounds; ++i) {
        scalar buf[N];
        fill_random(buf, rng);

        bool mbuf[N];
        fill_random(mbuf, rng);

        mask m(mbuf);
        simd s(buf, m);

        for (unsigned i = 0; i<N; ++i) {
            if (!mbuf[i]) continue;
            EXPECT_EQ(buf[i], s[i]);
        }
    }
}

TYPED_TEST_P(simd_value, arithmetic) {
    using simd = TypeParam;
    using scalar = typename simd::scalar_type;
    constexpr unsigned N = simd::width;

    std::minstd_rand rng(1002);
    scalar u[N], v[N], w[N], r[N];

    for (unsigned i = 0; i<nrounds; ++i) {
        fill_random(u, rng);
        fill_random(v, rng);
        fill_random(w, rng);

        scalar neg_u[N];
        for (unsigned i = 0; i<N; ++i) neg_u[i] = -u[i];

        scalar u_plus_v[N];
        for (unsigned i = 0; i<N; ++i) u_plus_v[i] = u[i]+v[i];

        scalar u_minus_v[N];
        for (unsigned i = 0; i<N; ++i) u_minus_v[i] = u[i]-v[i];

        scalar u_times_v[N];
        for (unsigned i = 0; i<N; ++i) u_times_v[i] = u[i]*v[i];

        scalar u_divide_v[N];
        for (unsigned i = 0; i<N; ++i) u_divide_v[i] = u[i]/v[i];

        scalar fma_u_v_w[N];
        for (unsigned i = 0; i<N; ++i) fma_u_v_w[i] = compat::fma(u[i],v[i],w[i]);

        simd us(u), vs(v), ws(w);

        (-us).copy_to(r);
        EXPECT_TRUE(testing::seq_eq(neg_u, r));

        (us+vs).copy_to(r);
        EXPECT_TRUE(testing::seq_eq(u_plus_v, r));

        (us-vs).copy_to(r);
        EXPECT_TRUE(testing::seq_eq(u_minus_v, r));

        (us*vs).copy_to(r);
        EXPECT_TRUE(testing::seq_eq(u_times_v, r));

        (us/vs).copy_to(r);
#if defined(__INTEL_COMPILER)
        // icpc will by default use an approximation for scalar division,
        // and a different one for vectorized scalar division; the latter,
        // in particular, is often out by 1 ulp for normal quotients.
        //
        // Unfortunately, we can't check at compile time the floating
        // point dodginess quotient.

        if (std::is_floating_point<scalar>::value) {
            EXPECT_TRUE(testing::seq_almost_eq<scalar>(u_divide_v, r));
        }
        else {
            EXPECT_TRUE(testing::seq_eq(u_divide_v, r));
        }
#else
        EXPECT_TRUE(testing::seq_eq(u_divide_v, r));
#endif

        (fma(us, vs, ws)).copy_to(r);
        EXPECT_TRUE(testing::seq_eq(fma_u_v_w, r));
    }
}

TYPED_TEST_P(simd_value, compound_assignment) {
    using simd = TypeParam;

    simd a, b, r;

    std::minstd_rand rng(1003);
    fill_random(a, rng);
    fill_random(b, rng);

    EXPECT_TRUE(simd_eq(a+b, (r = a)+=b));
    EXPECT_TRUE(simd_eq(a-b, (r = a)-=b));
    EXPECT_TRUE(simd_eq(a*b, (r = a)*=b));
    EXPECT_TRUE(simd_eq(a/b, (r = a)/=b));
}

TYPED_TEST_P(simd_value, comparison) {
    using simd = TypeParam;
    using mask = typename simd::simd_mask;
    constexpr unsigned N = simd::width;

    std::minstd_rand rng(1004);
    std::uniform_int_distribution<> sgn(-1, 1); // -1, 0 or 1.

    for (unsigned i = 0; i<nrounds; ++i) {
        int cmp[N];
        bool test[N];
        simd a, b;

        fill_random(b, rng);

        for (unsigned j = 0; j<N; ++j) {
            cmp[j] = sgn(rng);
            a[j] = b[j]+17*cmp[j];
        }

        mask gt = a>b;
        for (unsigned j = 0; j<N; ++j) { test[j] = cmp[j]>0; }
        EXPECT_TRUE(testing::indexed_eq_n(N, test, gt));

        mask geq = a>=b;
        for (unsigned j = 0; j<N; ++j) { test[j] = cmp[j]>=0; }
        EXPECT_TRUE(testing::indexed_eq_n(N, test, geq));

        mask lt = a<b;
        for (unsigned j = 0; j<N; ++j) { test[j] = cmp[j]<0; }
        EXPECT_TRUE(testing::indexed_eq_n(N, test, lt));

        mask leq = a<=b;
        for (unsigned j = 0; j<N; ++j) { test[j] = cmp[j]<=0; }
        EXPECT_TRUE(testing::indexed_eq_n(N, test, leq));

        mask eq = a==b;
        for (unsigned j = 0; j<N; ++j) { test[j] = cmp[j]==0; }
        EXPECT_TRUE(testing::indexed_eq_n(N, test, eq));

        mask ne = a!=b;
        for (unsigned j = 0; j<N; ++j) { test[j] = cmp[j]!=0; }
        EXPECT_TRUE(testing::indexed_eq_n(N, test, ne));
    }
}

TYPED_TEST_P(simd_value, mask_elements) {
    using simd = TypeParam;
    using mask = typename simd::simd_mask;
    constexpr unsigned N = simd::width;

    std::minstd_rand rng(1005);

    // bool broadcast:
    mask a(true);
    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(true, a[i]);
    }

    // scalar assignment:
    mask d;
    d = false;
    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(false, d[i]);
    }
    d = true;
    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(true, d[i]);
    }

    for (unsigned i = 0; i<nrounds; ++i) {
        bool bv[N], cv[N], dv[N];

        fill_random(bv, rng);
        fill_random(cv, rng);
        fill_random(dv, rng);

        // array initialization:
        mask b(bv);
        EXPECT_TRUE(testing::indexed_eq_n(N, bv, b));

        // array rvalue initialization:
        auto cv_copy = cv;
        mask c(std::move(cv));
        EXPECT_TRUE(testing::indexed_eq_n(N, cv_copy, c));

        // pointer initialization:
        mask d(&dv[0]);
        EXPECT_TRUE(testing::indexed_eq_n(N, dv, d));

        // copy construction:
        mask e(d);
        EXPECT_TRUE(testing::indexed_eq_n(N, dv, e));

        // copy assignment:
        b = d;
        EXPECT_TRUE(testing::indexed_eq_n(N, dv, b));
    }
}

TYPED_TEST_P(simd_value, mask_element_lvalue) {
    using simd = TypeParam;
    using mask = typename simd::simd_mask;
    constexpr unsigned N = simd::width;

    std::minstd_rand rng(1006);

    for (unsigned i = 0; i<nrounds; ++i) {
        bool v[N];
        fill_random(v, rng);

        mask m(v);
        for (unsigned j = 0; j<N; ++j) {
            bool b = v[j];
            m[j] = !b;
            v[j] = !b;

            EXPECT_EQ(m[j], !b);
            EXPECT_TRUE(testing::indexed_eq_n(N, v, m));

            m[j] = b;
            v[j] = b;
            EXPECT_EQ(m[j], b);
            EXPECT_TRUE(testing::indexed_eq_n(N, v, m));
        }
    }
}

TYPED_TEST_P(simd_value, mask_copy_to_from) {
    using simd = TypeParam;
    using simd_mask = typename simd::simd_mask;
    constexpr unsigned N = simd::width;

    std::minstd_rand rng(1012);

    for (unsigned i = 0; i<nrounds; ++i) {
        bool buf1[N], buf2[N];
        fill_random(buf1, rng);
        fill_random(buf2, rng);

        simd_mask m;
        m.copy_from(buf1);
        m.copy_to(buf2);

        EXPECT_TRUE(testing::indexed_eq_n(N, buf1, m));
        EXPECT_TRUE(testing::seq_eq(buf1, buf2));
    }
}

TYPED_TEST_P(simd_value, mask_unpack) {
    using simd = TypeParam;
    using mask = typename simd::simd_mask;
    constexpr unsigned N = simd::width;

    std::minstd_rand rng(1035);
    std::uniform_int_distribution<unsigned long long> U(0, (1ull<<N)-1);

    for (unsigned i = 0; i<nrounds; ++i) {
        unsigned long long packed = U(rng);
        bool b[N];
        mask::unpack(packed).copy_to(b);

        for (unsigned j = 0; j<N; ++j) {
            EXPECT_EQ((bool)(packed&(1ull<<j)), b[j]);
        }
    }
}

TYPED_TEST_P(simd_value, maths) {
    // min, max, abs tests valid for both fp and int types.

    using simd = TypeParam;
    using scalar = typename simd::scalar_type;
    constexpr unsigned N = simd::width;

    std::minstd_rand rng(1013);

    for (unsigned i = 0; i<nrounds; ++i) {
        scalar a[N], b[N], test[N];
        fill_random(a, rng);
        fill_random(b, rng);

        simd as(a), bs(b);

        for (unsigned j = 0; j<N; ++j) { test[j] = std::abs(a[j]); }
        EXPECT_TRUE(testing::indexed_eq_n(N, test, abs(as)));

        for (unsigned j = 0; j<N; ++j) { test[j] = std::min(a[j], b[j]); }
        EXPECT_TRUE(testing::indexed_eq_n(N, test, min(as, bs)));

        for (unsigned j = 0; j<N; ++j) { test[j] = std::max(a[j], b[j]); }
        EXPECT_TRUE(testing::indexed_eq_n(N, test, max(as, bs)));
    }
}

TYPED_TEST_P(simd_value, reductions) {
    // Only addition for now.

    using simd = TypeParam;
    using scalar = typename simd::scalar_type;
    constexpr unsigned N = simd::width;

    std::minstd_rand rng(1041);

    for (unsigned i = 0; i<nrounds; ++i) {
        scalar a[N], test = 0;

        // To avoid discrepancies due to catastrophic cancelation,
        // keep f.p. values non-negative.

        if (std::is_floating_point<scalar>::value) {
            fill_random(a, rng, 0, 1);
        }
        else {
            fill_random(a, rng);
        }

        simd as(a);

        for (unsigned j = 0; j<N; ++j) { test += a[j]; }
        EXPECT_TRUE(testing::almost_eq(test, as.sum()));
    }
}

TYPED_TEST_P(simd_value, simd_array_cast) {
    // Test conversion to/from array of scalar type.

    using simd = TypeParam;
    using scalar = typename simd::scalar_type;
    constexpr unsigned N = simd::width;

    std::minstd_rand rng(1032);

    for (unsigned i = 0; i<nrounds; ++i) {
        std::array<scalar, N> a;

        fill_random(a, rng);
        simd as = simd_cast<simd>(a);
        EXPECT_TRUE(testing::indexed_eq_n(N, as, a));
        EXPECT_TRUE(testing::seq_eq(a, simd_cast<std::array<scalar, N>>(as)));
    }
}

REGISTER_TYPED_TEST_CASE_P(simd_value, meta, elements, element_lvalue, copy_to_from, copy_to_from_masked, construct_masked, arithmetic, compound_assignment, comparison, mask_elements, mask_element_lvalue, mask_copy_to_from, mask_unpack, maths, simd_array_cast, reductions);

typedef ::testing::Types<

#ifdef __AVX__
    simd<int, 4, simd_abi::avx>,
    simd<double, 4, simd_abi::avx>,
#endif
#ifdef __AVX2__
    simd<int, 4, simd_abi::avx2>,
    simd<double, 4, simd_abi::avx2>,
#endif
#ifdef __AVX512F__
    simd<int, 8, simd_abi::avx512>,
    simd<double, 8, simd_abi::avx512>,
#endif
#if defined(__ARM_NEON)
    simd<int, 2, simd_abi::neon>,
    simd<double, 2, simd_abi::neon>,
#endif

    simd<int, 4, simd_abi::generic>,
    simd<double, 4, simd_abi::generic>,
    simd<float, 16, simd_abi::generic>,

    simd<int, 4, simd_abi::default_abi>,
    simd<double, 4, simd_abi::default_abi>,
    simd<int, 8, simd_abi::default_abi>,
    simd<double, 8, simd_abi::default_abi>
> simd_test_types;

INSTANTIATE_TYPED_TEST_CASE_P(S, simd_value, simd_test_types);

// FP-only SIMD value tests (maths).

template <typename S>
struct simd_fp_value: public ::testing::Test {};

TYPED_TEST_CASE_P(simd_fp_value);

TYPED_TEST_P(simd_fp_value, fp_maths) {
    using simd = TypeParam;
    using fp = typename simd::scalar_type;
    constexpr unsigned N = simd::width;

    std::minstd_rand rng(1014);

    for (unsigned i = 0; i<nrounds; ++i) {
        fp epsilon = std::numeric_limits<fp>::epsilon();
        int min_exponent = std::numeric_limits<fp>::min_exponent;
        int max_exponent = std::numeric_limits<fp>::max_exponent;

        fp u[N], v[N], r[N];

        // Trigonometric functions (sin, cos):
        fill_random(u, rng);

        fp sin_u[N];
        for (unsigned i = 0; i<N; ++i) sin_u[i] = std::sin(u[i]);
        sin(simd(u)).copy_to(r);
        EXPECT_TRUE(testing::seq_almost_eq<fp>(sin_u, r));

        fp cos_u[N];
        for (unsigned i = 0; i<N; ++i) cos_u[i] = std::cos(u[i]);
        cos(simd(u)).copy_to(r);
        EXPECT_TRUE(testing::seq_almost_eq<fp>(cos_u, r));

        // Logarithms (natural log):
        fill_random(u, rng, -max_exponent*std::log(2.), max_exponent*std::log(2.));
        for (auto& x: u) {
            x = std::exp(x);
            // SIMD log implementation may treat subnormal as zero
            if (std::fpclassify(x)==FP_SUBNORMAL) x = 0;
        }

        fp log_u[N];
        for (unsigned i = 0; i<N; ++i) log_u[i] = std::log(u[i]);
        log(simd(u)).copy_to(r);
        EXPECT_TRUE(testing::seq_almost_eq<fp>(log_u, r));

        // Exponential functions (exp, expm1, exprelr):

        // Use max_exponent to get coverage over finite domain.
        fp exp_min_arg = min_exponent*std::log(2.);
        fp exp_max_arg = max_exponent*std::log(2.);
        fill_random(u, rng, exp_min_arg, exp_max_arg);

        fp exp_u[N];
        for (unsigned i = 0; i<N; ++i) exp_u[i] = std::exp(u[i]);
        exp(simd(u)).copy_to(r);
        EXPECT_TRUE(testing::seq_almost_eq<fp>(exp_u, r));

        fp expm1_u[N];
        for (unsigned i = 0; i<N; ++i) expm1_u[i] = std::expm1(u[i]);
        expm1(simd(u)).copy_to(r);
        EXPECT_TRUE(testing::seq_almost_eq<fp>(expm1_u, r));

        fp exprelr_u[N];
        for (unsigned i = 0; i<N; ++i) {
            exprelr_u[i] = u[i]+fp(1)==fp(1)? fp(1): u[i]/(std::expm1(u[i]));
        }
        exprelr(simd(u)).copy_to(r);
        EXPECT_TRUE(testing::seq_almost_eq<fp>(exprelr_u, r));

        // Test expm1 and exprelr with small (magnitude < fp epsilon) values.
        fill_random(u, rng, -epsilon, epsilon);
        fp expm1_u_small[N];
        for (unsigned i = 0; i<N; ++i) {
            expm1_u_small[i] = std::expm1(u[i]);
            EXPECT_NEAR(u[i], expm1_u_small[i], std::abs(4*u[i]*epsilon)); // just to confirm!
        }
        expm1(simd(u)).copy_to(r);
        EXPECT_TRUE(testing::seq_almost_eq<fp>(expm1_u_small, r));

        fp exprelr_u_small[N];
        for (unsigned i = 0; i<N; ++i) exprelr_u_small[i] = 1;
        exprelr(simd(u)).copy_to(r);
        EXPECT_TRUE(testing::seq_almost_eq<fp>(exprelr_u_small, r));

        // Test zero result for highly negative exponents.
        fill_random(u, rng, 4*exp_min_arg, 2*exp_min_arg);
        fp exp_u_very_negative[N];
        for (unsigned i = 0; i<N; ++i) exp_u_very_negative[i] = std::exp(u[i]);
        exp(simd(u)).copy_to(r);
        EXPECT_TRUE(testing::seq_almost_eq<fp>(exp_u_very_negative, r));

        // Power function:

        // Non-negative base, arbitrary exponent.
        fill_random(u, rng, 0., std::exp(1));
        fill_random(v, rng, exp_min_arg, exp_max_arg);
        fp pow_u_pos_v[N];
        for (unsigned i = 0; i<N; ++i) pow_u_pos_v[i] = std::pow(u[i], v[i]);
        pow(simd(u), simd(v)).copy_to(r);
        EXPECT_TRUE(testing::seq_almost_eq<fp>(pow_u_pos_v, r));

        // Arbitrary base, small magnitude integer exponent.
        fill_random(u, rng);
        int int_exponent[N];
        fill_random(int_exponent, rng, -2, 2);
        for (unsigned i = 0; i<N; ++i) v[i] = int_exponent[i];
        fp pow_u_v_int[N];
        for (unsigned i = 0; i<N; ++i) pow_u_v_int[i] = std::pow(u[i], v[i]);
        pow(simd(u), simd(v)).copy_to(r);
        EXPECT_TRUE(testing::seq_almost_eq<fp>(pow_u_v_int, r));
    }

    // The tests can cause floating point exceptions, which may set errno to nonzero
    // value.
    // Reset errno so that subsequent tests are not affected.
    errno = 0;
}

// Check special function behaviour for specific values including
// qNAN, infinity etc.

TYPED_TEST_P(simd_fp_value, exp_special_values) {
    using simd = TypeParam;
    using fp = typename simd::scalar_type;
    constexpr unsigned N = simd::width;

    using limits = std::numeric_limits<fp>;

    constexpr fp inf = limits::infinity();
    constexpr fp eps = limits::epsilon();
    constexpr fp largest = limits::max();
    constexpr fp normal_least = limits::min();
    constexpr fp denorm_least = limits::denorm_min();
    constexpr fp qnan = limits::quiet_NaN();

    const fp exp_minarg = std::log(normal_least);
    const fp exp_maxarg = std::log(largest);

    fp values[] = { inf, -inf, eps, -eps,
                    eps/2, -eps/2, 0., -0.,
                    1., -1., 2., -2.,
                    normal_least, denorm_least, -normal_least, -denorm_least,
                    exp_minarg, exp_maxarg, qnan, -qnan };

    constexpr unsigned n_values = sizeof(values)/sizeof(fp);
    constexpr unsigned n_packed = (n_values+N-1)/N;
    fp data[n_packed][N];

    std::fill((fp *)data, (fp *)data+N*n_packed, fp(0));
    std::copy(std::begin(values), std::end(values), (fp *)data);

    for (unsigned i = 0; i<n_packed; ++i) {
        fp expected[N], result[N];
        for (unsigned j = 0; j<N; ++j) {
            expected[j] = std::exp(data[i][j]);
        }

        simd s(data[i]);
        s = exp(s);
        s.copy_to(result);

        EXPECT_TRUE(testing::seq_almost_eq<fp>(expected, result));
    }
}

TYPED_TEST_P(simd_fp_value, expm1_special_values) {
    using simd = TypeParam;
    using fp = typename simd::scalar_type;
    constexpr unsigned N = simd::width;

    using limits = std::numeric_limits<fp>;

    constexpr fp inf = limits::infinity();
    constexpr fp eps = limits::epsilon();
    constexpr fp largest = limits::max();
    constexpr fp normal_least = limits::min();
    constexpr fp denorm_least = limits::denorm_min();
    constexpr fp qnan = limits::quiet_NaN();

    const fp expm1_minarg = std::nextafter(std::log(eps/4), fp(0));
    const fp expm1_maxarg = std::log(largest);

    fp values[] = { inf, -inf, eps, -eps,
                    eps/2, -eps/2, 0., -0.,
                    1., -1., 2., -2.,
                    normal_least, denorm_least, -normal_least, -denorm_least,
                    expm1_minarg, expm1_maxarg, qnan, -qnan };

    constexpr unsigned n_values = sizeof(values)/sizeof(fp);
    constexpr unsigned n_packed = (n_values+N-1)/N;
    fp data[n_packed][N];

    std::fill((fp *)data, (fp *)data+N*n_packed, fp(0));
    std::copy(std::begin(values), std::end(values), (fp *)data);

    for (unsigned i = 0; i<n_packed; ++i) {
        fp expected[N], result[N];
        for (unsigned j = 0; j<N; ++j) {
            expected[j] = std::expm1(data[i][j]);
        }

        simd s(data[i]);
        s = expm1(s);
        s.copy_to(result);

        EXPECT_TRUE(testing::seq_almost_eq<fp>(expected, result));
    }
}

TYPED_TEST_P(simd_fp_value, log_special_values) {
    using simd = TypeParam;
    using fp = typename simd::scalar_type;
    constexpr unsigned N = simd::width;

    using limits = std::numeric_limits<fp>;

    // NOTE: simd log implementations may treat subnormal
    // numbers as zero, so omit the denorm_least tests...

    constexpr fp inf = limits::infinity();
    constexpr fp eps = limits::epsilon();
    constexpr fp largest = limits::max();
    constexpr fp normal_least = limits::min();
    //constexpr fp denorm_least = limits::denorm_min();
    constexpr fp qnan = limits::quiet_NaN();

    fp values[] = { inf, -inf, eps, -eps,
                    eps/2, -eps/2, 0., -0.,
                    1., -1., 2., -2.,
                    //normal_least, denorm_least, -normal_least, -denorm_least,
                    normal_least, -normal_least,
                    qnan, -qnan, largest };

    constexpr unsigned n_values = sizeof(values)/sizeof(fp);
    constexpr unsigned n_packed = (n_values+N-1)/N;
    fp data[n_packed][N];

    std::fill((fp *)data, (fp *)data+N*n_packed, fp(0));
    std::copy(std::begin(values), std::end(values), (fp *)data);

    for (unsigned i = 0; i<n_packed; ++i) {
        fp expected[N], result[N];
        for (unsigned j = 0; j<N; ++j) {
            expected[j] = std::log(data[i][j]);
        }

        simd s(data[i]);
        s = log(s);
        s.copy_to(result);

        EXPECT_TRUE(testing::seq_almost_eq<fp>(expected, result));
    }
}

REGISTER_TYPED_TEST_CASE_P(simd_fp_value, fp_maths, exp_special_values, expm1_special_values, log_special_values);

typedef ::testing::Types<

#ifdef __AVX__
    simd<double, 4, simd_abi::avx>,
#endif
#ifdef __AVX2__
    simd<double, 4, simd_abi::avx2>,
#endif
#ifdef __AVX512F__
    simd<double, 8, simd_abi::avx512>,
#endif
#ifdef __ARM_NEON
    simd<double, 2, simd_abi::neon>,
#endif

    simd<float, 2, simd_abi::generic>,
    simd<double, 4, simd_abi::generic>,
    simd<float, 8, simd_abi::generic>,

    simd<double, 4, simd_abi::default_abi>,
    simd<double, 8, simd_abi::default_abi>
> simd_fp_test_types;

INSTANTIATE_TYPED_TEST_CASE_P(S, simd_fp_value, simd_fp_test_types);

// Gather/scatter tests.

template <typename A, typename B>
struct simd_and_index {
    using simd = A;
    using simd_index = B;
};

template <typename SI>
struct simd_indirect: public ::testing::Test {};

TYPED_TEST_CASE_P(simd_indirect);

TYPED_TEST_P(simd_indirect, gather) {
    using simd = typename TypeParam::simd;
    using simd_index = typename TypeParam::simd_index;

    constexpr unsigned N = simd::width;
    using scalar = typename simd::scalar_type;
    using index = typename simd_index::scalar_type;

    std::minstd_rand rng(1011);

    constexpr std::size_t buflen = 1000;

    for (unsigned i = 0; i<nrounds; ++i) {
        scalar array[buflen];
        index offset[N];

        fill_random(array, rng);
        fill_random(offset, rng, 0, (int)(buflen-1));

        simd s = simd_cast<simd>(indirect(array, simd_index(offset), N));

        scalar test[N];
        for (unsigned j = 0; j<N; ++j) {
            test[j] = array[offset[j]];
        }

        EXPECT_TRUE(::testing::indexed_eq_n(N, test, s));
    }
}

TYPED_TEST_P(simd_indirect, masked_gather) {
    using simd = typename TypeParam::simd;
    using simd_index = typename TypeParam::simd_index;
    using simd_mask = typename simd::simd_mask;

    constexpr unsigned N = simd::width;
    using scalar = typename simd::scalar_type;
    using index = typename simd_index::scalar_type;

    std::minstd_rand rng(1011);

    constexpr std::size_t buflen = 1000;

    for (unsigned i = 0; i<nrounds; ++i) {
        scalar array[buflen], original[N], test[N];
        index offset[N];
        bool mask[N];

        fill_random(array, rng);
        fill_random(original, rng);
        fill_random(offset, rng, 0, (int)(buflen-1));
        fill_random(mask, rng);

        for (unsigned j = 0; j<N; ++j) {
            test[j] = mask[j]? array[offset[j]]: original[j];
        }

        simd s(original);
        simd_mask m(mask);

        where(m, s) = indirect(array, simd_index(offset), N);

        EXPECT_TRUE(::testing::indexed_eq_n(N, test, s));
    }
}

TYPED_TEST_P(simd_indirect, scatter) {
    using simd = typename TypeParam::simd;
    using simd_index = typename TypeParam::simd_index;

    constexpr unsigned N = simd::width;
    using scalar = typename simd::scalar_type;
    using index = typename simd_index::scalar_type;

    std::minstd_rand rng(1011);

    constexpr std::size_t buflen = 1000;

    for (unsigned i = 0; i<nrounds; ++i) {
        scalar array[buflen], test[buflen], values[N];
        index offset[N];

        fill_random(array, rng);
        fill_random(values, rng);
        fill_random(offset, rng, 0, (int)(buflen-1));

        for (unsigned j = 0; j<buflen; ++j) {
            test[j] = array[j];
        }
        for (unsigned j = 0; j<N; ++j) {
            test[offset[j]] = values[j];
        }

        simd s(values);
        indirect(array, simd_index(offset), N) = s;

        EXPECT_TRUE(::testing::indexed_eq_n(buflen, test, array));
    }
}

TYPED_TEST_P(simd_indirect, masked_scatter) {
    using simd = typename TypeParam::simd;
    using simd_index = typename TypeParam::simd_index;
    using simd_mask = typename simd::simd_mask;

    constexpr unsigned N = simd::width;
    using scalar = typename simd::scalar_type;
    using index = typename simd_index::scalar_type;

    std::minstd_rand rng(1011);

    constexpr std::size_t buflen = 1000;

    for (unsigned i = 0; i<nrounds; ++i) {
        scalar array[buflen], test[buflen], values[N];
        index offset[N];
        bool mask[N];

        fill_random(array, rng);
        fill_random(values, rng);
        fill_random(offset, rng, 0, (int)(buflen-1));
        fill_random(mask, rng);

        for (unsigned j = 0; j<buflen; ++j) {
            test[j] = array[j];
        }
        for (unsigned j = 0; j<N; ++j) {
            if (mask[j]) { test[offset[j]] = values[j]; }
        }

        simd s(values);
        simd_mask m(mask);
        indirect(array, simd_index(offset), N) = where(m, s);

        EXPECT_TRUE(::testing::indexed_eq_n(buflen, test, array));

        for (unsigned j = 0; j<buflen; ++j) {
            array[j] = test[j];
        }

        simd v(values);
        indirect(array, simd_index(offset), N) = where(m, v);

        EXPECT_TRUE(::testing::indexed_eq_n(buflen, test, array));
    }
}

TYPED_TEST_P(simd_indirect, add_and_subtract) {
    using simd = typename TypeParam::simd;
    using simd_index = typename TypeParam::simd_index;

    constexpr unsigned N = simd::width;
    using scalar = typename simd::scalar_type;
    using index = typename simd_index::scalar_type;

    std::minstd_rand rng(1011);

    constexpr std::size_t buflen = 1000;

    for (unsigned i = 0; i<nrounds; ++i) {
        scalar array[buflen], test[buflen], values[N];
        index offset[N];

        fill_random(array, rng);
        fill_random(values, rng);
        fill_random(offset, rng, 0, (int)(buflen-1));

        for (unsigned j = 0; j<buflen; ++j) {
            test[j] = array[j];
        }
        for (unsigned j = 0; j<N; ++j) {
            test[offset[j]] += values[j];
        }

        indirect(array, simd_index(offset), N) += simd(values);
        EXPECT_TRUE(::testing::indexed_eq_n(buflen, test, array));

        fill_random(offset, rng, 0, (int)(buflen-1));

        for (unsigned j = 0; j<buflen; ++j) {
            test[j] = array[j];
        }
        for (unsigned j = 0; j<N; ++j) {
            test[offset[j]] -= values[j];
        }

        indirect(array, simd_index(offset), N) -= simd(values);
        EXPECT_TRUE(::testing::indexed_eq_n(buflen, test, array));
    }
}

template <typename X>
bool unique_elements(const X& xs) {
    using std::begin;
// WOOPWOOP
    std::unordered_set<std::decay_t<decltype(*begin(xs))>> set;
    for (auto& x: xs) {
        if (!set.insert(x).second) return false;
    }
    return true;
}

TYPED_TEST_P(simd_indirect, constrained_add) {
    using simd = typename TypeParam::simd;
    using simd_index = typename TypeParam::simd_index;

    constexpr unsigned N = simd::width;
    using scalar = typename simd::scalar_type;
    using index = typename simd_index::scalar_type;

    std::minstd_rand rng(1011);

    constexpr std::size_t buflen = 1000;

    for (unsigned i = 0; i<nrounds; ++i) {
        scalar array[buflen], test[buflen], values[N];
        index offset[N];

        fill_random(array, rng);
        fill_random(values, rng);

        auto make_test_array = [&]() {
            for (unsigned j = 0; j<buflen; ++j) {
                test[j] = array[j];
            }
            for (unsigned j = 0; j<N; ++j) {
                test[offset[j]] += values[j];
            }
        };

        // Independent:

        do {
            fill_random(offset, rng, 0, (int)(buflen-1));
        } while (!unique_elements(offset));

        make_test_array();
        indirect(array, simd_index(offset), N, index_constraint::independent) += simd(values);

        EXPECT_TRUE(::testing::indexed_eq_n(buflen, test, array));

        // Contiguous:

        offset[0] = make_udist<index>(0, (int)(buflen)-N)(rng);
        for (unsigned j = 1; j<N; ++j) {
            offset[j] = offset[0]+j;
        }

        make_test_array();
        indirect(array, simd_index(offset), N, index_constraint::contiguous) += simd(values);

        EXPECT_TRUE(::testing::indexed_eq_n(buflen, test, array));

        // Constant:

        for (unsigned j = 1; j<N; ++j) {
            offset[j] = offset[0];
        }

        // Reduction can be done in a different order, so 1) use approximate test
        // and 2) keep f.p. values non-negative to avoid catastrophic cancellation.

        if (std::is_floating_point<scalar>::value) {
            fill_random(array, rng, 0, 1);
            fill_random(values, rng, 0, 1);
        }

        make_test_array();
        indirect(array, simd_index(offset), N, index_constraint::constant) += simd(values);

        EXPECT_TRUE(::testing::indexed_almost_eq_n(buflen, test, array));

    }
}

REGISTER_TYPED_TEST_CASE_P(simd_indirect, gather, masked_gather, scatter, masked_scatter, add_and_subtract, constrained_add);

typedef ::testing::Types<

#ifdef __AVX__
    simd_and_index<simd<double, 4, simd_abi::avx>,
                   simd<int, 4, simd_abi::avx>>,

    simd_and_index<simd<int, 4, simd_abi::avx>,
                   simd<int, 4, simd_abi::avx>>,
#endif

#ifdef __AVX2__
    simd_and_index<simd<double, 4, simd_abi::avx2>,
                   simd<int, 4, simd_abi::avx2>>,

    simd_and_index<simd<int, 4, simd_abi::avx2>,
                   simd<int, 4, simd_abi::avx2>>,
#endif

#ifdef __AVX512F__
    simd_and_index<simd<double, 8, simd_abi::avx512>,
                   simd<int, 8, simd_abi::avx512>>,

    simd_and_index<simd<int, 8, simd_abi::avx512>,
                   simd<int, 8, simd_abi::avx512>>,
#endif
#ifdef __ARM_NEON
    simd_and_index<simd<double, 2, simd_abi::neon>,
                   simd<int, 2, simd_abi::neon>>,

    simd_and_index<simd<int, 2, simd_abi::neon>,
                   simd<int, 2, simd_abi::neon>>,
#endif

    simd_and_index<simd<float, 4, simd_abi::generic>,
                   simd<std::int64_t, 4, simd_abi::generic>>,

    simd_and_index<simd<double, 8, simd_abi::generic>,
                   simd<unsigned, 8, simd_abi::generic>>,

    simd_and_index<simd<double, 4, simd_abi::default_abi>,
                   simd<int, 4, simd_abi::default_abi>>,

    simd_and_index<simd<double, 8, simd_abi::default_abi>,
                   simd<int, 8, simd_abi::default_abi>>
> simd_indirect_test_types;

INSTANTIATE_TYPED_TEST_CASE_P(S, simd_indirect, simd_indirect_test_types);


// SIMD cast tests

template <typename A, typename B>
struct simd_pair {
    using simd_first = A;
    using simd_second = B;
};

template <typename SI>
struct simd_casting: public ::testing::Test {};

TYPED_TEST_CASE_P(simd_casting);

TYPED_TEST_P(simd_casting, cast) {
    using simd_x = typename TypeParam::simd_first;
    using simd_y = typename TypeParam::simd_second;

    constexpr unsigned N = simd_x::width;
    using scalar_x = typename simd_x::scalar_type;
    using scalar_y = typename simd_y::scalar_type;

    std::minstd_rand rng(1011);

    for (unsigned i = 0; i<nrounds; ++i) {
        scalar_x x[N], test_x[N];
        scalar_y y[N], test_y[N];

        fill_random(x, rng);
        fill_random(y, rng);

        for (unsigned j = 0; j<N; ++j) {
            test_y[j] = static_cast<scalar_y>(x[j]);
            test_x[j] = static_cast<scalar_x>(y[j]);
        }

        simd_x xs(x);
        simd_y ys(y);

        EXPECT_TRUE(testing::indexed_eq_n(N, test_y, simd_cast<simd_y>(xs)));
        EXPECT_TRUE(testing::indexed_eq_n(N, test_x, simd_cast<simd_x>(ys)));
    }
}

REGISTER_TYPED_TEST_CASE_P(simd_casting, cast);


typedef ::testing::Types<

#ifdef __AVX__
    simd_pair<simd<double, 4, simd_abi::avx>,
              simd<int, 4, simd_abi::avx>>,
#endif

#ifdef __AVX2__
    simd_pair<simd<double, 4, simd_abi::avx2>,
              simd<int, 4, simd_abi::avx2>>,
#endif

#ifdef __AVX512F__
    simd_pair<simd<double, 8, simd_abi::avx512>,
              simd<int, 8, simd_abi::avx512>>,
#endif
#ifdef __ARM_NEON
    simd_pair<simd<double, 2, simd_abi::neon>,
              simd<int, 2, simd_abi::neon>>,
#endif

    simd_pair<simd<double, 4, simd_abi::default_abi>,
              simd<float, 4, simd_abi::default_abi>>
> simd_casting_test_types;

INSTANTIATE_TYPED_TEST_CASE_P(S, simd_casting, simd_casting_test_types);


// Sizeless simd types API tests

template <typename T, typename V, unsigned N>
struct simd_t {
    using simd_type = T;
    using scalar_type = V;
    static constexpr unsigned width = N;
};

template <typename T, typename I, typename M>
struct simd_types_t {
    using simd_value = T;
    using simd_index = I;
    using simd_value_mask =  M;
};

template <typename SI>
struct sizeless_api: public ::testing::Test {};

TYPED_TEST_CASE_P(sizeless_api);

TYPED_TEST_P(sizeless_api, construct) {
    using simd_value   = typename TypeParam::simd_value::simd_type;
    using scalar_value = typename TypeParam::simd_value::scalar_type;

    using simd_index   = typename TypeParam::simd_index::simd_type;
    using scalar_index = typename TypeParam::simd_index::scalar_type;

    constexpr unsigned N = TypeParam::simd_value::width;

    std::minstd_rand rng(1001);

    {
        scalar_value a_in[N], a_out[N];
        fill_random(a_in, rng);

        simd_value av = simd_cast<simd_value>(indirect(a_in, N));

        indirect(a_out, N) = av;

        EXPECT_TRUE(testing::indexed_eq_n(N, a_in, a_out));
    }
    {
        scalar_value a_in[2*N], b_in[N], a_out[N], exp_0[N], exp_1[2*N];
        fill_random(a_in, rng);
        fill_random(b_in, rng);

        scalar_index idx[N];

        auto make_test_indirect2simd = [&]() {
            for (unsigned i = 0; i<N; ++i) {
                exp_0[i] = a_in[idx[i]];
            }
        };

        auto make_test_simd2indirect = [&]() {
            for (unsigned i = 0; i<2*N; ++i) {
                exp_1[i] = a_in[i];
            }
            for (unsigned i = 0; i<N; ++i) {
                exp_1[idx[i]] = b_in[i];
            }
        };

        // Independent
        for (unsigned i = 0; i < N; ++i) {
            idx[i] = i*2;
        }
        simd_index idxv = simd_cast<simd_index>(indirect(idx, N));

        make_test_indirect2simd();

        simd_value av   = simd_cast<simd_value>(indirect(a_in, idxv, N, index_constraint::independent));
        indirect(a_out, N) = av;

        EXPECT_TRUE(testing::indexed_eq_n(N, exp_0, a_out));

        make_test_simd2indirect();

        indirect(a_in, idxv, N, index_constraint::independent) = simd_cast<simd_value>(indirect(b_in, N));

        EXPECT_TRUE(testing::indexed_eq_n(2*N, exp_1, a_in));

        // contiguous
        for (unsigned i = 0; i < N; ++i) {
            idx[i] = i;
        }
        idxv = simd_cast<simd_index>(indirect(idx, N));

        make_test_indirect2simd();

        av   = simd_cast<simd_value>(indirect(a_in, idxv, N, index_constraint::contiguous));
        indirect(a_out, N) = av;

        EXPECT_TRUE(testing::indexed_eq_n(N, exp_0, a_out));

        make_test_simd2indirect();

        indirect(a_in, idxv, N, index_constraint::contiguous) = simd_cast<simd_value>(indirect(b_in, N));

        EXPECT_TRUE(testing::indexed_eq_n(2*N, exp_1, a_in));

        // none
        for (unsigned i = 0; i < N; ++i) {
            idx[i] = i/2;
        }

        idxv = simd_cast<simd_index>(indirect(idx, N));

        make_test_indirect2simd();

        av   = simd_cast<simd_value>(indirect(a_in, idxv, N, index_constraint::none));
        indirect(a_out, N) = av;

        EXPECT_TRUE(testing::indexed_eq_n(N, exp_0, a_out));

        make_test_simd2indirect();

        indirect(a_in, idxv, N, index_constraint::none) = simd_cast<simd_value>(indirect(b_in, N));

        EXPECT_TRUE(testing::indexed_eq_n(2*N, exp_1, a_in));

        // constant
        for (unsigned i = 0; i < N; ++i) {
            idx[i] = 0;
        }

        idxv = simd_cast<simd_index>(indirect(idx, N));

        make_test_indirect2simd();

        av   = simd_cast<simd_value>(indirect(a_in, idxv, N, index_constraint::constant));
        indirect(a_out, N) = av;

        EXPECT_TRUE(testing::indexed_eq_n(N, exp_0, a_out));

        make_test_simd2indirect();

        indirect(a_in, idxv, N, index_constraint::constant) = simd_cast<simd_value>(indirect(b_in, N));

        EXPECT_TRUE(testing::indexed_eq_n(2*N, exp_1, a_in));
    }
}

TYPED_TEST_P(sizeless_api, where_exp) {
    using simd_value   = typename TypeParam::simd_value::simd_type;
    using scalar_value = typename TypeParam::simd_value::scalar_type;

    using simd_index   = typename TypeParam::simd_index::simd_type;
    using scalar_index = typename TypeParam::simd_index::scalar_type;

    using mask_simd    = typename TypeParam::simd_value_mask::simd_type;

    constexpr unsigned N = TypeParam::simd_value::width;

    std::minstd_rand rng(201);

    bool m[N];
    fill_random(m, rng);
    mask_simd mv = simd_cast<mask_simd>(indirect(m, N));

    scalar_value a[N], b[N], exp[N];
    fill_random(a, rng);
    fill_random(b, rng);

    {
        bool c[N];
        indirect(c, N) = mv;
        EXPECT_TRUE(testing::indexed_eq_n(N, c, m));
    }

    // where = constant
    {
        scalar_value c[N];

        simd_value av = simd_cast<simd_value>(indirect(a, N));

        where(mv, av) = 42.3;
        indirect(c, N) = av;

        for (unsigned i = 0; i<N; ++i) {
            exp[i] = m[i]? 42.3 : a[i];
        }
        EXPECT_TRUE(testing::indexed_eq_n(N, c, exp));
    }

    // where = simd
    {
        scalar_value c[N];

        simd_value av = simd_cast<simd_value>(indirect(a, N));
        simd_value bv = simd_cast<simd_value>(indirect(b, N));

        where(mv, av) = bv;
        indirect(c, N) = av;

        for (unsigned i = 0; i<N; ++i) {
            exp[i] = m[i]? b[i] : a[i];
        }
        EXPECT_TRUE(testing::indexed_eq_n(N, c, exp));
    }

    // simd = where
    {
        scalar_value c[N];

        simd_value av = simd_cast<simd_value>(indirect(a, N));
        simd_value bv = simd_cast<simd_value>(indirect(b, N));

        simd_value cv = simd_cast<simd_value>(where(mv, add(av, bv)));
        indirect(c, N) = cv;

        for (unsigned i = 0; i<N; ++i) {
            exp[i] = m[i]? (a[i] + b[i]) : 0;
        }
        EXPECT_TRUE(testing::indexed_eq_n(N, c, exp));
    }

    // where = indirect
    {
        scalar_value c[N];

        simd_value av = simd_cast<simd_value>(indirect(a, N));

        where(mv, av) = indirect(b, N);
        indirect(c, N) = av;

        for (unsigned i = 0; i<N; ++i) {
            exp[i] = m[i]? b[i] : a[i];
        }
        EXPECT_TRUE(testing::indexed_eq_n(N, c, exp));
    }

    // indirect = where
    {
        scalar_value c[N];
        fill_random(c, rng);

        simd_value av = simd_cast<simd_value>(indirect(a, N));

        indirect(c, N)  = where(mv, av);

        for (unsigned i = 0; i<N; ++i) {
            exp[i] = m[i]? a[i] : c[i];
        }
        EXPECT_TRUE(testing::indexed_eq_n(N, c, exp));

        indirect(c, N)  = where(mv, neg(av));

        for (unsigned i = 0; i<N; ++i) {
            exp[i] = m[i]? -a[i] : c[i];
        }
        EXPECT_TRUE(testing::indexed_eq_n(N, c, exp));
    }

    // where = indirect indexed
    {
        scalar_value c[N];

        simd_value av = simd_cast<simd_value>(indirect(a, N));

        scalar_index idx[N];
        for (unsigned i =0; i<N; ++i) {
            idx[i] = i/2;
        }
        simd_index idxv = simd_cast<simd_index>(indirect(idx, N));

        where(mv, av) = indirect(b, idxv, N, index_constraint::none);
        indirect(c, N) = av;

        for (unsigned i = 0; i<N; ++i) {
            exp[i] = m[i]? b[idx[i]] : a[i];
        }
        EXPECT_TRUE(testing::indexed_eq_n(N, c, exp));
    }

    // indirect indexed = where
    {
        scalar_value c[N];
        fill_random(c, rng);

        simd_value av = simd_cast<simd_value>(indirect(a, N));
        simd_value bv = simd_cast<simd_value>(indirect(b, N));

        scalar_index idx[N];
        for (unsigned i =0; i<N; ++i) {
            idx[i] = i;
        }
        simd_index idxv = simd_cast<simd_index>(indirect(idx, N));

        indirect(c, idxv, N, index_constraint::contiguous)  = where(mv, av);

        for (unsigned i = 0; i<N; ++i) {
            exp[idx[i]] = m[i]? a[i] : c[idx[i]];
        }
        EXPECT_TRUE(testing::indexed_eq_n(N, c, exp));

        indirect(c, idxv, N, index_constraint::contiguous)  = where(mv, sub(av, bv));

        for (unsigned i = 0; i<N; ++i) {
            exp[idx[i]] = m[i]? a[i] - b[i] : c[idx[i]];
        }
        EXPECT_TRUE(testing::indexed_eq_n(N, c, exp));
    }
}

TYPED_TEST_P(sizeless_api, arithmetic) {
    using simd_value   = typename TypeParam::simd_value::simd_type;
    using scalar_value = typename TypeParam::simd_value::scalar_type;

    constexpr unsigned N = TypeParam::simd_value::width;

    std::minstd_rand rng(201);

    scalar_value a[N], b[N], c[N], expected[N];
    fill_random(a, rng);
    fill_random(b, rng);

    bool m[N], expected_m[N];
    fill_random(m, rng);

    simd_value av = simd_cast<simd_value>(indirect(a, N));
    simd_value bv = simd_cast<simd_value>(indirect(b, N));

    // add
    {
        indirect(c, N) = add(av, bv);
        for (unsigned i = 0; i<N; ++i) {
            expected[i] = a[i] + b[i];
        }
        EXPECT_TRUE(testing::indexed_almost_eq_n(N, c, expected));
    }
    // sub
    {
        indirect(c, N) = sub(av, bv);
        for (unsigned i = 0; i<N; ++i) {
            expected[i] = a[i] - b[i];
        }
        EXPECT_TRUE(testing::indexed_almost_eq_n(N, c, expected));
    }
    // mul
    {
        indirect(c, N) = mul(av, bv);
        for (unsigned i = 0; i<N; ++i) {
            expected[i] = a[i] * b[i];
        }
        EXPECT_TRUE(testing::indexed_almost_eq_n(N, c, expected));
    }
    // div
    {
        indirect(c, N) = div(av, bv);
        for (unsigned i = 0; i<N; ++i) {
            expected[i] = a[i] / b[i];
        }
        EXPECT_TRUE(testing::indexed_almost_eq_n(N, c, expected));
    }
    // pow
    {
        indirect(c, N) = pow(av, bv);
        for (unsigned i = 0; i<N; ++i) {
            expected[i] = std::pow(a[i], b[i]);
        }
        EXPECT_TRUE(testing::indexed_almost_eq_n(N, c, expected));
    }
    // min
    {
        indirect(c, N) = min(av, bv);
        for (unsigned i = 0; i<N; ++i) {
            expected[i] = std::min(a[i], b[i]);
        }
        EXPECT_TRUE(testing::indexed_almost_eq_n(N, c, expected));
    }
    // max
    {
        indirect(c, N) = max(av, bv);
        for (unsigned i = 0; i<N; ++i) {
            expected[i] = std::max(a[i], b[i]);
        }
        EXPECT_TRUE(testing::indexed_almost_eq_n(N, c, expected));
    }
    // cmp_eq
    {
        indirect(m, N) = cmp_eq(av, bv);
        for (unsigned i = 0; i<N; ++i) {
            expected_m[i] = a[i] == b[i];
        }
        EXPECT_TRUE(testing::indexed_eq_n(N, m, expected_m));
    }
    // cmp_neq
    {
        indirect(m, N) = cmp_neq(av, bv);
        for (unsigned i = 0; i<N; ++i) {
            expected_m[i] = a[i] != b[i];
        }
        EXPECT_TRUE(testing::indexed_eq_n(N, m, expected_m));
    }
    // cmp_leq
    {
        indirect(m, N) = cmp_leq(av, bv);
        for (unsigned i = 0; i<N; ++i) {
            expected_m[i] = a[i] <= b[i];
        }
        EXPECT_TRUE(testing::indexed_eq_n(N, m, expected_m));
    }
    // cmp_geq
    {
        indirect(m, N) = cmp_geq(av, bv);
        for (unsigned i = 0; i<N; ++i) {
            expected_m[i] = a[i] >= b[i];
        }
        EXPECT_TRUE(testing::indexed_eq_n(N, m, expected_m));
    }
    // sum
    {
        auto s = sum(av);
        scalar_value expected_sum = 0;

        for (unsigned i = 0; i<N; ++i) {
            expected_sum += a[i];
        }
        EXPECT_FLOAT_EQ(expected_sum, s);
    }
    // neg
    {
        indirect(c, N) = neg(av);
        for (unsigned i = 0; i<N; ++i) {
            expected[i] = -a[i];
        }
        EXPECT_TRUE(testing::indexed_almost_eq_n(N, c, expected));
    }
    // abs
    {
        indirect(c, N) = abs(av);
        for (unsigned i = 0; i<N; ++i) {
            expected[i] = std::abs(a[i]);
        }
        EXPECT_TRUE(testing::indexed_almost_eq_n(N, c, expected));
    }
    // exp
    {
        indirect(c, N) = exp(av);
        for (unsigned i = 0; i<N; ++i) {
            EXPECT_NEAR(std::exp(a[i]), c[i], 1e-6);
        }
    }
    // expm1
    {
        indirect(c, N) = expm1(av);
        for (unsigned i = 0; i<N; ++i) {
            EXPECT_NEAR(std::expm1(a[i]), c[i], 1e-6);
        }
    }
    // exprelr
    {
        indirect(c, N) = exprelr(av);
        for (unsigned i = 0; i<N; ++i) {
            EXPECT_NEAR(a[i]/(std::expm1(a[i])), c[i], 1e-6);
        }
    }
    // log
    {
        scalar_value l[N];
        int max_exponent = std::numeric_limits<scalar_value>::max_exponent;
        fill_random(l, rng, -max_exponent*std::log(2.), max_exponent*std::log(2.));
        for (auto& x: l) {
            x = std::exp(x);
            // SIMD log implementation may treat subnormal as zero
            if (std::fpclassify(x)==FP_SUBNORMAL) x = 0;
        }
        simd_value lv = simd_cast<simd_value>(indirect(l, N));

        indirect(c, N) = log(lv);

        for (unsigned i = 0; i<N; ++i) {
            expected[i] = std::log(l[i]);
        }
        EXPECT_TRUE(testing::indexed_almost_eq_n(N, c, expected));
    }

}

REGISTER_TYPED_TEST_CASE_P(sizeless_api, construct, where_exp, arithmetic);

typedef ::testing::Types<

#ifdef __AVX__
    simd_types_t< simd_t<     simd<double, 4, simd_abi::avx>, double, 4>,
                  simd_t<     simd<int,    4, simd_abi::avx>, int,    4>,
                  simd_t<simd_mask<double, 4, simd_abi::avx>, int,    4>>,
#endif
#ifdef __AVX2__
    simd_types_t< simd_t<     simd<double, 4, simd_abi::avx2>, double, 4>,
                  simd_t<     simd<int,    4, simd_abi::avx2>, int,    4>,
                  simd_t<simd_mask<double, 4, simd_abi::avx2>, int,    4>>,
#endif
#ifdef __AVX512F__
    simd_types_t< simd_t<     simd<double, 8, simd_abi::avx512>, double, 8>,
                  simd_t<     simd<int,    8, simd_abi::avx512>, int,    8>,
                  simd_t<simd_mask<double, 8, simd_abi::avx512>, int,    8>>,
#endif
#ifdef __ARM_NEON
    simd_types_t< simd_t<     simd<double, 2, simd_abi::neon>, double, 2>,
                  simd_t<     simd<int,    2, simd_abi::neon>, int,    2>,
                  simd_t<simd_mask<double, 2, simd_abi::neon>, double, 2>>,
#endif
#ifdef __ARM_FEATURE_SVE
    simd_types_t< simd_t<     simd<double, 0, simd_abi::sve>, double, 4>,
                  simd_t<     simd<int,    0, simd_abi::sve>, int,    4>,
                  simd_t<simd_mask<double, 0, simd_abi::sve>, bool,   4>>,

    simd_types_t< simd_t<     simd<double, 0, simd_abi::sve>, double, 8>,
                  simd_t<     simd<int,    0, simd_abi::sve>, int,    8>,
                  simd_t<simd_mask<double, 0, simd_abi::sve>, bool,   8>>,
#endif
    simd_types_t< simd_t<     simd<double, 8, simd_abi::default_abi>, double, 8>,
                  simd_t<     simd<int,    8, simd_abi::default_abi>, int,    8>,
                  simd_t<simd_mask<double, 8, simd_abi::default_abi>, bool,   8>>
> sizeless_api_test_types;

INSTANTIATE_TYPED_TEST_CASE_P(S, sizeless_api, sizeless_api_test_types);
