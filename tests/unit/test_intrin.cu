#include "../gtest.h"

#include <limits>

#include <backends/gpu/intrinsics.hpp>
#include <backends/gpu/managed_ptr.hpp>
#include <memory/memory.hpp>
#include <util/rangeutil.hpp>
#include <util/span.hpp>

namespace kernels {
    template <typename T>
    __global__
    void test_atomic_add(T* x) {
        cuda_atomic_add(x, threadIdx.x+1);
    }

    template <typename T>
    __global__
    void test_atomic_sub(T* x) {
        cuda_atomic_sub(x, threadIdx.x+1);
    }

    __global__
    void test_min(double* x, double* y, double* result) {
        const auto i = threadIdx.x;
        result[i] = min(x[i], y[i]);
    }

    __global__
    void test_max(double* x, double* y, double* result) {
        const auto i = threadIdx.x;
        result[i] = max(x[i], y[i]);
    }

}

// test atomic addition wrapper for single and double precision
TEST(gpu_intrinsics, cuda_atomic_add) {
    int expected = (128*129)/2;

    auto f = arb::gpu::make_managed_ptr<float>(0.f);
    kernels::test_atomic_add<<<1, 128>>>(f.get());
    cudaDeviceSynchronize();

    EXPECT_EQ(float(expected), *f);

    auto d = arb::gpu::make_managed_ptr<double>(0.);
    kernels::test_atomic_add<<<1, 128>>>(d.get());
    cudaDeviceSynchronize();

    EXPECT_EQ(double(expected), *d);
}

// test atomic subtraction wrapper for single and double precision
TEST(gpu_intrinsics, cuda_atomic_sub) {
    int expected = -(128*129)/2;

    auto f = arb::gpu::make_managed_ptr<float>(0.f);
    kernels::test_atomic_sub<<<1, 128>>>(f.get());
    cudaDeviceSynchronize();

    EXPECT_EQ(float(expected), *f);

    auto d = arb::gpu::make_managed_ptr<double>(0.);
    kernels::test_atomic_sub<<<1, 128>>>(d.get());
    cudaDeviceSynchronize();

    EXPECT_EQ(double(expected), *d);
}

TEST(gpu_intrinsics, minmax) {
    const double inf = std::numeric_limits<double>::infinity();
    struct X {
        double lhs;
        double rhs;
        double expected_min;
        double expected_max;
    };

    std::vector<X> inputs = {
        {  0,    1,    0,   1},
        { -1,    1,   -1,   1},
        { 42,   42,   42,  42},
        {inf, -inf, -inf, inf},
        {  0,  inf,    0, inf},
        {  0, -inf, -inf,   0},
    };

    const auto n = arb::util::size(inputs);

    arb::memory::device_vector<double> lhs(n);
    arb::memory::device_vector<double> rhs(n);
    arb::memory::device_vector<double> result(n);

    using arb::util::make_span;

    for (auto i: make_span(0, n)) {
        lhs[i] = inputs[i].lhs;
        rhs[i] = inputs[i].rhs;
    }

    // test min
    kernels::test_min<<<1, n>>>(lhs.data(), rhs.data(), result.data());
    cudaDeviceSynchronize();
    for (auto i: make_span(0, n)) {
        EXPECT_EQ(double(result[i]), inputs[i].expected_min);
    }

    kernels::test_min<<<1, n>>>(rhs.data(), lhs.data(), result.data());
    cudaDeviceSynchronize();
    for (auto i: make_span(0, n)) {
        EXPECT_EQ(double(result[i]), inputs[i].expected_min);
    }

    // test max
    kernels::test_max<<<1, n>>>(lhs.data(), rhs.data(), result.data());
    cudaDeviceSynchronize();
    for (auto i: make_span(0, n)) {
        EXPECT_EQ(double(result[i]), inputs[i].expected_max);
    }

    kernels::test_max<<<1, n>>>(rhs.data(), lhs.data(), result.data());
    cudaDeviceSynchronize();
    for (auto i: make_span(0, n)) {
        EXPECT_EQ(double(result[i]), inputs[i].expected_max);
    }
}
