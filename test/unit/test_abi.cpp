#include <vector>
#include <string>

#include "../test/gtest.h"

#include <arbor/mechanism_abi.h>
#include <arbor/mechanism.hpp>
#include <arbor/version.hpp>

#include "backends/multicore/shared_state.hpp"
#ifdef ARB_GPU_ENABLED
#include "backends/gpu/shared_state.hpp"
#include "memory/gpu_wrappers.hpp"
#endif

using namespace std::string_literals;

TEST(abi, multicore_initialisation) {
    std::vector<arb_field_info> globals = {{ "G0", "kg",  123.0,     0.0, 2000.0},
                                           { "G1", "lb",  456.0,     0.0, 2000.0},
                                           { "G2", "gr",  789.0,     0.0, 2000.0}};
    std::vector<arb_field_info> states  = {{ "S0", "nA",      0.123, 0.0, 2000.0},
                                           { "S1", "mV",      0.456, 0.0, 2000.0}};
    std::vector<arb_field_info> params  = {{ "P0", "lm", -123.0,     0.0, 2000.0}};

    arb_mechanism_type type{};
    type.abi_version = ARB_MECH_ABI_VERSION;
    type.globals    = globals.data(); type.n_globals    = globals.size();
    type.parameters = params.data();  type.n_parameters = params.size();
    type.state_vars = states.data();  type.n_state_vars = states.size();

    arb_mechanism_interface iface { arb_backend_kind_cpu,
                                    1,
                                    1,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr };

    auto mech = arb::mechanism(type, iface);

    arb_size_type ncell = 1;
    arb_size_type ncv = 1;
    std::vector<arb_index_type> cv_to_intdom(ncv, 0);
    std::vector<arb_value_type> temp(ncv, 23);
    std::vector<arb_value_type> diam(ncv, 1.);
    std::vector<arb_value_type> vinit(ncv, -65);
    std::vector<arb::fvm_gap_junction> gj = {};
    std::vector<arb_index_type> src_to_spike = {};

    arb::multicore::shared_state shared_state(ncell, ncell, 0,
                                              cv_to_intdom, cv_to_intdom,
                                              gj, vinit, temp, diam, src_to_spike,
                                              mech.data_alignment());

    arb::mechanism_layout layout;
    layout.weight.assign(ncv, 1.);
    for (arb_size_type i = 0; i<ncv; ++i) layout.cv.push_back(i);

    shared_state.instantiate(mech, 42, {}, layout);

    {
        ASSERT_EQ(globals.size(), mech.mech_.n_globals);
        for (auto i = 0ul; i < globals.size(); ++i) {
            EXPECT_EQ(globals[i].default_value, mech.ppack_.globals[i]);
        }
    }

    {
        ASSERT_EQ(states.size(), mech.mech_.n_state_vars);
        for (auto i = 0ul; i < states.size(); ++i) {
            const auto* var_data = mech.ppack_.state_vars[i];

            std::vector<arb_value_type> expected(ncv, states[i].default_value);
            std::vector<arb_value_type> values(var_data, var_data+ncv);

            EXPECT_EQ(expected, values);
        }
    }

    {
        ASSERT_EQ(params.size(), mech.mech_.n_parameters);
        for (auto i = 0ul; i < params.size(); ++i) {
            const auto* param_data = mech.ppack_.parameters[i];

            std::vector<arb_value_type> expected(ncv, params[i].default_value);
            std::vector<arb_value_type> values(param_data, param_data+ncv);

            EXPECT_EQ(expected, values);
        }
    }
}

#ifdef ARB_GPU_ENABLED

namespace {
template <typename T>
T deref(const T* device_ptr) {
    T r;
    arb::memory::gpu_memcpy_d2h(&r, device_ptr, sizeof(T));
    return r;
}

template <typename T>
std::vector<T> vec_n(const T* device_ptr, std::size_t n) {
    std::vector<T> r(n);
    arb::memory::gpu_memcpy_d2h(r.data(), device_ptr, n*sizeof(T));
    return r;
}
}

TEST(abi, gpu_initialisation) {
    std::vector<arb_field_info> globals = {{ "G0", "kg",  123.0,     0.0, 2000.0},
                                           { "G1", "lb",  456.0,     0.0, 2000.0},
                                           { "G2", "gr",  789.0,     0.0, 2000.0}};
    std::vector<arb_field_info> states  = {{ "S0", "nA",      0.123, 0.0, 2000.0},
                                           { "S1", "mV",      0.456, 0.0, 2000.0}};
    std::vector<arb_field_info> params  = {{ "P0", "lm", -123.0,     0.0, 2000.0}};

    arb_mechanism_type type{};
    type.abi_version = ARB_MECH_ABI_VERSION;
    type.globals    = globals.data(); type.n_globals    = globals.size();
    type.parameters = params.data();  type.n_parameters = params.size();
    type.state_vars = states.data();  type.n_state_vars = states.size();

    arb_mechanism_interface iface { arb_backend_kind_gpu,
                                    1,
                                    1,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    nullptr };

    auto mech = arb::mechanism(type, iface);

    arb_size_type ncell = 1;
    arb_size_type ncv = 1;
    std::vector<arb_index_type> cv_to_intdom(ncv, 0);
    std::vector<arb_value_type> temp(ncv, 23);
    std::vector<arb_value_type> diam(ncv, 1.);
    std::vector<arb_value_type> vinit(ncv, -65);
    std::vector<arb::fvm_gap_junction> gj = {};
    std::vector<arb_index_type> src_to_spike = {};

    arb::gpu::shared_state shared_state(ncell, ncell, 0,
                                        cv_to_intdom, cv_to_intdom,
                                        gj, vinit, temp, diam, src_to_spike,
                                        1);

    arb::mechanism_layout layout;
    layout.weight.assign(ncv, 1.);
    for (arb_size_type i = 0; i<ncv; ++i) layout.cv.push_back(i);

    shared_state.instantiate(mech, 42, {}, layout);

    {
        ASSERT_EQ(globals.size(), mech.mech_.n_globals);
        for (auto i = 0ul; i < globals.size(); ++i) {
            EXPECT_EQ(globals[i].default_value, deref(mech.ppack_.globals+i));
        }
    }

    {
        ASSERT_EQ(states.size(), mech.mech_.n_state_vars);
        auto state_var_ptrs = vec_n(mech.ppack_.state_vars, states.size());

        for (auto i = 0ul; i < states.size(); ++i) {
            std::vector<arb_value_type> expected(ncv, states[i].default_value);
            std::vector<arb_value_type> values = vec_n(state_var_ptrs[i], ncv);

            EXPECT_EQ(expected, values);
        }
    }

    {
        ASSERT_EQ(params.size(), mech.mech_.n_parameters);
        auto param_ptrs = vec_n(mech.ppack_.parameters, params.size());
        for (auto i = 0ul; i < params.size(); ++i) {
            std::vector<arb_value_type> expected(ncv, params[i].default_value);
            std::vector<arb_value_type> values = vec_n(param_ptrs[i], ncv);

            EXPECT_EQ(expected, values);
        }
    }
}
#endif
