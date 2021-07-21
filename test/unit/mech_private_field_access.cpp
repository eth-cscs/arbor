#include <cstddef>

#include <arbor/version.hpp>
#include <arbor/mechanism.hpp>

#include "backends/multicore/fvm.hpp"
#ifdef ARB_GPU_ENABLED
#include "backends/gpu/fvm.hpp"
#include "memory/gpu_wrappers.hpp"
#endif

#include "common.hpp"
#include "mech_private_field_access.hpp"

using namespace arb;

namespace {
arb_value_type** field_lookup(const mechanism* m, const std::string& key) {
    for (arb_size_type i = 0; i<m->mech_.n_parameters; ++i) {
        if (key==m->mech_.parameters[i].name) return m->ppack_.parameters+i;
    }
    for (arb_size_type i = 0; i<m->mech_.n_state_vars; ++i) {
        if (key==m->mech_.state_vars[i].name) return m->ppack_.state_vars+i;
    }
    throw std::logic_error("internal error: no such field in mechanism");
}

arb_value_type* global_lookup(const mechanism* m, const std::string& key) {
    for (arb_size_type i = 0; i<m->mech_.n_globals; ++i) {
        if (key==m->mech_.globals[i].name) return m->ppack_.globals+i;
    }
    throw std::logic_error("internal error: no such field in mechanism");
}

arb_ion_state* ion_lookup(const mechanism* m, const std::string& ion) {
    for (arb_size_type i = 0; i<m->mech_.n_ions; ++i) {
        if (ion==m->mech_.ions[i].name) return m->ppack_.ion_states+i;
    }
    throw std::logic_error("internal error: no such field in mechanism");
}

// Multicore mechanisms:
std::vector<arb_value_type> mc_mechanism_field(const mechanism* m, const std::string& key) {
    auto p = *field_lookup(m, key);
    return std::vector<arb_value_type>(p, p+m->ppack_.width);
}

void mc_write_mechanism_field(const arb::mechanism* m, const std::string& key, const std::vector<arb::arb_value_type>& values) {
    auto p = *field_lookup(m, key);
    std::size_t n = std::min(values.size(), std::size_t(m->ppack_.width));
    std::copy_n(values.data(), n, p);
}

std::vector<arb_index_type> mc_mechanism_ion_index(const mechanism* m, const std::string& ion) {
    auto istate = *ion_lookup(m, ion);
    return std::vector<arb_index_type>(istate.index, istate.index+m->ppack_.width);
}

arb_value_type mc_mechanism_global(const mechanism* m, const std::string& key) {
    return *global_lookup(m, key);
}

// GPU mechanisms:
#ifdef ARB_GPU_ENABLED
std::vector<arb_value_type> gpu_mechanism_field(const mechanism* m, const std::string& key) {
    auto p_ptr = field_lookup(m, key);
    arb_value_type* p;
    memory::gpu_memcpy_d2h(&p, p_ptr, sizeof(p));

    std::size_t n = m->ppack_.width;
    std::vector<arb_value_type> values(n);
    memory::gpu_memcpy_d2h(values.data(), p, sizeof(arb_value_type)*n);
    return values;
}

void gpu_write_mechanism_field(const arb::mechanism* m, const std::string& key, const std::vector<arb::arb_value_type>& values) {
    auto p_ptr = field_lookup(m, key);
    arb_value_type* p;
    memory::gpu_memcpy_d2h(&p, p_ptr, sizeof(p));

    std::size_t n = std::min(values.size(), std::size_t(m->ppack_.width));
    memory::gpu_memcpy_h2d(const_cast<arb_value_type*>(values.data()), p, sizeof(arb_value_type)*n);
}

std::vector<arb_index_type> gpu_mechanism_ion_index(const mechanism* m, const std::string& ion) {
    auto istate_ptr = ion_lookup(m, ion);
    arb_ion_state istate;
    memory::gpu_memcpy_d2h(&istate, istate_ptr, sizeof(istate));
    std::vector<arb_index_type> vec(m->ppack_.width);
    memory::gpu_memcpy_d2h(vec.data(), istate.index, sizeof(arb_index_type)*m->ppack_.width);
    return vec;
}

arb_value_type gpu_mechanism_global(const mechanism* m, const std::string& key) {
    auto p = global_lookup(m, key);
    arb_value_type v;
    memory::gpu_memcpy_d2h(p, &v, sizeof(v));
    return v;
}
#endif
} // anonymous namespace

// Generic access:

std::vector<arb_value_type> mechanism_field(const mechanism* m, const std::string& key) {
    if (m->iface_.backend == arb_backend_kind_cpu) {
        return mc_mechanism_field(m, key);
    }

#ifdef ARB_GPU_ENABLED
    if (m->iface_.backend == arb_backend_kind_gpu) {
        return gpu_mechanism_field(m, key);
    }
#endif

    throw std::logic_error("internal error: mechanism instantiated on unknown backend");
}

void write_mechanism_field(const arb::mechanism* m, const std::string& key, const std::vector<arb::arb_value_type>& values) {
    if (m->iface_.backend == arb_backend_kind_cpu) {
        return mc_write_mechanism_field(m, key, values);
    }

#ifdef ARB_GPU_ENABLED
    if (m->iface_.backend == arb_backend_kind_gpu) {
        return gpu_write_mechanism_field(m, key, values);
    }
#endif

    throw std::logic_error("internal error: mechanism instantiated on unknown backend");
}

std::vector<arb_index_type> mechanism_ion_index(const mechanism* m, const std::string& ion) {
    if (m->iface_.backend == arb_backend_kind_cpu) {
        return mc_mechanism_ion_index(m, ion);
    }

#ifdef ARB_GPU_ENABLED
    if (m->iface_.backend == arb_backend_kind_gpu) {
        return gpu_mechanism_ion_index(m, ion);
    }
#endif

    throw std::logic_error("internal error: mechanism instantiated on unknown backend");
}

arb_value_type mechanism_global(const mechanism* m, const std::string& key) {
    if (m->iface_.backend == arb_backend_kind_cpu) {
        return mc_mechanism_global(m, key);
    }

#ifdef ARB_GPU_ENABLED
    if (m->iface_.backend == arb_backend_kind_gpu) {
        return gpu_mechanism_global(m, key);
    }
#endif

    throw std::logic_error("internal error: mechanism instantiated on unknown backend");
}
