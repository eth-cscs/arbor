#pragma once

#include "matrix_multicore.hpp"

namespace nest {
namespace mc {
namespace multicore {

template <typename T, typename I>
struct fvm_policy {
    // define basic types
    using value_type = T;
    using size_type  = I;

    using base = memory_policy<value_type, size_type>;

    // define storage types
    using vector_type  = memory::HostVector<value_type>;
    using index_type   = memory::HostVector<size_type>;

    using base::view;
    using base::const_view;
    using base::iview;
    using base::const_iview;

    using base::host_vector_type;
    using base::host_index_type;

    /// define matrix type
    template <typename T, typename I>
    using matrix_policy = nest::mc::multicore::matrix_policy<T, I>;

    /// mechanism pointer type
    using mechanism_type =
        nest::mc::mechanisms::mechanism_ptr<value_type, size_type>;

    /// mechanism factory
    using mechanism_catalogue =
        nest::mc::mechanisms::catalogue<value_type, size_type>;

    /// helper function that converts containers into target specific view/rvalue
    template <typename U>
    auto on_target(U&& u) -> decltype(memory::on_host(u)) {
        return memory::on_host(u);
    }
};

} // namespace multicore
} // namespace mc
} // namespace nest

