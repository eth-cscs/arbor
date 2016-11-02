#pragma once

#include "catalogue_multicore.hpp"
#include "matrix_multicore.hpp"

namespace nest {
namespace mc {
namespace multicore {

struct fvm_policy : public memory_traits {
    /// define matrix solver
    using matrix_solver = nest::mc::multicore::matrix_solver;

    /// define matrix builder
    using matrix_builder = nest::mc::multicore::fvm_matrix_builder;

    /// mechanism factory
    using mechanism_catalogue = nest::mc::multicore::catalogue;

    /// back end specific storage for mechanisms
    using mechanism_type = mechanism_catalogue::mechanism_ptr_type;

    /// back end specific storage for shared ion specie state
    using ion_type = mechanism_catalogue::ion_type;

    static std::string name() {
        return "multicore";
    }
};

} // namespace multicore
} // namespace mc
} // namespace nest

