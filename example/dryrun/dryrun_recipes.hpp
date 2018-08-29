#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>

#include <arbor/util/optional.hpp>
#include <arbor/recipe.hpp>
#include <arbor/symmetric_recipe.hpp>

#include "morphology_desc.hpp"

// miniapp-specific recipes

namespace arb {

struct probe_distribution {
    float proportion = 1.f; // what proportion of cells should get probes?
    bool all_segments = true;    // false => soma only
    bool membrane_voltage = true;
    bool membrane_current = true;
};

struct basic_recipe_param {
    // `num_compartments` is the number of compartments to place in each
    // unbranched section of the morphology, A value of zero indicates that
    // the number of compartments should equal the number of piecewise
    // linear segments in the morphology description of that branch.
    unsigned num_compartments = 1;

    // Total number of synapses on each cell.
    unsigned num_synapses = 1;

    std::string synapse_type = "expsyn";
    float min_connection_delay_ms = 20.0;
    float mean_connection_delay_ms = 20.75;
    float syn_weight_per_cell = 0.3;

    morphology morph = make_basic_y_morphology();
};

std::unique_ptr<recipe> make_basic_rgraph_symmetric_recipe(
        cell_gid_type ncell,
        cell_gid_type ntiles,
        basic_recipe_param param,
        probe_distribution pdist = probe_distribution{});

} // namespace arb
