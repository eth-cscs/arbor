#include <fstream>
#include <iterator>

#include <json/json.hpp>

#include <common_types.hpp>
#include <cell.hpp>
#include <cell_group.hpp>
#include <fvm_multicell.hpp>
#include <util/rangeutil.hpp>

#include "gtest.h"
#include "../test_util.hpp"
#include "validation_data.hpp"

// For storing the results of a simulation along with the information required
// to compare two simulations for accuracy.
struct result {
    std::vector<std::vector<double>> spikes;
    std::vector<std::vector<double>> baseline_spikes;
    std::vector<testing::spike_comparison> comparisons;
    std::vector<double> thresh;
    int n_comparments;

    result(int nseg, double dt,
      std::vector<std::vector<double>> &v,
      nlohmann::json& measurements
    ) {
        n_comparments = nseg;
        baseline_spikes = {
            measurements["soma"]["spikes"],
            measurements["dend"]["spikes"]
        };
        thresh = {
            measurements["soma"]["thresh"],
            measurements["dend"]["thresh"]
        };
        for(auto i=0u; i<v.size(); ++i) {
            // calculate the NEST MC spike times
            spikes.push_back
                (testing::find_spikes(v[i], thresh[i], dt));
            // compare NEST MC and baseline NEURON spike times
            comparisons.push_back
                (testing::compare_spikes(spikes[i], baseline_spikes[i]));
        }
    }
};

// compares results with those generated by nrn/simple_synapse.py
void run_neuron_baseline(const char* syn_type, const char* data_file)
{
    using namespace nest::mc;
    using namespace nlohmann;
    using lowered_cell = fvm::fvm_multicell<multicore::fvm_policy>;

    nest::mc::cell cell;

    // Soma with diameter 12.6157 um and HH channel
    auto soma = cell.add_soma(12.6157/2.0);
    soma->add_mechanism(hh_parameters());

    // add dendrite of length 200 um and diameter 1 um with passive channel
    auto dendrite = cell.add_cable(0, segmentKind::dendrite, 0.5, 0.5, 200);
    dendrite->add_mechanism(pas_parameters());

    dendrite->mechanism("membrane").set("r_L", 100);
    soma->mechanism("membrane").set("r_L", 100);

    // add synapse
    parameter_list syn_default(syn_type);
    cell.add_synapse({1, 0.5}, syn_default);

    // add probes
    auto probe_soma_idx = cell.add_probe({{0,0}, probeKind::membrane_voltage});
    auto probe_dend_idx = cell.add_probe({{1,0.5}, probeKind::membrane_voltage});

    cell_member_type probe_soma{0u, probe_soma_idx};
    cell_member_type probe_dend{0u, probe_dend_idx};

    // injected spike events
    postsynaptic_spike_event<float> synthetic_events[] = {
        {{0u, 0u}, 10.0, 0.04},
        {{0u, 0u}, 20.0, 0.04},
        {{0u, 0u}, 40.0, 0.04}
    };

    // load data from file
    auto cell_data = testing::g_validation_data.load(data_file);
    EXPECT_TRUE(cell_data.size()>0);
    if(cell_data.size()==0) return;

    json& nrn =
        *std::max_element(
            cell_data.begin(), cell_data.end(),
            [](json& lhs, json& rhs) {return lhs["nseg"]<rhs["nseg"];}
        );

    auto& measurements = nrn["measurements"];

    double dt = nrn["dt"];
    double tfinal = 50.; // ms

    std::vector<result> results;
    for(auto run_index=0u; run_index<cell_data.size(); ++run_index) {
        auto& run = cell_data[run_index];
        int num_compartments = run["nseg"];
        dendrite->set_compartments(num_compartments);
        std::vector<std::vector<double>> v(2);

        // make the lowered finite volume cell
        cell_group<lowered_cell> group(0, util::singleton_view(cell));

        // add the 3 spike events to the queue
        group.enqueue_events(synthetic_events);

        // run the simulation
        v[0].push_back(group.probe(probe_soma));
        v[1].push_back(group.probe(probe_dend));
        double t  = 0.;
        while(t < tfinal) {
            t += dt;
            group.advance(t, dt);
            // save voltage at soma and dendrite
            v[0].push_back(group.probe(probe_soma));
            v[1].push_back(group.probe(probe_dend));
        }

        results.push_back({num_compartments, dt, v, measurements});
    }

    // print results
    auto colors = {memory::util::kWhite, memory::util::kGreen, memory::util::kYellow};
    for(auto& r : results){
        auto color = colors.begin();
        for(auto const& result : r.comparisons) {
            std::cout << std::setw(5) << r.n_comparments << " compartments : ";
            std::cout << memory::util::colorize(util::pprintf("%\n", result), *(color++));
        }
    }

    // sort results in ascending order of compartments
    std::sort(
        results.begin(), results.end(),
        [](const result& l, const result& r)
            {return l.n_comparments<r.n_comparments;}
    );

    // the strategy for testing is the following:
    //  1. test that the solution converges to the finest reference solution as
    //     the number of compartments increases (i.e. spatial resolution is
    //     refined)
    for(auto j=0; j<2; ++j) {
        EXPECT_TRUE(
              results.back().comparisons[j].max_relative_error()
            < results.front().comparisons[j].max_relative_error()
        );
    }

    //  2. test that the best solution (i.e. with most compartments) matches the
    //     reference solution closely (less than 0.5% over the course of 100ms
    //     simulation)
    auto tol = 0.5;
    for(auto j=0; j<2; ++j) {
        EXPECT_TRUE(results.back().comparisons[j].max_relative_error()*100<tol);
    }
}

TEST(simple_synapse, expsyn_neuron_baseline) {
    SCOPED_TRACE("expsyn");
    run_neuron_baseline("expsyn","simple_exp_synapse.json");
}

TEST(simple_synapse, exp2syn_neuron_baseline) {
    SCOPED_TRACE("exp2syn");
    run_neuron_baseline("exp2syn","simple_exp2_synapse.json");
}
