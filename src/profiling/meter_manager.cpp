#include <communication/global_policy.hpp>
#include <util/hostname.hpp>

#include "meter_manager.hpp"

#include "memory_meter.hpp"
#include "power_meter.hpp"
#include "time_meter.hpp"

#include <json/json.hpp>

namespace nest {
namespace mc {
namespace util {

meter_manager::meter_manager() {
    // add time-measurement meter
    meters_.emplace_back(new time_meter());

    // add memory consumption meter
    if (has_memory_metering) {
        meters_.emplace_back(new memory_meter());
    }

    if (has_power_measurement) {
        meters_.emplace_back(new power_meter());
    }
};

void meter_manager::checkpoint(std::string name) {
    // Enforce a global synchronization point the first time that the meters
    // are used, to ensure that times measured across all domains are
    // synchronised.
    if (checkpoint_names_.size()==0) {
        communication::global_policy::barrier();
    }

    checkpoint_names_.push_back(std::move(name));
    for (auto& m: meters_) {
        m->take_reading();
    }
}

const std::vector<std::unique_ptr<meter>>& meter_manager::meters() const {
    return meters_;
}

const std::vector<std::string>& meter_manager::checkpoint_names() const {
    return checkpoint_names_;
}

nlohmann::json to_json(const meter_manager& manager) {
    using gcom = communication::global_policy;

    // Gather the meter outputs into a json Array
    nlohmann::json meter_out;
    for (auto& m: manager.meters()) {
        for (auto& measure: m->measurements()) {
            meter_out.push_back(to_json(measure));
        }
    }

    // Gather a vector with the names of the node that each rank
    // is running on.
    auto hosts = gcom::gather(hostname(), 0);

    // Only the "root" process returns meter information
    if (gcom::id()==0) {
        return {
            {"checkpoints", manager.checkpoint_names()},
            {"num_domains", gcom::size()},
            {"global_model", std::to_string(gcom::kind())},
            {"meters", meter_out},
            {"hosts", hosts},
        };
    }

    return {};
}

void save_to_file(const meter_manager& manager, const std::string& name) {
    auto measurements = to_json(manager);
    if (!communication::global_policy::id()) {
        std::ofstream fid;
        fid.exceptions(std::ios_base::badbit | std::ios_base::failbit);
        fid.open(name);
        fid /*<< std::setw(1)*/ << measurements << "\n";
    }
}

} // namespace util
} // namespace mc
} // namespace nest
