#include <string>
#include <sstream>

#include <arbor/arbexcept.hpp>
#include <arbor/common_types.hpp>

#include "util/strprintf.hpp"

namespace arb {

using arb::util::pprintf;

bad_cell_probe::bad_cell_probe(cell_kind kind, cell_gid_type gid):
    arbor_exception(pprintf("recipe::get_grobe() is not supported for cell with gid {} of kind {})", gid, kind)),
    gid(gid),
    kind(kind)
{}

bad_cell_description::bad_cell_description(cell_kind kind, cell_gid_type gid):
    arbor_exception(pprintf("recipe::get_cell_kind(gid={}) -> {} does not match the cell type provided by recipe::get_cell_description(gid={})", gid, kind, gid)),
    gid(gid),
    kind(kind)
{}

bad_connection_source_gid::bad_connection_source_gid(cell_gid_type gid, cell_gid_type src_gid, cell_size_type num_cells):
    arbor_exception(pprintf("Model building error on cell {}: connection source gid {} is out of range: there are only {} cells in the model, in the range [{}:{}].", gid, src_gid, num_cells, 0, num_cells-1)),
    gid(gid), src_gid(src_gid), num_cells(num_cells)
{}

bad_connection_label::bad_connection_label(cell_gid_type gid, const cell_tag_type& label, const std::string& msg):
    arbor_exception(pprintf("Model building error on cell {}: connection endpoint label \"{}\": {}.", gid, label, msg)),
    gid(gid), label(label)
{}

bad_global_property::bad_global_property(cell_kind kind):
    arbor_exception(pprintf("bad global property for cell kind {}", kind)),
    kind(kind)
{}

bad_probe_id::bad_probe_id(cell_member_type probe_id):
    arbor_exception(pprintf("bad probe id {}", probe_id)),
    probe_id(probe_id)
{}

gj_unsupported_domain_decomposition::gj_unsupported_domain_decomposition(cell_gid_type gid_0, cell_gid_type gid_1):
    arbor_exception(pprintf("No support for gap junctions across domain decomposition groups for gid {} and {}", gid_0, gid_1)),
    gid_0(gid_0),
    gid_1(gid_1)
{}

gj_unsupported_lid_selection_policy::gj_unsupported_lid_selection_policy(cell_gid_type gid, cell_tag_type label):
    arbor_exception(pprintf("Model building error on cell {}: gap junction site label \"{}\" must be univalent.", gid, label)),
    gid(gid),
    label(label)
{}

gj_kind_mismatch::gj_kind_mismatch(cell_gid_type gid_0, cell_gid_type gid_1):
    arbor_exception(pprintf("Cells on gid {} and {} connected via gap junction have different cell kinds", gid_0, gid_1)),
    gid_0(gid_0),
    gid_1(gid_1)
{}

bad_event_time::bad_event_time(time_type event_time, time_type sim_time):
    arbor_exception(pprintf("event time {} precedes current simulation time {}", event_time, sim_time)),
    event_time(event_time),
    sim_time(sim_time)
{}

no_such_mechanism::no_such_mechanism(const std::string& mech_name):
    arbor_exception(pprintf("no mechanism {} in catalogue", mech_name)),
    mech_name(mech_name)
{}

duplicate_mechanism::duplicate_mechanism(const std::string& mech_name):
    arbor_exception(pprintf("mechanism {} already exists", mech_name)),
    mech_name(mech_name)
{}

fingerprint_mismatch::fingerprint_mismatch(const std::string& mech_name):
    arbor_exception(pprintf("mechanism {} has different fingerprint in schema", mech_name)),
    mech_name(mech_name)
{}

no_such_parameter::no_such_parameter(const std::string& mech_name, const std::string& param_name):
    arbor_exception(pprintf("mechanism {} has no parameter {}", mech_name, param_name)),
    mech_name(mech_name),
    param_name(param_name)
{}

invalid_parameter_value::invalid_parameter_value(const std::string& mech_name, const std::string& param_name, double value):
    arbor_exception(pprintf("invalid parameter value for mechanism {} parameter {}: {}", mech_name, param_name, value)),
    mech_name(mech_name),
    param_name(param_name),
    value_str(),
    value(value)
{}

invalid_parameter_value::invalid_parameter_value(const std::string& mech_name, const std::string& param_name, const std::string& value_str):
    arbor_exception(pprintf("invalid parameter value for mechanism {} parameter {}: {}", mech_name, param_name, value_str)),
    mech_name(mech_name),
    param_name(param_name),
    value_str(value_str),
    value(0)
{}

invalid_ion_remap::invalid_ion_remap(const std::string& mech_name):
    arbor_exception(pprintf("invalid ion parameter remapping for mechanism {}", mech_name))
{}

invalid_ion_remap::invalid_ion_remap(const std::string& mech_name, const std::string& from_ion = "", const std::string& to_ion = ""):
    arbor_exception(pprintf("invalid ion parameter remapping for mechanism {}: {} -> {}", mech_name, from_ion, to_ion)),
    from_ion(from_ion),
    to_ion(to_ion)
{}

no_such_implementation::no_such_implementation(const std::string& mech_name):
    arbor_exception(pprintf("missing implementation for mechanism {} in catalogue", mech_name)),
    mech_name(mech_name)
{}

range_check_failure::range_check_failure(const std::string& whatstr, double value):
    arbor_exception(pprintf("range check failure: {} with value {}", whatstr, value)),
    value(value)
{}

file_not_found_error::file_not_found_error(const std::string &fn)
    : arbor_exception(pprintf("Could not find file '{}'", fn)),
      filename{fn}
{}

bad_catalogue_error::bad_catalogue_error(const std::string &fn, const std::string& call)
    : arbor_exception(pprintf("Error in '{}' while opening catalogue '{}'", call, fn)),
      filename{fn},
      failed_call{call}
{}

unsupported_abi_error::unsupported_abi_error(size_t v):
    arbor_exception(pprintf("ABI version is not supported by this version of arbor '{}'", v)),
    version{v} {}

bad_alignment::bad_alignment(size_t a):
    arbor_exception(pprintf("Mechanism reported unsupported alignment '{}'", a)),
    alignment{a} {}

} // namespace arb

