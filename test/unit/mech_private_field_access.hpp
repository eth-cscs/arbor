#pragma once

#include <memory>
#include <string>
#include <vector>

#include <arbor/arb_types.hpp>
#include <arbor/mechanism.hpp>

// Get a copy of the data within a mechanisms's named field.

std::vector<arb::arb_value_type> mechanism_field(const arb::mechanism* m, const std::string& key);
void write_mechanism_field(const arb::mechanism* m, const std::string& key, const std::vector<arb::arb_value_type>& values);
std::vector<arb_index_type> mechanism_ion_index(const arb::mechanism* m, const std::string& ion);
arb::arb_value_type mechanism_global(const arb::mechanism* m, const std::string& key);

inline std::vector<arb::arb_value_type> mechanism_field(const std::unique_ptr<arb::mechanism>& m, const std::string& key) {
    return mechanism_field(m.get(), key);
}
inline void write_mechanism_field(const std::unique_ptr<arb::mechanism>& m, const std::string& key, const std::vector<arb::arb_value_type>& values) {
    write_mechanism_field(m.get(), key, values);
}
inline std::vector<arb::arb_index_type> mechanism_ion_index(const std::unique_ptr<arb::mechanism>& m, const std::string& ion) {
    return mechanism_ion_index(m.get(), ion);
}
inline arb::arb_value_type mechanism_global(const std::unique_ptr<arb::mechanism>& m, const std::string& key) {
    return mechanism_global(m.get(), key);
}
