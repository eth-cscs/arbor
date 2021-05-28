#pragma once

#include <any>
#include <string>

#include <arbor/cv_policy.hpp>
#include <arbor/arbexcept.hpp>
#include <arbor/util/expected.hpp>

namespace arborio {

struct cv_policy_parse_error: arb::arbor_exception {
    cv_policy_parse_error(const std::string& msg): arb::arbor_exception(msg) {}
};

using parse_cv_policy_hopefully = arb::util::expected<arb::cv_policy, cv_policy_parse_error>;

parse_cv_policy_hopefully parse_cv_policy_expression(const std::string& s);

namespace literals {

inline
arb::cv_policy operator "" _cvp(const char* s, std::size_t) {
    if (auto r = parse_cv_policy_expression(s)) return *r;
    else throw r.error();
}

} // namespace literals

} // namespace arb
