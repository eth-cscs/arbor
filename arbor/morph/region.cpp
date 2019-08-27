#include <set>
#include <string>
#include <vector>

#include <arbor/morph/error.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/region.hpp>

#include "morph/em_morphology.hpp"
#include "util/span.hpp"
#include "util/strprintf.hpp"
#include "util/range.hpp"
#include "util/rangeutil.hpp"

namespace arb {
namespace reg {

mcable_list merge(const mcable_list& lhs, const mcable_list& rhs) {
    mcable_list v;
    v.resize(lhs.size() + rhs.size());
    std::merge(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), v.begin());
    return v;
}

bool is_disjoint_union(const mcable& a, const mcable& b) {
    if (a.branch!=b.branch) return true;
    return a<b? a.dist_pos<b.prox_pos: b.dist_pos<a.prox_pos;
}

bool is_nonnull_intersection(const mcable& a, const mcable& b) {
    if (a==b) return true; // handles special case of spherical branch
    if (a.branch!=b.branch) return false;
    return a<b? a.dist_pos>b.prox_pos: b.dist_pos>a.prox_pos;
}

mcable make_union(const mcable& a, const mcable& b) {
    assert(!is_disjoint_union(a,b));
    return mcable{a.branch, std::min(a.prox_pos, b.prox_pos), std::max(a.dist_pos, b.dist_pos)};
}

mcable make_intersection(const mcable& a, const mcable& b) {
    assert(is_nonnull_intersection(a,b));

    return mcable{a.branch, std::max(a.prox_pos, b.prox_pos), std::min(a.dist_pos, b.dist_pos)};
}

//
// Explicit cable section
//

struct cable_ {
    mcable cable;
};

region cable(mcable c) {
    if (!test_invariants(c)) {
        throw morphology_error(util::pprintf("Invalid cable section {}", c));
    }
    return region(cable_{c});
}

region branch(msize_t bid) {
    return cable({bid, 0, 1});
}

mcable_list concretise_(const cable_& reg, const em_morphology& em) {
    auto& m = em.morph();

    if (reg.cable.branch>=m.num_branches()) {
        throw morphology_error(util::pprintf("Branch {} does not exist in morpology", reg.cable.branch));
    }

    return {reg.cable};
}

std::set<std::string> get_named_dependencies_(const cable_&) {
    return {};
}

region replace_named_dependencies_(const cable_& reg,
                                   const region_dictionary& reg_dict,
                                   const locset_dictionary& ps_dict)
{
    return region(reg);
}

std::ostream& operator<<(std::ostream& o, const cable_& c) {
    return o << c.cable;
}

//
//  Explicit list of cable sections (concretised region).
//

struct cable_list_ {
    mcable_list list;
};

region cable_list(mcable_list l) {
    std::sort(l.begin(), l.end());
    if (!test_invariants(l)) {
        throw morphology_error(util::pprintf("Invalid cable list {}", l));
    }
    return region(cable_list_{std::move(l)});
}

mcable_list concretise_(const cable_list_& reg, const em_morphology& em) {
    auto& m = em.morph();
    const auto &L = reg.list;

    for (auto& c: L) {
        if (c.branch>=m.num_branches()) {
            throw morphology_error(util::pprintf("Branch {} does not exist in morpology", c.branch));
        }
    }

    return L;
}

std::set<std::string> get_named_dependencies_(const cable_list_&) {
    return {};
}

region replace_named_dependencies_(const cable_list_& reg,
                                   const region_dictionary& reg_dict,
                                   const locset_dictionary& ps_dict)
{
    return region(reg);
}

std::ostream& operator<<(std::ostream& o, const cable_list_& c) {
    return o << c.list;
}

//
// region with all segments with the same numeric tag
//
struct tagged_ {
    int tag;
};

region tagged(int id) {
    return region(tagged_{id});
}

mcable_list concretise_(const tagged_& reg, const em_morphology& em) {
    auto& m = em.morph();
    size_t nb = m.num_branches();

    std::vector<mcable> L;
    L.reserve(nb);
    auto& samples = m.samples();
    auto matches     = [reg, &samples](msize_t i) {return samples[i].tag==reg.tag;};
    auto not_matches = [&matches](msize_t i) {return !matches(i);};

    for (msize_t i: util::make_span(nb)) {
        auto ids = util::make_range(m.branch_indexes(i)); // range of sample ids in branch.
        size_t ns = util::size(ids);        // number of samples in branch.

        if (ns==1) {
            // The branch is a spherical soma
            if (samples[0].tag==reg.tag) {
                L.push_back({0,0,1});
            }
            continue;
        }

        // The branch has at least 2 samples.
        // Start at begin+1 because a segment gets its tag from its distal sample.
        auto beg = std::cbegin(ids);
        auto end = std::cend(ids);

        // Find the next sample that matches reg.tag.
        auto start = std::find_if(beg+1, end, matches);
        while (start!=end) {
            // find the next sample that does not match reg.tag
            auto first = start-1;
            auto last = std::find_if(start, end, not_matches);

            auto l = first==beg? 0.: em.sample2loc(*first).pos;
            auto r = last==end?  1.: em.sample2loc(*(last-1)).pos;
            L.push_back({i, l, r});

            // Find the next sample in the branch that matches reg.tag.
            start = std::find_if(last, end, matches);
        }
    }
    return L;
}

std::set<std::string> get_named_dependencies_(const tagged_&) {
    return {};
}

region replace_named_dependencies_(const tagged_& reg,
                                     const region_dictionary& reg_dict,
                                     const locset_dictionary& ps_dict)
{
    return region(reg);
}

std::ostream& operator<<(std::ostream& o, const tagged_& t) {
    return o << "(tag " << t.tag << ")";
}

//
// region with all segments in a cell
//
struct all_ {};

region all() {
    return region(all_{});
}

mcable_list concretise_(const all_&, const em_morphology& m) {
    auto nb = m.morph().num_branches();
    mcable_list branches;
    branches.reserve(nb);
    for (auto i: util::make_span(nb)) {
        branches.push_back({i,0,1});
    }
    return branches;
}

std::set<std::string> get_named_dependencies_(const all_&) {
    return {};
}

region replace_named_dependencies_(const all_& reg, const region_dictionary& reg_dict, const locset_dictionary& ps_dict) {
    return region(reg);
}

std::ostream& operator<<(std::ostream& o, const all_& t) {
    return o << "all";
}

//
// a named region
//
struct named_ {
    named_(std::string n): name(std::move(n)) {}
    std::string name;
};

region named(std::string n) {
    return region(named_{std::move(n)});
}

mcable_list concretise_(const named_&, const em_morphology& m) {
    throw morphology_error("not possible to generated cable segments from a named region");
    return {};
}

std::set<std::string> get_named_dependencies_(const named_& n) {
    return {n.name};
}

region replace_named_dependencies_(const named_& reg, const region_dictionary& reg_dict, const locset_dictionary& ps_dict) {
    auto it = reg_dict.find(reg.name);
    if (it == reg_dict.end()) {
        throw morphology_error(
            util::pprintf("internal error: unable to replace label {}, unavailable in label dictionary", reg.name));
    }
    return it->second;
}

std::ostream& operator<<(std::ostream& o, const named_& n) {
    return o << "\"" << n.name << "\"";
}

//
// intersection of two point sets
//
struct reg_and {
    region lhs;
    region rhs;
    reg_and(region lhs, region rhs): lhs(std::move(lhs)), rhs(std::move(rhs)) {}
};

mcable_list concretise_(const reg_and& P, const em_morphology& m) {
    using cable_it = std::vector<mcable>::const_iterator;
    using cable_it_pair = std::pair<cable_it, cable_it>;

    auto lhs = concretise(P.lhs, m);
    auto rhs = concretise(P.rhs, m);
    cable_it_pair it{lhs.begin(), rhs.begin()};
    cable_it_pair end{lhs.end(), rhs.end()};
    std::vector<mcable> L;

    bool at_end = it.first==end.first || it.second==end.second;
    while (!at_end) {
        bool first_less = *(it.first) < *(it.second);
        auto& lhs = first_less? it.first: it.second;
        auto& rhs = first_less? it.second: it.first;
        if (is_nonnull_intersection(*lhs, *rhs)) {
            L.push_back(make_intersection(*lhs, *rhs));
        }
        if (dist_loc(*lhs) < dist_loc(*rhs)) {
            ++lhs;
        }
        else {
            ++rhs;
        }
        at_end = it.first==end.first || it.second==end.second;
    }

    return L;
}

std::set<std::string> get_named_dependencies_(const reg_and& reg) {
    auto l = named_dependencies(reg.lhs);
    auto r = named_dependencies(reg.rhs);
    l.insert(r.begin(), r.end());
    return l;
}

region replace_named_dependencies_(const reg_and& reg, const region_dictionary& reg_dict, const locset_dictionary& ps_dict) {
    return region(reg_and(replace_named_dependencies(reg.lhs, reg_dict, ps_dict),
                          replace_named_dependencies(reg.rhs, reg_dict, ps_dict)));
}

std::ostream& operator<<(std::ostream& o, const reg_and& x) {
    return o << "(and " << x.lhs << " " << x.rhs << ")";
}

//
// union of two point sets
//
struct reg_or {
    region lhs;
    region rhs;
    reg_or(region lhs, region rhs): lhs(std::move(lhs)), rhs(std::move(rhs)) {}
};

mcable_list concretise_(const reg_or& P, const em_morphology& m) {
    auto l = merge(concretise(P.lhs, m), concretise(P.rhs, m));
    if (l.size()<2) return l;
    std::vector<mcable> L;
    L.reserve(l.size());
    auto c = l.front();
    auto it = l.begin()+1;
    while (it!=l.end()) {
        if (!is_disjoint_union(c, *it)) {
            c = make_union(c, *it);
        }
        else {
            L.push_back(c);
            c = *it;
        }
        ++it;
    }
    L.push_back(c);
    return L;
}

std::set<std::string> get_named_dependencies_(const reg_or& reg) {
    auto l = named_dependencies(reg.lhs);
    auto r = named_dependencies(reg.rhs);
    l.insert(r.begin(), r.end());
    return l;
}

region replace_named_dependencies_(const reg_or& reg, const region_dictionary& reg_dict, const locset_dictionary& ps_dict) {
    return region(reg_or(replace_named_dependencies(reg.lhs, reg_dict, ps_dict),
                         replace_named_dependencies(reg.rhs, reg_dict, ps_dict)));
}

std::ostream& operator<<(std::ostream& o, const reg_or& x) {
    return o << "(or " << x.lhs << " " << x.rhs << ")";
}

} // namespace reg

// The and_ and or_ operations in the arb:: namespace with region so that
// ADL allows for construction of expressions with regions without having
// to namespace qualify the and_/or_.

region and_(region l, region r) {
    return region{reg::reg_and(std::move(l), std::move(r))};
}

region or_(region l, region r) {
    return region{reg::reg_or(std::move(l), std::move(r))};
}

} // namespace arb
