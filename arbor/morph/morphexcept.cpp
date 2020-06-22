#include <string>
#include <sstream>

#include <arbor/morph/primitives.hpp>
#include <arbor/morph/morphexcept.hpp>

#include "util/strprintf.hpp"

namespace arb {

using arb::util::pprintf;

static std::string msize_string(msize_t x) {
    return x==mnpos? "mnpos": pprintf("{}", x);
}

invalid_mlocation::invalid_mlocation(mlocation loc):
    morphology_error(pprintf("invalid mlocation {}", loc)),
    loc(loc)
{}

no_such_branch::no_such_branch(msize_t bid):
    morphology_error(pprintf("no such branch id {}", msize_string(bid))),
    bid(bid)
{}

invalid_mcable::invalid_mcable(mcable cable):
    morphology_error(pprintf("invalid mcable {}", cable)),
    cable(cable)
{}

invalid_mcable_list::invalid_mcable_list():
    morphology_error("bad mcable_list")
{}

invalid_sample_parent::invalid_sample_parent(msize_t parent, msize_t tree_size):
    morphology_error(pprintf("invalid sample parent {} for a sample tree of size {}", msize_string(parent), tree_size)),
    parent(parent),
    tree_size(tree_size)
{
}

invalid_segment_parent::invalid_segment_parent(msize_t parent, msize_t tree_size):
    morphology_error(pprintf("invalid segment parent {} for a segment tree of size {}", msize_string(parent), tree_size)),
    parent(parent),
    tree_size(tree_size)
{
}

label_type_mismatch::label_type_mismatch(const std::string& label):
    morphology_error(pprintf("label \"{}\" is already bound to a different type of object", label)),
    label(label)
{}

incomplete_branch::incomplete_branch(msize_t bid):
    morphology_error(pprintf("insufficent samples to define branch id {}", msize_string(bid))),
    bid(bid)
{}

unbound_name::unbound_name(const std::string& name):
    morphology_error(pprintf("no definition for '{}'", name)),
    name(name)
{}

circular_definition::circular_definition(const std::string& name):
    morphology_error(pprintf("definition of '{}' requires a definition for '{}'", name, name)),
    name(name)
{}

} // namespace arb

