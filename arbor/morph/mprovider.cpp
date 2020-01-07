#include <string>
#include <utility>

#include <arbor/morph/label_dict.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/morphexcept.hpp>
#include <arbor/morph/mprovider.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/region.hpp>

namespace arb {

void mprovider::init() {
    // Evaluate each named region or locset in provided dictionary
    // to populate concrete regions_, locsets_ maps.

    if (!label_dict_ptr) return;

    for (const auto& pair: label_dict_ptr->regions()) {
        (void)region(pair.first);
    }

    for (const auto& pair: label_dict_ptr->locsets()) {
        (void)locset(pair.first);
    }

    label_dict_ptr = nullptr;
}

// Evaluation of a named region or locset requires the recursive evaluation of
// any component regions or locsets in its definition.
//
// During the initialization phase, 'named' expressions will be looked up in the
// provided label_dict, and the maps updated accordingly. Post-initialization,
// label_dict_ptr will be null, and concrete regions/locsets will only be retrieved
// from the maps established during initialization.

template <typename RegOrLocMap, typename LabelDictMap, typename Err>
static const auto& try_lookup(const mprovider& provider, const std::string& name, RegOrLocMap& map, const LabelDictMap* dict_ptr, Err errval) {
    auto it = map.find(name);
    if (it==map.end()) {
        if (dict_ptr) {
            map.emplace(name, errval);

            auto it = dict_ptr->find(name);
            if (it==dict_ptr->end()) {
                throw unbound_name(name);
            }

            return (map[name] = thingify(it->second, provider)).first();
        }
        else {
            throw unbound_name(name);
        }
    }
    else if (!it->second) {
        throw circular_definition(name);
    }
    else {
        return it->second.first();
    }
}

const mcable_list& mprovider::region(const std::string& name) const {
    const auto* regions_ptr = label_dict_ptr? &(label_dict_ptr->regions()): nullptr;
    return try_lookup(*this, name, regions_, regions_ptr, circular_def{});
}

const mlocation_list& mprovider::locset(const std::string& name) const {
    const auto* locsets_ptr = label_dict_ptr? &(label_dict_ptr->locsets()): nullptr;
    return try_lookup(*this, name, locsets_, locsets_ptr, circular_def{});
}


} // namespace arb
