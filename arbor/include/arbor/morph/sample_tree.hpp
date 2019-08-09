#pragma once

#include <cassert>
#include <functional>
#include <vector>
#include <string>

#include <arbor/swcio.hpp>
#include <arbor/morph/primitives.hpp>

namespace arb {

/// Morphology composed of samples.
class sample_tree {
    std::vector<msample> samples_;
    std::vector<size_t> parents_;

public:
    sample_tree() = default;
    sample_tree(std::vector<msample>, std::vector<size_t>);

    // Reserve space for n samples.
    void reserve(std::size_t n);

    // The append functions return a handle to the last sample appended by the call.

    // Append a single sample.
    size_t append(size_t p, const msample& s);

    // Append a sequence of samples.
    size_t append(size_t p, const std::vector<msample>& slist);

    // The number of samples in the tree.
    std::size_t size() const;

    // The samples in the tree.
    const std::vector<msample>& samples() const;

    // The parent index of the samples.
    const std::vector<size_t>& parents() const;

    friend std::ostream& operator<<(std::ostream&, const sample_tree&);
};

/// Build a sample tree from a sequence of swc records.
sample_tree swc_as_sample_tree(const std::vector<swc_record>& swc_records);

} // namesapce arb

