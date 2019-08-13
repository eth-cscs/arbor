/*
 * Unit tests for sample_tree and morphology.
 */

#include <fstream>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "../test/gtest.h"

#include <arbor/math.hpp>
#include <arbor/morph/error.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/sample_tree.hpp>
#include <arbor/cable_cell.hpp>

#include "algorithms.hpp"
#include "io/sepval.hpp"
#include "util/range.hpp"
#include "util/span.hpp"
#include "util/strprintf.hpp"

// Forward declare non-public functions that are used internally to build
// morphologies so that they can be tested.
/*
namespace arb { namespace impl {
    std::vector<point_prop> mark_branch_props(const std::vector<size_t>& parents);
    std::vector<mbranch> branches_from_parent_index(const std::vector<size_t>& parents, bool spherical_root);
}}
*/

template <typename T>
std::ostream& operator<<(std::ostream& o, const std::vector<T>& v) {
    return o << "[" << arb::io::csv(v) << "]";
}

TEST(morphology, point_props) {
    arb::point_prop p = arb::point_prop_mask_none;

    EXPECT_FALSE(arb::is_terminal(p));
    EXPECT_FALSE(arb::is_fork(p));
    EXPECT_FALSE(arb::is_root(p));
    EXPECT_FALSE(arb::is_collocated(p));

    arb::set_root(p);
    EXPECT_FALSE(arb::is_terminal(p));
    EXPECT_FALSE(arb::is_fork(p));
    EXPECT_TRUE(arb::is_root(p));
    EXPECT_FALSE(arb::is_collocated(p));

    arb::set_terminal(p);
    EXPECT_TRUE(arb::is_terminal(p));
    EXPECT_FALSE(arb::is_fork(p));
    EXPECT_TRUE(arb::is_root(p));
    EXPECT_FALSE(arb::is_collocated(p));

    arb::unset_root(p);
    EXPECT_TRUE(arb::is_terminal(p));
    EXPECT_FALSE(arb::is_fork(p));
    EXPECT_FALSE(arb::is_root(p));
    EXPECT_FALSE(arb::is_collocated(p));

    arb::set_collocated(p);
    EXPECT_TRUE(arb::is_terminal(p));
    EXPECT_FALSE(arb::is_fork(p));
    EXPECT_FALSE(arb::is_root(p));
    EXPECT_TRUE(arb::is_collocated(p));

    arb::set_fork(p);
    EXPECT_TRUE(arb::is_terminal(p));
    EXPECT_TRUE(arb::is_fork(p));
    EXPECT_FALSE(arb::is_root(p));
    EXPECT_TRUE(arb::is_collocated(p));

    arb::unset_fork(p);
    arb::unset_terminal(p);
    arb::unset_collocated(p);
    EXPECT_FALSE(arb::is_terminal(p));
    EXPECT_FALSE(arb::is_fork(p));
    EXPECT_FALSE(arb::is_root(p));
    EXPECT_FALSE(arb::is_collocated(p));
}

// TODO: test sample_tree marks properties correctly

// Test internal function that parses a parent list, and marks
// each node as either root, sequential, fork or terminal.
/*
TEST(morphology, mark_branch_props) {
    using arb::io::csv;
    using arb::point_prop;

    point_prop r = arb::point_prop_mask_root;
    point_prop t = arb::point_prop_mask_terminal;
    point_prop s = arb::point_prop_mask_none;
    point_prop f = arb::point_prop_mask_fork;

    EXPECT_EQ((std::vector<point_prop> {r}),
               arb::impl::mark_branch_props({0}));

    EXPECT_EQ((std::vector<point_prop> {r,t}),
               arb::impl::mark_branch_props({0,0}));

    EXPECT_EQ((std::vector<point_prop> {r,s,s,t}),
               arb::impl::mark_branch_props({0,0,1,2}));

    EXPECT_EQ((std::vector<point_prop> {r,s,s,t,s,s,t}),
               arb::impl::mark_branch_props({0,0,1,2,0,4,5}));

    EXPECT_EQ((std::vector<point_prop> {r,s,f,s,f,t,t,s,t}),
               arb::impl::mark_branch_props({0,0,1,2,3,2,4,4,7}));
}

TEST(morphology, branches_from_parent_index) {
    const auto npos = arb::mbranch::npos;
    using mb = arb::mbranch;

    {   // single sample: can only be used to build a morphology with one spherical branch
        std::vector<size_t> parents = {0};
        auto bc = arb::impl::branches_from_parent_index(parents, true);
        EXPECT_EQ(1u, bc.size());
        EXPECT_EQ(mb(0, 1, 1, npos), bc[0]);

        // A cable morphology can't be constructed from a single sample.
        EXPECT_THROW(arb::impl::branches_from_parent_index(parents, false),
                     arb::morphology_error);
    }
    {
        std::vector<size_t> parents = {0, 0};
        auto bc = arb::impl::branches_from_parent_index(parents, false);
        EXPECT_EQ(1u, bc.size());
        EXPECT_EQ(mb(0, 1, 2, npos), bc[0]);

        // A morphology can't be constructed with a spherical soma from two samples.
        EXPECT_THROW(arb::impl::branches_from_parent_index(parents, true),
                     arb::morphology_error);
    }

    {
        std::vector<size_t> parents = {0, 0, 1};

        // With cable soma: one cable with 3 samples.
        auto bc = arb::impl::branches_from_parent_index(parents, false);
        EXPECT_EQ(1u, bc.size());
        EXPECT_EQ(mb(0,1,3,npos), bc[0]);

        // With spherical soma: one sphere and a 2-segment cable.
        // The cable branch is attached to the sphere (i.e. the sphere is the parent branch).
        auto bs = arb::impl::branches_from_parent_index(parents, true);
        EXPECT_EQ(2u, bs.size());
        EXPECT_EQ(mb(0,1,1,npos), bs[0]);
        EXPECT_EQ(mb(1,2,3,0), bs[1]);
    }

    {
        std::vector<size_t> parents = {0, 0, 0};

        // A spherical root is not valid: each cable branch would have only one sample.
        EXPECT_THROW(arb::impl::branches_from_parent_index(parents, true), arb::morphology_error);

        // Two cables, with two samples each, with the first sample in each being the root
        auto bc = arb::impl::branches_from_parent_index(parents, false);
        EXPECT_EQ(2u, bc.size());
        EXPECT_EQ(mb(0,1,2,npos), bc[0]);
        EXPECT_EQ(mb(0,2,3,npos), bc[1]);
    }

    {
        std::vector<size_t> parents = {0, 0, 1, 2};

        // With cable soma: one cable with 4 samples.
        auto bc = arb::impl::branches_from_parent_index(parents, false);
        EXPECT_EQ(1u, bc.size());
        EXPECT_EQ(mb(0,1,4,npos), bc[0]);

        // With spherical soma: one sphere and on 3-segment cable.
        // The cable branch is attached to the sphere (i.e. the sphere is the parent branch).
        auto bs = arb::impl::branches_from_parent_index(parents, true);
        EXPECT_EQ(2u, bs.size());
        EXPECT_EQ(mb(0,1,1,npos), bs[0]);
        EXPECT_EQ(mb(1,2,4,0), bs[1]);
    }

    {
        std::vector<size_t> parents = {0, 0, 1, 0};

        // With cable soma: two cables with 3 and 2 samples respectively.
        auto bc = arb::impl::branches_from_parent_index(parents, false);
        EXPECT_EQ(2u, bc.size());
        EXPECT_EQ(mb(0,1,3,npos), bc[0]);
        EXPECT_EQ(mb(0,3,4,npos), bc[1]);

        // A spherical root is not valid: the second cable branch would have only one sample.
        EXPECT_THROW(arb::impl::branches_from_parent_index(parents, true), arb::morphology_error);
    }

    {
        std::vector<size_t> parents = {0, 0, 1, 0, 3};

        // With cable soma: two cables with 3 samples each [0,1,2] and [0,3,4]
        auto bc = arb::impl::branches_from_parent_index(parents, false);
        EXPECT_EQ(2u, bc.size());
        EXPECT_EQ(mb(0,1,3,npos), bc[0]);
        EXPECT_EQ(mb(0,3,5,npos), bc[1]);

        // With spherical soma: one sphere and 2 2-sample cables.
        // each of the cable branches is attached to the sphere (i.e. the sphere is the parent branch).
        auto bs = arb::impl::branches_from_parent_index(parents, true);
        EXPECT_EQ(3u, bs.size());
        EXPECT_EQ(mb(0,1,1,npos), bs[0]);
        EXPECT_EQ(mb(1,2,3,0), bs[1]);
        EXPECT_EQ(mb(3,4,5,0), bs[2]);
    }

    {
        std::vector<size_t> parents = {0, 0, 1, 0, 3, 4, 4, 6};

        // With cable soma: 4 cables: [0,1,2] [0,3,4] [4,5] [4,6,7]
        auto bc = arb::impl::branches_from_parent_index(parents, false);
        EXPECT_EQ(4u, bc.size());
        EXPECT_EQ(mb(0,1,3,npos), bc[0]);
        EXPECT_EQ(mb(0,3,5,npos), bc[1]);
        EXPECT_EQ(mb(4,5,6,1), bc[2]);
        EXPECT_EQ(mb(4,6,8,1), bc[3]);

        // With spherical soma: 1 sphere and 4 cables: [1,2] [3,4] [4,5] [4,6,7]
        auto bs = arb::impl::branches_from_parent_index(parents, true);
        EXPECT_EQ(5u, bs.size());
        EXPECT_EQ(mb(0,1,1,npos), bs[0]);
        EXPECT_EQ(mb(1,2,3,0), bs[1]);
        EXPECT_EQ(mb(3,4,5,0), bs[2]);
        EXPECT_EQ(mb(4,5,6,2), bs[3]);
        EXPECT_EQ(mb(4,6,8,2), bs[4]);
    }
}
*/

// For different parent index vectors, attempt multiple valid and invalid sample sets.
TEST(morphology, segments) {
    using arb::util::make_span;
    using ms = arb::msample;
    {
        std::vector<size_t> p = {0, 0};
        std::vector<ms> s = {
            {{0.0, 0.0, 0.0, 1.0}, 1},
            {{0.0, 0.0, 1.0, 1.0}, 1} };

        arb::sample_tree sm(s, p);
        auto m = arb::morphology(sm);

        EXPECT_EQ(1u, m.num_branches());
    }
    {
        std::vector<size_t> p = {0, 0, 1};
        { // 2-segment cable (1 seg soma, 1 seg dendrite)
            std::vector<ms> s = {
                {{0.0, 0.0, 0.0, 5.0}, 1},
                {{0.0, 0.0, 5.0, 1.0}, 1},
                {{0.0, 0.0, 8.0, 1.0}, 2} };

            arb::sample_tree sm(s, p);
            auto m = arb::morphology(sm);

            EXPECT_EQ(1u, m.num_branches());
        }
        { // spherical soma and single-segment cable
            std::vector<ms> s = {
                {{0.0, 0.0, 0.0, 5.0}, 1},
                {{0.0, 0.0, 1.0, 1.0}, 2},
                {{0.0, 0.0, 8.0, 1.0}, 2} };

            arb::sample_tree sm(s, p);
            auto m = arb::morphology(sm);

            EXPECT_EQ(2u, m.num_branches());
        }
        /* not a bug anymore?
        { // spherical soma and single-segment cable
          // error because the terminal point on the cable section is collocated.
            std::vector<ms> s = {
                {{0.0, 0.0, 0.0, 5.0}, 1},
                {{0.0, 0.0, 1.0, 1.0}, 2},
                {{0.0, 0.0, 1.0, 1.0}, 2} };

            arb::sample_tree sm(s, p);
            EXPECT_THROW((arb::morphology(sm)), arb::morphology_error);
        }
        */
    }
    {
        //              0       |
        //            1   3     |
        //          2           |
        std::vector<size_t> p = {0, 0, 1, 0};
        {
            // two cables: 1x2 segments, 1x1 segment.
            std::vector<ms> s = {
                {{0.0, 0.0, 0.0, 5.0}, 1},
                {{0.0, 0.0, 5.0, 1.0}, 1},
                {{0.0, 0.0, 6.0, 1.0}, 2},
                {{0.0, 4.0, 0.0, 1.0}, 1}};

            arb::sample_tree sm(s, p);
            auto m = arb::morphology(sm);

            EXPECT_EQ(2u, m.num_branches());
        }
        {
            // error: spherical soma with a single point cable attached via sample 3
            std::vector<ms> s = {
                {{0.0, 0.0, 0.0, 5.0}, 1},
                {{0.0, 0.0, 5.0, 1.0}, 2},
                {{0.0, 0.0, 8.0, 1.0}, 2},
                {{0.0, 5.0, 0.0, 1.0}, 2}};

            arb::sample_tree sm(s, p);
            EXPECT_THROW((arb::morphology(sm)), arb::morphology_error);
        }
    }
    {
        //              0       |
        //            1   3     |
        //          2       4    |
        std::vector<size_t> p = {0, 0, 1, 0, 3};
        {
            // two cables: 1x2 segments, 1x1 segment.
            std::vector<ms> s = {
                {{0.0, 0.0, 0.0, 5.0}, 1},
                {{0.0, 0.0, 5.0, 1.0}, 2},
                {{0.0, 0.0, 8.0, 1.0}, 2},
                {{0.0, 5.0, 0.0, 1.0}, 2},
                {{0.0, 8.0, 0.0, 1.0}, 2}};

            arb::sample_tree sm(s, p);
            auto m = arb::morphology(sm);

            EXPECT_EQ(3u, m.num_branches());
        }
    }
}

// test that morphology generates branch child-parent structure correctly.
TEST(morphology, branches) {
    using vec = std::vector<size_t>;
    auto npos = arb::mbranch::npos;
    {
        // 0
        vec parents = {0};
        std::vector<arb::msample> samples = {
            {{ 0,0,0,3}, 1},
        };
        arb::sample_tree sm(samples, parents);
        arb::morphology m(sm);

        EXPECT_EQ(1u, m.num_branches());
        EXPECT_EQ(npos, m.branch_parent(0));
        EXPECT_EQ(vec{}, m.branch_children(0));
    }
    {
        // 0 - 1
        vec parents = {0, 0};
        std::vector<arb::msample> samples = {
            {{ 0,0,0,3}, 1},
            {{ 10,0,0,3}, 1},
        };
        arb::sample_tree sm(samples, parents);
        arb::morphology m(sm);

        EXPECT_EQ(1u, m.num_branches());
        EXPECT_EQ(npos, m.branch_parent(0));
        EXPECT_EQ(vec{}, m.branch_children(0));
    }
    {
        // 0 - 1 - 2
        vec parents = {0, 0, 1};
        {
            // All samples have same tag -> the morphology is a single unbranched cable.
            std::vector<arb::msample> samples = {
                {{ 0,0,0,3}, 1},
                {{10,0,0,3}, 1},
                {{100,0,0,3}, 1},
            };
            arb::sample_tree sm(samples, parents);
            arb::morphology m(sm);

            EXPECT_EQ(1u, m.num_branches());
            EXPECT_EQ(npos, m.branch_parent(0));
            EXPECT_EQ(vec{}, m.branch_children(0));
        }
        {
            // First sample has unique tag -> spherical soma attached to a single-segment cable.
            std::vector<arb::msample> samples = {
                {{  0,0,0,10}, 1},
                {{ 10,0,0, 3}, 3},
                {{100,0,0, 3}, 3},
            };
            arb::sample_tree sm(samples, parents);
            arb::morphology m(sm);

            EXPECT_EQ(2u, m.num_branches());
            EXPECT_EQ(npos, m.branch_parent(0));
            EXPECT_EQ(0u,   m.branch_parent(1));
            EXPECT_EQ(vec{1}, m.branch_children(0));
            EXPECT_EQ(vec{},  m.branch_children(1));
        }
    }
    {
        // 2 - 0 - 1
        vec parents = {0, 0, 0};

        std::vector<arb::msample> samples = {
            {{  0, 0,0, 5}, 3},
            {{ 10, 0,0, 5}, 3},
            {{  0,10,0, 5}, 3},
        };
        arb::sample_tree sm(samples, parents);
        arb::morphology m(sm);

        EXPECT_EQ(2u, m.num_branches());
        EXPECT_EQ(npos, m.branch_parent(0));
        EXPECT_EQ(npos,   m.branch_parent(1));
        EXPECT_EQ(vec{}, m.branch_children(0));
        EXPECT_EQ(vec{},  m.branch_children(1));
    }
    {
        // 0 - 1 - 2 - 3
        vec parents = {0, 0, 1, 2};
    }
    {
        //              0       |
        //             / \      |
        //            1   3     |
        //           /          |
        //          2           |
        vec parents = {0, 0, 1, 0};
    }
    {
        // Eight samples
        //
        //              0           |
        //             / \          |
        //            1   3         |
        //           /     \        |
        //          2       4       |
        //                 / \      |
        //                5   6     |
        //                     \    |
        //                      7   |
        vec parents = {0, 0, 1, 0, 3, 4, 4, 6};
        {
            std::vector<arb::msample> samples = {
                {{  0,  0,  0, 10}, 1},
                {{ 10,  0,  0,  2}, 3},
                {{100,  0,  0,  2}, 3},
                {{  0, 10,  0,  2}, 3},
                {{  0,100,  0,  2}, 3},
                {{100,100,  0,  2}, 3},
                {{  0,200,  0,  2}, 3},
                {{  0,300,  0,  2}, 3},
            };
            arb::sample_tree sm(samples, parents);
            arb::morphology m(sm);

            EXPECT_EQ(5u, m.num_branches());
            EXPECT_EQ(npos, m.branch_parent(0));
            EXPECT_EQ(0u,   m.branch_parent(1));
            EXPECT_EQ(0u,   m.branch_parent(2));
            EXPECT_EQ(2u,   m.branch_parent(3));
            EXPECT_EQ(2u,   m.branch_parent(4));
            EXPECT_EQ((vec{1,2}), m.branch_children(0));
            EXPECT_EQ((vec{}),    m.branch_children(1));
            EXPECT_EQ((vec{3,4}), m.branch_children(2));
            EXPECT_EQ((vec{}),    m.branch_children(3));
            EXPECT_EQ((vec{}),    m.branch_children(4));
        }
        {
            std::vector<arb::msample> samples = {
                {{  0,  0,  0, 10}, 3},
                {{ 10,  0,  0,  2}, 3},
                {{100,  0,  0,  2}, 3},
                {{  0, 10,  0,  2}, 3},
                {{  0,100,  0,  2}, 3},
                {{100,100,  0,  2}, 3},
                {{  0,200,  0,  2}, 3},
                {{  0,300,  0,  2}, 3},
            };
            arb::sample_tree sm(samples, parents);
            arb::morphology m(sm);

            EXPECT_EQ(4u, m.num_branches());
            EXPECT_EQ(npos, m.branch_parent(0));
            EXPECT_EQ(npos, m.branch_parent(1));
            EXPECT_EQ(1u,   m.branch_parent(2));
            EXPECT_EQ(1u,   m.branch_parent(3));
            EXPECT_EQ((vec{}),    m.branch_children(0));
            EXPECT_EQ((vec{2,3}), m.branch_children(1));
            EXPECT_EQ((vec{}),    m.branch_children(2));
            EXPECT_EQ((vec{}),    m.branch_children(3));
        }
    }
}

TEST(morphology, swc) {
    std::string datadir{DATADIR};
    auto fname = datadir + "/example.swc";
    std::ifstream fid(fname);
    if (!fid.is_open()) {
        std::cerr << "unable to open file " << fname << "... skipping test\n";
        return;
    }

    // Load swc samples from file.
    auto swc_samples = arb::parse_swc_file(fid);

    // Build a sample_tree from swc samples.
    auto sm = arb::swc_as_sample_tree(swc_samples);
    EXPECT_EQ(1058u, sm.size()); // file contains 195 samples

    // Test that the morphology contains the expected number of branches.
    auto m = arb::morphology(sm);
    EXPECT_EQ(31u, m.num_branches());

    // Confirm that converting to a cable_cell generates the same number of branches.
    auto c = arb::make_cable_cell(m, false);
    EXPECT_EQ(31u, c.num_segments());
}

