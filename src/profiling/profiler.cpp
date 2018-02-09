#include <cstdio>
#include <iostream>

#include <util/span.hpp>
#include <util/rangeutil.hpp>

#include "profiler.hpp"

namespace arb {
namespace util {

namespace impl {
    bool is_valid_region_string(const std::string& s) {
        if (s.size()==0u || s.front()=='_' || s.back()=='_') return false;
        return s.find("__") == s.npos;
    }

    std::vector<std::string> split(const std::string& str) {
        std::vector<std::string> cont;
        std::size_t first = 0;
        std::size_t last = str.find('_');
        while (last != std::string::npos) {
            cont.push_back(str.substr(first, last - first));
            first = last + 1;
            last = str.find('_', first);
        }
        cont.push_back(str.substr(first, last - first));
        return cont;
    }
}

namespace data {
    arb::util::profiler profiler;
}

// recorder implementation

const std::vector<sample>& recorder::samples() const {
    return samples_;
}

void recorder::enter(std::size_t index) {
    stamps_.push_back({index, timer_type::tic()});
    if (index>=samples_.size()) {
        samples_.resize(index+1);
    }
}

void recorder::leave() {
    if (stamps_.size()==0) {
        throw std::runtime_error("attempting to leave root region");
    }
    const auto& last = stamps_.back();
    samples_[last.index].count++;
    samples_[last.index].time += timer_type::toc(last.time);
    stamps_.pop_back();
}

void recorder::clear() {
    stamps_.clear();
    for (auto& s:samples_) {
        s.time = 0;
        s.count = 0;
    }
}

// profiler implementation

profiler::profiler() {
    std::cout << "INITIALIZING PROFILER WITH " << threading::num_threads() << " threads\n";
    recorders_.resize(threading::num_threads());
}

void profiler::enter(std::size_t index) {
    recorders_[threading::thread_id()].enter(index);
}

void profiler::enter(const char* name) {
    const auto index = index_from_name(name);
    recorders_[threading::thread_id()].enter(index);
}

void profiler::leave() {
    recorders_[threading::thread_id()].leave();
}

void profiler::start() {
    if (running_) {
        throw std::runtime_error("Can't start a profiler that is running.");
    }
    running_ = true;
    tstart_ = timer_type::tic();
    tstart_= timer_type::tic();
}

void profiler::stop() {
    if (!running_) {
        throw std::runtime_error("Can't stop a profiler that isn't running.");
    }
    running_ = false;
    tstop_ = timer_type::tic();
}

void profiler::restart() {
    if (running_) {
        throw std::runtime_error("Can't restart a profiler that is running.");
    }
    for (auto& r: recorders_) {
        r.clear();
    }
    tstart_ = timer_type::tic();
}

std::size_t profiler::index_from_name(const char* name) {
    // The name_index_ hash table is shared by all threads, so all access
    // has to be protected by a mutex.
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = name_index_.find(name);
    if (it==name_index_.end()) {
        const auto index = region_names_.size();
        name_index_[name] = index;
        region_names_.emplace_back(name);
        return index;
    }
    return it->second;
}

void print(
        profile_node& n,
        float total_time,
        unsigned nthreads,
        float thresh=0.5,
        std::string indent="")
{
    auto name = indent + n.name;
    float thread_time = n.time/nthreads;
    float proportion = thread_time/total_time*100;

    // If the percentage of overall time for this region is below the
    // threashold, stop drawing this branch.
    if (proportion<thresh) return;

    printf(
        "%-20s%8lu%12.3f%12.3f%8.1f\n",
        name.c_str(), n.count, float(n.time), thread_time, proportion);

    // sort the children in descending order of time taken
    util::sort_by(n.children, [](const profile_node& n){return -n.time;});
    // print each of the children in turn
    for (auto& c: n.children) print(c, total_time, nthreads, thresh, indent+"  ");
};

profile profiler::results() const {
    using std::vector;
    using std::string;
    using util::make_span;
    using util::assign_from;
    using util::transform_view;

    profile p;
    p.time_taken = timer_type::difference(tstart_, tstop_);
    p.names = region_names_;
    const auto nreg = p.names.size();
    p.samples.reserve(recorders_.size());
    for (auto& r: recorders_) {
        p.samples.push_back(r.samples());
    }

    std::vector<double> times(nreg);
    std::vector<std::size_t> counts(nreg);
    for (auto& thread_samples: p.samples) {
        for (auto i: make_span(0, thread_samples.size())) {
            auto& s = thread_samples[i];
            times[i] += s.time;
            counts[i] += s.count;
        }
    }

    // build a tree description of the regions and sub-regions in the profile.
    vector<vector<string>> names = assign_from(transform_view(p.names, impl::split));

    std::vector<size_t> index = assign_from(make_span(0, nreg));
    util::sort_by(index, [&](std::size_t i){return names[i].size();});

    const auto nthread = threading::num_threads();
    profile_node tree("Total", nthread*p.time_taken, 1);
    auto depth = 1u;
    auto j = 0u;
    auto idx = index[j];
    while (j<nreg) {
        while (j<nreg && names[idx].size()==depth) {
            profile_node* node = &tree;
            for (auto i=0u; i<depth-1; ++i) {
                auto& node_name = names[idx][i];
                auto& kids = node->children;

                // find child of node that matches node_name
                auto child = std::find_if(
                    kids.begin(), kids.end(), [&](profile_node& n){return n.name==node_name;});

                if (child==kids.end()) {
                    // TODO: we have to populate these nodes with time and count information
                    // derived from their children.
                    node->children.emplace_back(node_name, -1, 0);
                    node = &node->children.back();
                }
                else {
                    node = &(*child);
                }
            }
            node->children.emplace_back(names[idx].back(), times[idx], counts[idx]);
            ++j;
            idx = index[j];
        }
        ++depth;
    }

    p.tree = std::move(tree);

    return p;
}

const std::vector<std::string>& profiler::regions() const {
    return region_names_;
}

//
// convenience functions for instrumenting code.
//

#ifdef ARB_HAVE_PROFILING

void profiler_leave() {
    ::arb::util::data::profiler.leave();
}

void profiler_leave(unsigned n) {
    while (n--) {
        ::arb::util::data::profiler.leave();
    }
}

void profiler_start() {
    ::arb::util::data::profiler.start();
}

void profiler_stop() {
    ::arb::util::data::profiler.stop();
}

void profiler_restart() {
    ::arb::util::data::profiler.restart();
}

void profiler_print() {
    using util::make_span;

    std::cout << "\n-- PROFILER RESULTS --\n\n";
    auto results = data::profiler.results();

    const auto nthreads = results.samples.size();
    if (nthreads==0u) {
        std::cout << "\n  no results\n";
        return;
    }

    printf("------------------------------------------------------------\n");
    printf("%-20s%8s%12s%12s\n", "region", "calls", "time", "thread-time");
    printf("------------------------------------------------------------\n");
    print(results.tree, results.time_taken, nthreads, 0, "");
    printf("------------------------------------------------------------\n\n");
}

#else
void profiler_leave() {}
void profiler_leave(unsigned n) {}
void profiler_start() {}
void profiler_stop() {}
void profiler_restart() {}
void profiler_print() {}
#endif

} // namespace util
} // namespace arb
