#pragma once

/* 
 * Presents a half-open interval [a,b) of integral values as a container.
 */

#include <type_traits>
#include <utility>

#include "util/counter.hpp"
#include "util/meta.hpp"
#include "util/range.hpp"

namespace arb {
namespace util {

// TODO: simplify span-using code by:
// 1. replace type alias `span` with `span_type` alias;
// 2. rename `make_span` as `span`
// 3. add another `span(I n)` overload equivalent to `span(I{}, n)`.

template <typename I>
using span = range<counter<I>>;

template <typename I, typename J, typename = std::enable_if_t<std::is_integral<std::common_type_t<I, J>>::value>>
span<std::common_type_t<I, J>> make_span(I left, J right) {
    return span<std::common_type_t<I, J>>(left, right);
}

template <typename I, typename J>
span<std::common_type_t<I, J>> make_span(std::pair<I, J> interval) {
    return make_span(interval.first, interval.second);
}

template <typename I>
span<I> make_span(I right) {
    return make_span(I{}, right);
}

template <typename Seq>
auto count_along(const Seq& s) {
    return make_span(0ul, std::size(s));
}

} // namespace util
} // namespace arb
