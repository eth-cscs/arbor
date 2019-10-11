#pragma once

namespace arb {
namespace prefetch {
// Utilities

// prefetching wrapper
// First types for deduction in make_prefetch
// mode types that encapsulate 0 or 1
template<int v>
struct mode_type {
    static constexpr auto value = v;
    constexpr operator int() const {return value;}
};

//   constants: read or write
static constexpr mode_type<0> read;
static constexpr mode_type<1> write;

template<int v>
struct locality_type {
    static constexpr auto value = v;
    constexpr operator int() const {return value;}
};

//   constants: none, low, medium or high
static constexpr locality_type<0> none;
static constexpr locality_type<1> low;
static constexpr locality_type<2> medium;
static constexpr locality_type<3> high;

// default conversion from pointer-like P to pointer P
// set up this way so that it can be specialized for unusual P
// but this implies that *prefetched is a valid operation
template<typename P>
inline auto get_pointer(P&& prefetched) noexcept {
    return &*std::forward<P>(prefetched);
}

// if it's a plain pointer, can even be invalid
template<typename P>
inline auto get_pointer(P* prefetched) noexcept {
    return prefetched;
}

// encapsulate __builtin_prefetch
// uses get_pointer to convert pointer-like to pointer
// mode = 0 (read) | 1 (write)
// locality = 0 (none) | 1 (low) | 2 (medium) | 3 (high)
template<int mode=1, int locality=3, typename P> // do the prefetch
inline void fetch(P&& p) noexcept {
    __builtin_prefetch(get_pointer(std::forward<P>(p)), mode, locality);
}

// call fetch with mode_type, locality_type arguments
template<typename P, int mode, int locality>
inline void fetch(P&& p, mode_type<mode>, locality_type<locality>) noexcept {
    fetch<mode, locality>(std::forward<P>(p));
}

template<typename P, int mode>
inline void fetch(P&& p, mode_type<mode>) noexcept {
    fetch<mode>(std::forward<P>(p));
}

// some template tests for enabling
// check for triviality
template<typename T, typename U = int>
using is_trivially_destructible_t = typename std::enable_if_t<std::is_trivially_destructible<U>::value, U>;

template<typename T, typename U = int>
using is_not_trivially_destructible_t = typename std::enable_if_t<!std::is_trivially_destructible<T>::value, U>;

// check for same type
template<typename T, typename U>
using is_decay_same = std::is_same<std::decay_t<T>, std::decay_t<U>>;

template<typename T, typename U>
using enable_if_decay_same_t = std::enable_if_t<is_decay_same<T, U>::value>;

/*
  two class: buffer && prefetch:

  buffer<std::size_t s, typename.. Ts>
  which is a ring buffer of that stores up to 
  s tuples of type Ts...

  and

  prefetch<int mode, buffer, function>
  which binds the buffer within scope with a function
  and prefetches every store with mode.

  a buffer is declared in an outer, permanent scope:
  buffer<n, Args...> b;

  and then:
  void doit() {
    auto&& a = make_prefetch(
      prefetch::read or prefetch::write,
      b,
      [] (auto&& element) {
        do-something(element);
      });

      // where element is a tuple<Args...>
      for (obj: vec) {
        a.store(obj.pointer, {obj.pointer, obj.arg, ...});
      }
      // and {obj.pointer, obj.arg, ...} constructs tupe<Args...>
  }
  
  prefetches an address-like P,
  stores a list of argument addresses, 
  and later calls a function on them when full (or at destruction)

  the concept is that you continously `store` addresses from cuts through arrays until full
  (see `store()`). Prefetch is called on some associated address (by default the first
  `store` argument) before storing.

  When full, a functor is called on all the arguments in the hope that
  the prefetch has already pulled the data associated with the prefetch-pointer.


  The "hard bit" is determining the range of good sizes for the capacity.
  How many prefetch should we add before calling e.process?
  Not too many or we will be pushing things out of the cache
  and not too few or we'll hit the function application before
  the data has arrived

  prefetch_size s = number of read-aheads
  prefetch_mode m = read or write, do we prefetch with a read or write expection?
  Types... =  payload types for calling functions on (pointer-like objects) with qualifiers for f
  process is applied on f: f(Types...)
*/

// ring_buffer: E should have trivial constructor, destructor
template<std::size_t s, typename E>
struct ring_buffer_types {
    using element_type = E;
    
    template<typename T>
    using enable_if_element_t = enable_if_decay_same_t<T, element_type>;
    
    static constexpr auto size = s;
};

// requirement: s > 0, sizeof(E) > 0
template<std::size_t s, typename E>
class ring_buffer: public ring_buffer_types<s, E> {
public:
    using types = ring_buffer_types<s, E>;
    using typename types::element_type;
    
    template<typename T>
    using enable_if_element_t = typename types::template enable_if_element_t<T>;
    
    using types::size;
    
    ring_buffer() = default; // only default construct, no copies
    ring_buffer(ring_buffer&&) = delete;
    ring_buffer(const ring_buffer&) = delete;
    ring_buffer& operator=(ring_buffer&&) = delete;
    ring_buffer& operator=(const ring_buffer&) = delete;
    
    ~ring_buffer() {deconstruct();} // if needed, kill elements
    
    // precondition: ! is_full()
    template<typename T, typename = enable_if_element_t<T>>
    void push(T&& e) noexcept {
        push_emplace(std::forward<T>(e));
    }
    
    // precondition: ! is_full()
    template<typename... Ts>
    void push_emplace(Ts&&... args) noexcept {
        new (stop) element_type{std::forward<Ts>(args)...};
        stop = next;
        if (++next == end) {next = begin;}
    }
    
    // precondition: ! is_empty()
    element_type& pop() noexcept {
        invalidate(); // if popped element alive, deconstruct if needed
        const auto head = start;
        if (++start == end) {start = begin;}
        return *head; // only valid until next pop happens
    }
    
    bool empty() const noexcept {return start == stop;}
    bool full()  const noexcept {return start == next;}
    
private:
    template<typename U = element_type, is_trivially_destructible_t<U> = 0>
    void deconstruct() noexcept {}
    template<typename U = element_type, is_trivially_destructible_t<U> = 0>
    void invalidate() noexcept {}
    
    // deconstruct all elements, if not trivially destructible
    template<typename U = element_type, is_not_trivially_destructible_t<U> = 0>
    void deconstruct() noexcept {
        while (valid != stop) {
            valid->~element_type();
            if (++valid == end) {valid = begin;}
        }
    }
    
    // deconstruct last popped off, if not trivially destructible
    template<typename U = element_type, is_not_trivially_destructible_t<U> = 0>
    void invalidate() noexcept {
        if (valid != start) {
            valid->~element_type();
            valid = start;
        }
    }
    
    // ring buffer storage using an extra sentinel element
    static constexpr auto arraylen = size + 1;
    alignas(element_type) char array[sizeof(element_type)*arraylen];
    typedef element_type *iterator;
    const iterator begin = reinterpret_cast<iterator>(array);
    const iterator end   = begin + arraylen;
    
    // array pointers
    iterator start = begin;  // first element to pop off
    iterator valid = begin;  // last valid element, at most one behind start
    iterator stop  = begin;  // next element to push into
    iterator next  = stop+1; // sentinel: next == begin, we're out of space
};

template<typename E>
class ring_buffer<0, E>: public ring_buffer_types<0, E>
{};

template<std::size_t s, typename... Ts>
using buffer = ring_buffer<s, std::tuple<Ts...>>;

template<int m, int l, typename B, typename F>
struct prefetch_types {
    using buffer_type = B;
    using element_type = typename buffer_type::element_type;
    using function_type = F;
    
    template<typename E>
    using enable_if_element_t = typename buffer_type::template enable_if_element_t<E>;
    
    static constexpr auto mode = m;
    static constexpr auto locality = l;
    static constexpr auto size = buffer_type::size;
};

template<int m, int l, typename B, typename F>
class prefetch_base: public prefetch_types<m, l, B, F> {
public:
    using types = prefetch_types<m, l, B, F>;
    using typename types::buffer_type;
    using typename types::function_type;
    
    template<typename E>
    using enable_if_element_t = typename types::template enable_if_element_t<E>;
    
    using types::mode;
    using types::locality;
    
    prefetch_base(buffer_type& b_, function_type&& f) noexcept: b(b_), function{std::move(f)} {}
    prefetch_base(buffer_type& b_, const function_type& f) noexcept: b(b_), function{f} {}
    
    ~prefetch_base() noexcept { // clear buffer on destruct
        while (! b.empty()) {pop();}
    }
    
protected:    
    // append an element to process after
    // prefetching pointer-like P associated
    // with pointer-like args to be passed to F function.
    // If enough look-aheads pending process one (call F function on it).
    template<typename P, typename... Ts>
    void store_internal(P&& p, Ts&&... args) noexcept {
        fetch<mode, locality>(std::forward<P>(p));
        if (b.full()) {pop();} // process and remove if look ahead full
        push(std::forward<Ts>(args)...); // add new look ahead
    }
    
private:
    // apply function to first stored, and move pointer forward
    // precondition: begin != end
    void pop() noexcept {
        function(b.pop());
    }
    
    // add an element to end of ring
    // precondition: begin != next
    template<typename E, typename = enable_if_element_t<E>>
    void push(E&& e) noexcept {
        b.push(std::forward<E>(e));
    }
    
    template<typename... Ts>
    void push(Ts&&... args) noexcept {
        b.push_emplace(std::forward<Ts>(args)...);
    }
    
    buffer_type& b;
    const function_type function;
};

template<int m, int l, typename B, typename F>
class prefetch_base_zero: public prefetch_types<m, l, B, F> {
public:
    using types = prefetch_types<m, l, B, F>;
    using typename types::buffer_type;
    using typename types::element_type;
    using typename types::function_type;
    
    prefetch_base_zero(buffer_type&, function_type&& f) noexcept: function{std::move(f)} {}
    prefetch_base_zero(buffer_type&, const function_type& f) noexcept: function{f} {}
    
protected:
    template<typename P, typename... Ts>
    void store_internal(P&&, Ts&&... args) noexcept {
        function(element_type{std::forward<Ts>(args)...});
    }
    
private:
    const function_type function;
};

// specialization to turn off prefetch
template<int m, int l, typename... Ts, typename F>
class prefetch_base<m, l, buffer<0, Ts...>, F>: public prefetch_base_zero<m, l, buffer<0, Ts...>, F> {
public:
    using base = prefetch_base_zero<m, l, buffer<0, Ts...>, F>;
    using base::base;
};

// prefetch class: storage is in prefetch_base
template<int m, int l, typename B, typename F>
class prefetch: public prefetch_base<m, l, B, F> {
public:
    using base = prefetch_base<m, l, B, F>;
    using typename base::element_type;
    
    using base::base;
    prefetch(prefetch&&) noexcept = default;
    
    prefetch(const prefetch&) = delete;
    prefetch& operator=(prefetch&&) = delete;
    prefetch& operator=(const prefetch&) = delete;
    
    template<typename P>
    void store(P&& p, const element_type& e) noexcept {
        store_internal(std::forward<P>(p), e);
    }
    
    template<typename P>
    void store(P&& p, element_type&& e) noexcept {
        store_internal(std::forward<P>(p), std::move(e));
    }
    
    template<typename P, typename... Ts>
    void store(P&& p, Ts&&... args) noexcept {
        store_internal(std::forward<P>(p), std::forward<Ts>(args)...);
    }

private:
    using base::store_internal;
};

/* make_prefetch: returns a constructed a prefetch instance
   hopefully returns elided on the stack
*/

// and now the utility functions `make_prefetch`

// make_prefetch(
//    prefetch::read|write,
//    prefetch::none|low|middle|high,
//    buffer,
//    [] (auto&& element) {}
// )
template<int m, int l, typename B, typename F>
inline constexpr auto make_prefetch(mode_type<m>, locality_type<l>, B& buf, F&& function) noexcept {
    return prefetch<m, l, B, F>{buf, std::forward<F>(function)};
}

} //prefetch
} //arb