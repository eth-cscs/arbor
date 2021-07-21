#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#include <arbor/simd/implbase.hpp>
#include <arbor/simd/generic.hpp>
#include <arbor/simd/native.hpp>
#include <arbor/util/pp_util.hpp>

namespace arb {
namespace simd {

namespace detail {
    template <typename Impl>
    struct simd_impl;

    template <typename Impl>
    struct simd_mask_impl;

    template <typename To>
    struct simd_cast_impl;

    template <typename I, typename V>
    class indirect_indexed_expression;

    template <typename V>
    class indirect_expression;

    template <typename T, typename M>
    class where_expression;

    template <typename T, typename M>
    class const_where_expression;
}

// Top level functions for second API
template <typename Impl, typename Other>
void assign(detail::simd_impl<Impl>& a, const Other& b) {
    a.copy_from(b);
}

template <typename Impl>
typename detail::simd_impl<Impl>::scalar_type sum(const detail::simd_impl<Impl>& a) {
    return a.sum();
};

#define ARB_UNARY_ARITHMETIC_(name)\
template <typename Impl>\
detail::simd_impl<Impl> name(const detail::simd_impl<Impl>& a) {\
    return detail::simd_impl<Impl>::wrap(Impl::name(a.value_));\
};

#define ARB_BINARY_ARITHMETIC_(name)\
template <typename Impl>\
detail::simd_impl<Impl> name(const detail::simd_impl<Impl>& a, detail::simd_impl<Impl> b) {\
    return detail::simd_impl<Impl>::wrap(Impl::name(a.value_, b.value_));\
};\
template <typename Impl>\
detail::simd_impl<Impl> name(const detail::simd_impl<Impl>& a, typename detail::simd_impl<Impl>::scalar_type b) {\
    return detail::simd_impl<Impl>::wrap(Impl::name(a.value_, Impl::broadcast(b)));\
};\
template <typename Impl>\
detail::simd_impl<Impl> name(const typename detail::simd_impl<Impl>::scalar_type a, detail::simd_impl<Impl> b) {\
    return detail::simd_impl<Impl>::wrap(Impl::name(Impl::broadcast(a), b.value_));\
};

#define ARB_BINARY_COMPARISON_(name)\
template <typename Impl>\
typename detail::simd_impl<Impl>::simd_mask name(const detail::simd_impl<Impl>& a, detail::simd_impl<Impl> b) {\
    return detail::simd_impl<Impl>::mask(Impl::name(a.value_, b.value_));\
};\
template <typename Impl>\
typename detail::simd_impl<Impl>::simd_mask name(const detail::simd_impl<Impl>& a, typename detail::simd_impl<Impl>::scalar_type b) {\
    return detail::simd_impl<Impl>::mask(Impl::name(a.value_, Impl::broadcast(b)));\
};\
template <typename Impl>\
typename detail::simd_impl<Impl>::simd_mask name(const typename detail::simd_impl<Impl>::scalar_type a, detail::simd_impl<Impl> b) {\
    return detail::simd_impl<Impl>::mask(Impl::name(Impl::broadcast(a), b.value_));\
};

ARB_PP_FOREACH(ARB_BINARY_ARITHMETIC_, add, sub, mul, div, pow, max, min)
ARB_PP_FOREACH(ARB_BINARY_COMPARISON_, cmp_eq, cmp_neq, cmp_leq, cmp_lt, cmp_geq, cmp_gt)
ARB_PP_FOREACH(ARB_UNARY_ARITHMETIC_,  neg, abs, sin, cos, exp, log, expm1, exprelr)

#undef ARB_BINARY_ARITHMETIC_
#undef ARB_BINARY_COMPARISON__
#undef ARB_UNARY_ARITHMETIC_

template <typename T>
detail::simd_mask_impl<T> logical_and(const detail::simd_mask_impl<T>& a, detail::simd_mask_impl<T> b) {
    return a && b;
}

template <typename T>
detail::simd_mask_impl<T> logical_or(const detail::simd_mask_impl<T>& a, detail::simd_mask_impl<T> b) {
    return a || b;
}

template <typename T>
detail::simd_mask_impl<T> logical_not(const detail::simd_mask_impl<T>& a) {
    return !a;
}

template <typename T>
detail::simd_impl<T> fma(const detail::simd_impl<T> a, detail::simd_impl<T> b, detail::simd_impl<T> c) {
    return detail::simd_impl<T>::wrap(T::fma(a.value_, b.value_, c.value_));
}

namespace detail {
    /// Indirect Expressions
    template <typename V>
    class indirect_expression {
    public:
        indirect_expression(V* p, unsigned width): p(p), width(width) {}

        indirect_expression& operator=(V s) {
            for (unsigned i = 0; i < width; ++i) {
                p[i] = s;
            }
            return *this;
        }

        template <typename Other>
        indirect_expression& operator=(const Other& s) {
            indirect_copy_to(s, p, width);
            return *this;
        }

        template <typename Impl, typename ImplMask>
        indirect_expression& operator=(const const_where_expression<Impl, ImplMask>& s) {
            indirect_copy_to(s.data_, s.mask_, p, width);
            return *this;
        }

        template <typename Impl, typename ImplMask>
        indirect_expression& operator=(const where_expression<Impl, ImplMask>& s) {
            indirect_copy_to(s.data_, s.mask_, p, width);
            return *this;
        }

        template <typename Impl> friend struct simd_impl;
        template <typename Impl> friend struct simd_mask_impl;
        template <typename To>   friend struct simd_cast_impl;
        template <typename T, typename M> friend class where_expression;

    private:
        V* p;
        unsigned width;
    };

    template <typename Impl, typename V>
    static void indirect_copy_to(const simd_mask_impl<Impl>& s, V* p, unsigned width) {
        Impl::mask_copy_to(s.value_, p);
    }

    template <typename Impl, typename V>
    static void indirect_copy_to(const simd_impl<Impl>& s, V* p, unsigned width) {
        Impl::copy_to(s.value_, p);
    }

    template <typename Impl, typename ImplMask, typename V>
    static void indirect_copy_to(const simd_impl<Impl>& data, const simd_mask_impl<ImplMask>& mask, V* p, unsigned width) {
        Impl::copy_to_masked(data.value_, p, mask.value_);
    }

    /// Indirect Indexed Expressions
    template <typename ImplIndex, typename V>
    class indirect_indexed_expression {
    public:
        indirect_indexed_expression(V* p, const ImplIndex& index_simd, unsigned width, index_constraint constraint):
            p(p), index(index_simd), width(width), constraint(constraint)
        {}

        indirect_indexed_expression& operator=(V s) {
            typename simd_traits<ImplIndex>::scalar_type idx[width];
            ImplIndex::copy_to(index.value_, idx);
            for (unsigned i = 0; i < width; ++i) {
                p[idx[i]] = s;
            }
            return *this;
        }

        template <typename Other>
        indirect_indexed_expression& operator=(const Other& s) {
            indirect_indexed_copy_to(s, p, index, width);
            return *this;
        }

        template <typename Impl, typename ImplMask>
        indirect_indexed_expression& operator=(const const_where_expression<Impl, ImplMask>& s) {
            indirect_indexed_copy_to(s.data_, s.mask_, p, index, width);
            return *this;
        }

        template <typename Impl, typename ImplMask>
        indirect_indexed_expression& operator=(const where_expression<Impl, ImplMask>& s) {
            indirect_indexed_copy_to(s.data_, s.mask_, p, index, width);
            return *this;
        }

        template <typename Other>
        indirect_indexed_expression& operator+=(const Other& s) {
            compound_indexed_add(s, p, index, width, constraint);
            return *this;
        }

        template <typename Other>
        indirect_indexed_expression& operator-=(const Other& s) {
            compound_indexed_add(neg(s), p, index, width, constraint);
            return *this;
        }

        template <typename Impl> friend struct simd_impl;
        template <typename To>   friend struct simd_cast_impl;
        template <typename T, typename M> friend class where_expression;

    private:
        V* p;
        const ImplIndex& index;
        unsigned width;
        index_constraint constraint;
    };

    template <typename Impl, typename ImplIndex, typename V>
    static void indirect_indexed_copy_to(const simd_impl<Impl>& s, V* p, const simd_impl<ImplIndex>& index, unsigned width) {
        Impl::scatter(tag<ImplIndex>{}, s.value_, p, index.value_);
    }

    template <typename Impl, typename ImplIndex, typename ImplMask, typename V>
    static void indirect_indexed_copy_to(const simd_impl<Impl>& data, const simd_mask_impl<ImplMask>& mask, V* p, const simd_impl<ImplIndex>& index, unsigned width) {
        Impl::scatter(tag<ImplIndex>{}, data.value_, p, index.value_, mask.value_);
    }

    template <typename ImplIndex, typename Impl, typename V>
    static void compound_indexed_add(
        const simd_impl<Impl>& s,
        V* p,
        const simd_impl<ImplIndex>& index,
        unsigned width,
        index_constraint constraint)
    {
        switch (constraint) {
            case index_constraint::none:
            {
                typename ImplIndex::scalar_type o[width];
                ImplIndex::copy_to(index.value_, o);

                V a[width];
                Impl::copy_to(s.value_, a);

                V temp = 0;
                for (unsigned i = 0; i<width-1; ++i) {
                    temp += a[i];
                    if (o[i] != o[i+1]) {
                        p[o[i]] += temp;
                        temp = 0;
                    }
                }
                temp += a[width-1];
                p[o[width-1]] += temp;
            }
                break;
            case index_constraint::independent:
            {
                auto v = Impl::add(Impl::gather(tag<ImplIndex>{}, p, index.value_), s.value_);
                Impl::scatter(tag<ImplIndex>{}, v, p, index.value_);
            }
                break;
            case index_constraint::contiguous:
            {
                p += ImplIndex::element0(index.value_);
                auto v = Impl::add(Impl::copy_from(p), s.value_);
                Impl::copy_to(v, p);
            }
                break;
            case index_constraint::constant:
                p += ImplIndex::element0(index.value_);
                *p += Impl::reduce_add(s.value_);
                break;
        }
    }

    /// Where Expressions
    template <typename Impl, typename ImplMask>
    class where_expression {
    public:
        where_expression(const ImplMask& m, Impl& s):
                mask_(m), data_(s) {}

        template <typename Other>
        where_expression& operator=(const Other& v) {
            where_copy_to(mask_, data_, v);
            return *this;
        }

        template <typename V>
        where_expression& operator=(const indirect_expression<V>& v) {
            where_copy_to(mask_, data_, v.p, v.width);
            return *this;
        }

        template <typename ImplIndex, typename V>
        where_expression& operator=(const indirect_indexed_expression<ImplIndex, V>& v) {
            where_copy_to(mask_, data_, v.p, v.index, v.width);
            return *this;
        }

        template <typename T>             friend struct simd_impl;
        template <typename To>            friend struct simd_cast_impl;
        template <typename V>             friend class indirect_expression;
        template <typename I, typename V> friend class indirect_indexed_expression;

    private:
        const ImplMask& mask_;
        Impl& data_;
    };

    template <typename Impl, typename ImplMask, typename V>
    static void where_copy_to(const simd_mask_impl<ImplMask>& mask, simd_impl<Impl>& f, const V& t) {
        f.value_ = Impl::ifelse(mask.value_, Impl::broadcast(t), f.value_);
    }

    template <typename Impl, typename ImplMask>
    static void where_copy_to(const simd_mask_impl<ImplMask>& mask, simd_impl<Impl>& f, const simd_impl<Impl>& t) {
        f.value_ = Impl::ifelse(mask.value_, t.value_, f.value_);
    }

    template <typename Impl, typename ImplMask, typename V>
    static void where_copy_to(const simd_mask_impl<ImplMask>& mask, simd_impl<Impl>& f, const V* t, unsigned width) {
        f.value_ = Impl::ifelse(mask.value_, Impl::copy_from_masked(t, mask.value_), f.value_);
    }

    template <typename Impl, typename ImplIndex, typename ImplMask, typename V>
    static void where_copy_to(const simd_mask_impl<ImplMask>& mask, simd_impl<Impl>& f,  const V* p, const simd_impl<ImplIndex>& index, unsigned width) {
        simd_impl<Impl> temp = Impl::broadcast(0);
        temp.value_ = Impl::gather(tag<ImplIndex>{}, temp.value_, p, index.value_, mask.value_);
        f.value_ = Impl::ifelse(mask.value_, temp.value_, f.value_);
    }

    /// Const Where Expressions
    template <typename Impl, typename ImplMask>
    class const_where_expression {
    public:
        const_where_expression(const ImplMask& m, const Impl& s):
                mask_(m), data_(s) {}

        template <typename T>             friend struct simd_impl;
        template <typename To>            friend struct simd_cast_impl;
        template <typename V>             friend class indirect_expression;
        template <typename I, typename V> friend class indirect_indexed_expression;

    private:
        const ImplMask& mask_;
        const Impl& data_;
    };

    template <typename Impl>
    struct simd_impl {
        static_assert(!std::is_void<Impl>::value, "no such SIMD ABI supported");

        // Type aliases:
        //
        //     scalar_type           internal value type in one simd lane,
        //     simd_mask             simd_mask_impl specialization represeting comparison results,
        //
        //     vector_type           underlying representation,
        //     mask_type             underlying representation for mask.

        using scalar_type = typename simd_traits<Impl>::scalar_type;
        using simd_mask   = simd_mask_impl<typename simd_traits<Impl>::mask_impl>;
        using simd_base = Impl;

    protected:
        using vector_type = typename simd_traits<Impl>::vector_type;
        using mask_type   = typename simd_traits<typename simd_traits<Impl>::mask_impl>::vector_type;

    public:
        static constexpr unsigned width = simd_traits<Impl>::width;
        static constexpr unsigned min_align = simd_traits<Impl>::min_align;

        template <typename Other>
        friend struct simd_impl;

        template <typename Other, typename V>
        friend class indirect_indexed_expression;

        template <typename V>
        friend class indirect_expression;

        simd_impl() = default;

        // Construct by filling with scalar value.
        simd_impl(const scalar_type& x) {
            value_ = Impl::broadcast(x);
        }

        // Construct from scalar values in memory.
        explicit simd_impl(const scalar_type* p) {
            value_ = Impl::copy_from(p);
        }

        // Construct from const array ref.
        explicit simd_impl(const scalar_type (&a)[width]) {
            value_ = Impl::copy_from(a);
        }

        // Construct from scalar values in memory with mask.
        explicit simd_impl(const scalar_type* p, const simd_mask& m) {
            value_ = Impl::copy_from_masked(p, m.value_);
        }

        // Construct from const array ref with mask.
        explicit simd_impl(const scalar_type (&a)[width], const simd_mask& m) {
            value_ = Impl::copy_from_masked(a, m.value_);
        }

        // Construct from a different SIMD value by casting.
        template <typename Other, typename = std::enable_if_t<width==simd_traits<Other>::width>>
        explicit simd_impl(const simd_impl<Other>& x) {
            value_ = Impl::cast_from(tag<Other>{}, x.value_);
        }

        // Construct from indirect expression (gather).
        template <typename IndexImpl, typename = std::enable_if_t<width==simd_traits<IndexImpl>::width>>
        explicit simd_impl(indirect_indexed_expression<IndexImpl, scalar_type> pi) {
            copy_from(pi);
        }

        template <typename IndexImpl, typename = std::enable_if_t<width==simd_traits<IndexImpl>::width>>
        explicit simd_impl(indirect_indexed_expression<IndexImpl, const scalar_type> pi) {
            copy_from(pi);
        }

        // Copy constructor.
        simd_impl(const simd_impl& other) {
            std::memcpy(&value_, &other.value_, sizeof(vector_type));
        }

        // Scalar asssignment fills vector.
        simd_impl& operator=(scalar_type x) {
            value_ = Impl::broadcast(x);
            return *this;
        }

        // Copy assignment.
        simd_impl& operator=(const simd_impl& other) {
            std::memcpy(&value_, &other.value_, sizeof(vector_type));
            return *this;
        }

        // Read/write operations: copy_to, copy_from.

        void copy_to(scalar_type* p) const {
            Impl::copy_to(value_, p);
        }

        template <typename Index, typename = std::enable_if_t<width==simd_traits<typename Index::simd_base>::width>>
        void copy_to(indirect_indexed_expression<Index, scalar_type> pi) const {
            using IndexImpl = typename Index::simd_base;
            Impl::scatter(tag<IndexImpl>{}, value_, pi.p, pi.index);
        }

        template <typename Other, typename = std::enable_if_t<width==simd_traits<Other>::width>>
        void copy_from(const simd_impl<Other>& x) {
            value_ = Impl::cast_from(tag<Other>{}, x.value_);
        }

        void copy_from(const scalar_type p) {
            value_ = Impl::broadcast(p);
        }

        void copy_from(const scalar_type* p) {
            value_ = Impl::copy_from(p);
        }


        template <typename Index, typename = std::enable_if_t<width==simd_traits<typename Index::simd_base>::width>>
        void copy_from(indirect_indexed_expression<Index, scalar_type> pi) {
            using IndexImpl = typename Index::simd_base;
            switch (pi.constraint) {
            case index_constraint::none:
                value_ = Impl::gather(tag<IndexImpl>{}, pi.p, pi.index.value_);
                break;
            case index_constraint::independent:
                value_ = Impl::gather(tag<IndexImpl>{}, pi.p, pi.index.value_);
                break;
            case index_constraint::contiguous:
                {
                    scalar_type* p = IndexImpl::element0(pi.index.value_) + pi.p;
                    value_ = Impl::copy_from(p);
                }
                break;
            case index_constraint::constant:
                {
                    scalar_type* p = IndexImpl::element0(pi.index.value_) + pi.p;
                    scalar_type l = (*p);
                    value_ = Impl::broadcast(l);
                }
                break;
            }
        }

        template <typename Index, typename = std::enable_if_t<width==simd_traits<typename Index::simd_base>::width>>
        void copy_from(indirect_indexed_expression<Index, const scalar_type> pi) {
            using IndexImpl = typename Index::simd_base;
            switch (pi.constraint) {
            case index_constraint::none:
                value_ = Impl::gather(tag<IndexImpl>{}, pi.p, pi.index.value_);
                break;
            case index_constraint::independent:
                value_ = Impl::gather(tag<IndexImpl>{}, pi.p, pi.index.value_);
                break;
            case index_constraint::contiguous:
                {
                    const scalar_type* p = IndexImpl::element0(pi.index.value_) + pi.p;
                    value_ = Impl::copy_from(p);
                }
                break;
            case index_constraint::constant:
                {
                    const scalar_type *p = IndexImpl::element0(pi.index.value_) + pi.p;
                    scalar_type l = (*p);
                    value_ = Impl::broadcast(l);
                }
                break;
            }
        }

        void copy_from(indirect_expression<scalar_type> pi) {
            value_ = Impl::copy_from(pi.p);
        }

        void copy_from(indirect_expression<const scalar_type> pi) {
            value_ = Impl::copy_from(pi.p);
        }

        template <typename T, typename M>
        void copy_from(const_where_expression<T, M> w) {
            value_ = Impl::ifelse(w.mask_.value_, w.data_.value_, value_);
        }

        template <typename T, typename M>
        void copy_from(where_expression<T, M> w) {
            value_ = Impl::ifelse(w.mask_.value_, w.data_.value_, value_);
        }

        // Arithmetic operations: +, -, *, /, fma.

        simd_impl operator-() const {
            return simd_impl::wrap(Impl::neg(value_));
        }

        friend simd_impl operator+(const simd_impl& a, simd_impl b) {
            return simd_impl::wrap(Impl::add(a.value_, b.value_));
        }

        friend simd_impl operator-(const simd_impl& a, simd_impl b) {
            return simd_impl::wrap(Impl::sub(a.value_, b.value_));
        }

        friend simd_impl operator*(const simd_impl& a, simd_impl b) {
            return simd_impl::wrap(Impl::mul(a.value_, b.value_));
        }

        friend simd_impl operator/(const simd_impl& a, simd_impl b) {
            return simd_impl::wrap(Impl::div(a.value_, b.value_));
        }

        friend simd_impl fma(const simd_impl& a, simd_impl b, simd_impl c) {
            return simd_impl::wrap(Impl::fma(a.value_, b.value_, c.value_));
        }

        // Lane-wise relational operations.

        friend simd_mask operator==(const simd_impl& a, const simd_impl& b) {
            return simd_impl::mask(Impl::cmp_eq(a.value_, b.value_));
        }

        friend simd_mask operator!=(const simd_impl& a, const simd_impl& b) {
            return simd_impl::mask(Impl::cmp_neq(a.value_, b.value_));
        }

        friend simd_mask operator<=(const simd_impl& a, const simd_impl& b) {
            return simd_impl::mask(Impl::cmp_leq(a.value_, b.value_));
        }

        friend simd_mask operator<(const simd_impl& a, const simd_impl& b) {
            return simd_impl::mask(Impl::cmp_lt(a.value_, b.value_));
        }

        friend simd_mask operator>=(const simd_impl& a, const simd_impl& b) {
            return simd_impl::mask(Impl::cmp_geq(a.value_, b.value_));
        }

        friend simd_mask operator>(const simd_impl& a, const simd_impl& b) {
            return simd_impl::mask(Impl::cmp_gt(a.value_, b.value_));
        }

        // Compound assignment operations: +=, -=, *=, /=.

        simd_impl& operator+=(const simd_impl& x) {
            value_ = Impl::add(value_, x.value_);
            return *this;
        }

        simd_impl& operator-=(const simd_impl& x) {
            value_ = Impl::sub(value_, x.value_);
            return *this;
        }

        simd_impl& operator*=(const simd_impl& x) {
            value_ = Impl::mul(value_, x.value_);
            return *this;
        }

        simd_impl& operator/=(const simd_impl& x) {
            value_ = Impl::div(value_, x.value_);
            return *this;
        }

        // Array subscript operations.

        struct reference {
            reference() = delete;
            reference(const reference&) = default;
            reference& operator=(const reference&) = delete;

            reference(vector_type& value, int i):
                ptr_(&value), i(i) {}

            reference& operator=(scalar_type v) && {
                Impl::set_element(*ptr_, i, v);
                return *this;
            }

            operator scalar_type() const {
                return Impl::element(*ptr_, i);
            }

            vector_type* ptr_;
            int i;
        };

        reference operator[](int i) {
            return reference(value_, i);
        }

        scalar_type operator[](int i) const {
            return Impl::element(value_, i);
        }

        // Reductions (horizontal operations).

        scalar_type sum() const {
            return Impl::reduce_add(value_);
        }

        // Maths functions are implemented as top-level functions; declare as friends for access to `wrap`

        #define ARB_DECLARE_UNARY_ARITHMETIC_(name)\
        template <typename T>\
        friend simd_impl<T> arb::simd::name(const simd_impl<T>& a);

        #define ARB_DECLARE_BINARY_ARITHMETIC_(name)\
        template <typename T>\
        friend simd_impl<T> arb::simd::name(const simd_impl<T>& a, simd_impl<T> b);\
        template <typename T>\
        friend simd_impl<T> arb::simd::name(const simd_impl<T>& a, typename simd_impl<T>::scalar_type b);\
        template <typename T>\
        friend simd_impl<T> arb::simd::name(const typename simd_impl<T>::scalar_type a, simd_impl<T> b);

        #define ARB_DECLARE_BINARY_COMPARISON_(name)\
        template <typename T>\
        friend typename simd_impl<T>::simd_mask arb::simd::name(const simd_impl<T>& a, simd_impl<T> b);\
        template <typename T>\
        friend typename simd_impl<T>::simd_mask arb::simd::name(const simd_impl<T>& a, typename simd_impl<T>::scalar_type b);\
        template <typename T>\
        friend typename simd_impl<T>::simd_mask arb::simd::name(const typename simd_impl<T>::scalar_type a, simd_impl<T> b);

        ARB_PP_FOREACH(ARB_DECLARE_BINARY_ARITHMETIC_, add, sub, mul, div, pow, max, min, cmp_eq)
        ARB_PP_FOREACH(ARB_DECLARE_BINARY_COMPARISON_, cmp_eq, cmp_neq, cmp_lt, cmp_leq, cmp_gt, cmp_geq)
        ARB_PP_FOREACH(ARB_DECLARE_UNARY_ARITHMETIC_,  neg, abs, sin, cos, exp, log, expm1, exprelr)

        #undef ARB_DECLARE_UNARY_ARITHMETIC_
        #undef ARB_DECLARE_BINARY_ARITHMETIC_
        #undef ARB_DECLARE_BINARY_COMPARISON_

        template <typename T>
        friend simd_impl<T> arb::simd::fma(const simd_impl<T> a, simd_impl<T> b, simd_impl<T> c);

        // Declare Indirect/Indirect indexed/Where Expression copy function as friends

        template <typename T, typename I, typename V>
        friend void compound_indexed_add(const simd_impl<I>& s, V* p, const simd_impl<T>& index, unsigned width, index_constraint constraint);

        template <typename I, typename V>
        friend void indirect_copy_to(const simd_impl<I>& s, V* p, unsigned width);

        template <typename T, typename M, typename V>
        friend void indirect_copy_to(const simd_impl<T>& data, const simd_mask_impl<M>& mask, V* p, unsigned width);

        template <typename T, typename I, typename V>
        friend void indirect_indexed_copy_to(const simd_impl<T>& s, V* p, const simd_impl<I>& index, unsigned width);

        template <typename T, typename I, typename M, typename V>
        friend void indirect_indexed_copy_to(const simd_impl<T>& data, const simd_mask_impl<M>& mask, V* p, const simd_impl<I>& index, unsigned width);

        template <typename T, typename M, typename V>
        friend void where_copy_to(const simd_mask_impl<M>& mask, simd_impl<T>& f, const V& t);

        template <typename T, typename M>
        friend void where_copy_to(const simd_mask_impl<M>& mask, simd_impl<T>& f, const simd_impl<T>& t);

        template <typename T, typename M, typename V>
        friend void where_copy_to(const simd_mask_impl<M>& mask, simd_impl<T>& f, const V* p, unsigned width);

        template <typename T, typename I, typename M, typename V>
        friend void where_copy_to(const simd_mask_impl<M>& mask, simd_impl<T>& f, const V* p, const simd_impl<I>& index, unsigned width);

    protected:
        vector_type value_;
        simd_impl(const vector_type& x) {
            value_ = x;
        }

    private:
        static simd_impl wrap(const vector_type& v) {
            simd_impl s;
            s.value_ = v;
            return s;
        }

        static simd_mask mask(const mask_type& v) {
            simd_mask m;
            m.value_ = v;
            return m;
        }
    };

    template <typename Impl>
    struct simd_mask_impl: simd_impl<Impl> {
        using base = simd_impl<Impl>;
        using typename base::vector_type;
        using typename base::scalar_type;
        using base::width;
        using base::value_;

        simd_mask_impl() = default;

        // Construct by filling with scalar value.
        simd_mask_impl(bool b) {
            value_ = Impl::mask_broadcast(b);
        }

        // Scalar asssignment fills vector.
        simd_mask_impl& operator=(bool b) {
            value_ = Impl::mask_broadcast(b);
            return *this;
        }

        // Construct from bool values in memory.
        explicit simd_mask_impl(const bool* y) {
            value_ = Impl::mask_copy_from(y);
        }

        // Construct from const array ref.
        explicit simd_mask_impl(const bool (&a)[width]) {
            value_ = Impl::mask_copy_from(&a[0]);
        }

        // Copy assignment.
        simd_mask_impl& operator=(const simd_mask_impl& other) {
            std::memcpy(&value_, &other.value_, sizeof(vector_type));
            return *this;
        }

        // Read/write bool operations: copy_to, copy_from.

        void copy_to(bool* w) const {
            Impl::mask_copy_to(value_, w);
        }

        void copy_from(const bool* y) {
            value_ = Impl::mask_copy_from(y);
        }

        void copy_from(indirect_expression<bool> pi) {
            value_ = Impl::mask_copy_from(pi.p);
        }

        // Array subscript operations.

        struct reference {
            reference() = delete;
            reference(const reference&) = default;
            reference& operator=(const reference&) = delete;

            reference(vector_type& value, int i):
                ptr_(&value), i(i) {}

            reference& operator=(bool v) && {
                Impl::mask_set_element(*ptr_, i, v);
                return *this;
            }

            operator bool() const {
                return Impl::mask_element(*ptr_, i);
            }

            vector_type* ptr_;
            int i;
        };

        reference operator[](int i) {
            return reference(value_, i);
        }

        bool operator[](int i) const {
            return Impl::element(value_, i);
        }

        // Logical operations.

        simd_mask_impl operator!() const {
            return simd_mask_impl::wrap(Impl::logical_not(value_));
        }

        friend simd_mask_impl operator&&(const simd_mask_impl& a, const simd_mask_impl& b) {
            return simd_mask_impl::wrap(Impl::logical_and(a.value_, b.value_));
        }

        friend simd_mask_impl operator||(const simd_mask_impl& a, const simd_mask_impl& b) {
            return simd_mask_impl::wrap(Impl::logical_or(a.value_, b.value_));
        }

        // Make mask from corresponding bits of integer.

        static simd_mask_impl unpack(unsigned long long bits) {
            return simd_mask_impl::wrap(Impl::mask_unpack(bits));
        }

    private:
        simd_mask_impl(const vector_type& v): base(v) {}

        template <class> friend struct simd_impl;

        static simd_mask_impl wrap(const vector_type& v) {
            simd_mask_impl m;
            m.value_ = v;
            return m;
        }
    };

    template <typename To>
    struct simd_cast_impl;

    template <typename ImplTo>
    struct simd_cast_impl<simd_mask_impl<ImplTo>> {
        static constexpr unsigned N = simd_traits<ImplTo>::width;
        using scalar_type = typename simd_traits<ImplTo>::scalar_type;

        template <typename ImplFrom, typename = std::enable_if_t<N==simd_traits<ImplFrom>::width>>
        static simd_mask_impl<ImplTo> cast(const simd_mask_impl<ImplFrom>& v) {
            return simd_mask_impl<ImplTo>(v);
        }

        static simd_mask_impl<ImplTo> cast(const std::array<scalar_type, N>& a) {
            return simd_mask_impl<ImplTo>(a.data());
        }

        static simd_mask_impl<ImplTo> cast(scalar_type a) {
            simd_mask_impl<ImplTo> r = a;
            return r;
        }

        static simd_mask_impl<ImplTo> cast(const indirect_expression<bool>& a) {
            simd_mask_impl<ImplTo> r;
            r.copy_from(a);
            return r;
        }
    };

    template <typename ImplTo>
    struct simd_cast_impl<simd_impl<ImplTo>> {
        static constexpr unsigned N = simd_traits<ImplTo>::width;
        using scalar_type = typename simd_traits<ImplTo>::scalar_type;

        template <typename ImplFrom, typename = std::enable_if_t<N==simd_traits<ImplFrom>::width>>
        static simd_impl<ImplTo> cast(const simd_impl<ImplFrom>& v) {
            return simd_impl<ImplTo>(v);
        }

        static simd_impl<ImplTo> cast(const std::array<scalar_type, N>& a) {
            return simd_impl<ImplTo>(a.data());
        }

        static simd_impl<ImplTo> cast(scalar_type a) {
            simd_impl<ImplTo> r = a;
            return r;
        }

        template <typename V>
        static simd_impl<ImplTo> cast(const indirect_expression<V>& a) {
            simd_impl<ImplTo> r;
            r.copy_from(a);
            return r;
        }

        template <typename Impl, typename V>
        static simd_impl<ImplTo> cast(const indirect_indexed_expression<Impl,V>& a) {
            simd_impl<ImplTo> r;
            r.copy_from(a);
            return r;
        }

        template <typename Impl, typename V>
        static simd_impl<ImplTo> cast(const const_where_expression<Impl,V>& a) {
            simd_impl<ImplTo> r = 0;
            r.copy_from(a);
            return r;
        }
    };

    template <typename V, std::size_t N>
    struct simd_cast_impl<std::array<V, N>> {
        template <
            typename ImplFrom,
            typename = std::enable_if_t<
                N==simd_traits<ImplFrom>::width &&
                std::is_same<V, typename simd_traits<ImplFrom>::scalar_type>::value
            >
        >
        static std::array<V, N> cast(const simd_impl<ImplFrom>& s) {
            std::array<V, N> a;
            s.copy_to(a.data());
            return a;
        }
    };
} // namespace detail

namespace simd_abi {
    // Note: `simd_abi::native` template class defined in `simd/native.hpp`,
    // `simd_abi::generic` in `simd/generic.hpp`.

    template <typename Value, unsigned N>
    struct default_abi {
        using type = typename std::conditional<
            std::is_same<void, typename native<Value, N>::type>::value,
            typename generic<Value, N>::type,
            typename native<Value, N>::type>::type;
    };
}

template <typename Value, unsigned N, template <class, unsigned> class Abi>
struct simd_wrap { using type = detail::simd_impl<typename Abi<Value, N>::type>; };

template <typename Value, unsigned N, template <class, unsigned> class Abi>
using simd = typename simd_wrap<Value, N, Abi>::type;

template <typename Value, unsigned N, template <class, unsigned> class Abi>
struct simd_mask_wrap { using type = typename simd<Value, N, Abi>::simd_mask; };

template <typename Value, unsigned N, template <class, unsigned> class Abi>
using simd_mask = typename simd_mask_wrap<Value, N, Abi>::type;

template <typename>
struct is_simd: std::false_type {};

template <typename Impl>
struct is_simd<detail::simd_impl<Impl>>: std::true_type {};

// Casting is dispatched to simd_cast_impl in order to handle conversions to
// and from std::array.

template <typename To, typename From>
To simd_cast(const From& s) {
    return detail::simd_cast_impl<To>::cast(s);
}

template <typename S, std::enable_if_t<is_simd<S>::value, int> = 0>
inline constexpr int width(const S a = S{}) {
    return S::width;
};

template <typename S, std::enable_if_t<is_simd<S>::value, int> = 0>
inline constexpr unsigned min_align(const S a = S{}) {
    return S::min_align;
};


// Gather/scatter indexed memory specification.

template <
    typename IndexImpl,
    typename PtrLike,
    typename V = std::remove_reference_t<decltype(*std::declval<PtrLike>())>
>
detail::indirect_indexed_expression<IndexImpl, V> indirect(
    PtrLike p,
    const IndexImpl& index,
    unsigned width,
    index_constraint constraint = index_constraint::none)
{
    return detail::indirect_indexed_expression<IndexImpl, V>(p, index, width, constraint);
}

template <
        typename PtrLike,
        typename V = std::remove_reference_t<decltype(*std::declval<PtrLike>())>
>
detail::indirect_expression<V> indirect(
        PtrLike p,
        unsigned width)
{
    return detail::indirect_expression<V>(p, width);
}

template <typename Impl, typename ImplMask>
detail::where_expression<Impl, ImplMask> where(const ImplMask& m, Impl& v) {
    return detail::where_expression<Impl, ImplMask>(m, v);
}

template <typename Impl, typename ImplMask>
detail::const_where_expression<Impl, ImplMask> where(const ImplMask& m, const Impl& v) {
    return detail::const_where_expression<Impl, ImplMask>(m, v);
}

} // namespace simd
} // namespace arb
