#pragma once

#include <type_traits>

#include <simd/generic.hpp>

namespace arb {

namespace simd_detail {
    // Specialize `native` for each implemented architecture.

    template <typename Value, unsigned N>
    struct native {
        using type = void;
    };

    // Architecure-specific implementation type I requires the specification of
    // the following interface, where 'a', 'b', etc. denote values of
    // `scalar_type`, and 'u', 'v', 'w', etc. denote values of `vector_type`.
    //
    // Logical operations, bool constructors and bool conversions need only be
    // provided for implementations that are used to proivde mask_type
    // operations (marked with [*] below).
    //
    // Types:
    //
    // I::vector_type                     Underlying SIMD representation type.
    // I::scalar_type                     Value type in one lane of vector_type.
    // I::mask_impl                       Implementation type for comparison results.
    // I::mask_type                       SIMD representation type for comparison results.
    //                                    (equivalent to I::mask_impl::vector_type)
    //
    // Reflection:
    //
    // constexpr static unsigned width()
    //
    // Construction:
    //
    // vector_type I::broadcast(a)        Fill SIMD type with scalar a.
    // vector_type I::broadcast(bool x)   Fill SIMD type with I::from_bool(x). [*]
    // vector_type I::immediate(a,b,...)  Populate SIMD type with given values.
    // vector_type I::immediate(bool...)  Populate SIMD type with representations of given booleans. [*]
    // void I::copy_to(v, scalar_type*)   Store v to memory (unaligned).
    // vector_type I::copy_from(const scalar_type*)  Load from memory (unaligned).
    //
    // Conversion:
    //
    // I::is_convertible<V>::value        True if I::convert(V) defined.
    // vector_type I::convert(V)          Convert from SIMD type V to vector_type.
    //
    // Element (lane) access:
    //
    // scalar_type I::element(u, i)       Value in ith lane of u.
    // scalar_type I::bool_element(u, i)  Boolean value in ith lane of u. [*]
    // void I::set_element(u, i, a)       Set u[i] to a.
    // void I::set_element(u, i, bool f)  Set u[i] to representation of bool f. [*]
    //
    // (Note indexing should be such that `copy_to(x, p), p[i]` should
    // have the same value as `x[i]`.)
    //
    // Arithmetic:
    //
    // vector_type I::mul(u, v)           Lane-wise multiplication.
    // vector_type I::add(u, v)           Lane-wise addition.
    // vector_type I::sub(u, v)           Lane-wise subtraction.
    // vector_type I::div(u, v)           Lane-wise division.
    // vector_type I::fma(u, v, w)        Lane-wise fused multiply-add (u*v+w).
    // (TODO: add unary minus; add bitwise operations if there is utility)
    //
    // Comparison:
    //
    // mask_type I::cmp_eq(u, v)          Lane-wise equality.
    // mask_type I::cmp_not_eq(u, v)      Lane-wise negated equality.
    // (TODO: extend)
    //
    // Logical operations:
    //
    // vector_type I::logical_not(u)      Lane-wise negation. [*]
    // vector_type I::logical_and(u, v)   Lane-wise logical and. [*]
    // vector_type I::logical_or(u, v)    Lane-wise logical or. [*]
    //
    // Mask operations:
    //
    // vector_type I::select(mask_type m, u, v)  Lane-wise m? v: u.


    template <typename Impl>
    struct simd_mask_impl;

    template <typename Impl>
    struct simd_impl {
        // Type aliases:
        //
        //     vector_type           underlying representation,
        //     scalar_type           internal value type in one simd lane,
        //     simd_mask             simd_mask_impl specialization represeting comparison results.

        using vector_type = typename Impl::vector_type;
        using scalar_type = typename Impl::scalar_type;
        using simd_mask   = simd_mask_impl<typename Impl::mask_impl>;

        static constexpr unsigned width() { return Impl::width(); }

        // Generic constructors:
        // 
        // 1. From underlying representation type.
        // 2. Fill from scalar_type.
        // 3. Copy from pointer to scalar_type (explicit).
        // 4. Immediate from scalar argument values.
        // 5. Conversions from other SIMD value types of the same width,
        //    with or without support from implementation class.

        simd_impl() = default;

        // Construct from underlying representation type.
        simd_impl(const vector_type& v) {
            std::memcpy(&value_, &v, sizeof(vector_type));
        }

        // Construct by filling with scalar value.
        simd_impl(const scalar_type& x) {
            value_ = Impl::broadcast(x);
        }

        // Construct from scalar values in memory.
        explicit simd_impl(const scalar_type* p) {
            value_ = Impl::copy_from(p);
        }

        // Construct from values provided for each simd lane.
        template <typename... Args,
            typename = typename std::enable_if<1+sizeof...(Args)==Impl::width()>::type>
        simd_impl(scalar_type v0, Args... rest) {
            value_ = Impl::immediate(v0, static_cast<scalar_type>(rest)...);
        }

        // Copy constructor.
        simd_impl(const simd_impl& other) {
            std::memcpy(&value_, &other.value_, sizeof(vector_type));
        }

        // Converting constructors.

        template <typename X,
            typename = typename std::enable_if<
                !std::is_same<X, scalar_type>::value &&
                Impl::width()==X::width()
            >::type>
        explicit simd_impl(const simd_impl<X>& other):
            simd_impl(other, typename Impl::template is_convertible<typename X::vector_type>::type{})
        {}

        // Scalar asssignment fills vector.
        simd_impl& operator=(scalar_type x) {
            value_ = Impl::broadcast(x);
            return *this;
        }

        // Conversion assignment.
        template <typename X,
            typename = typename std::enable_if<
                !std::is_same<X, scalar_type>::value &&
                Impl::width()==X::width()
            >::type>
        simd_impl& operator=(const simd_impl<X>& other) {
            std::memcpy(&value_, &simd_impl(other).value_, sizeof(vector_type));
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

        void copy_from(const scalar_type* p) {
            value_ = Impl::copy_from(p);
        }

        // Arithmetic operations: +, -, *, /, fma.

        friend simd_impl operator+(const simd_impl& a, simd_impl b) {
            return Impl::add(a.value_, b.value_);
        }

        friend simd_impl operator-(const simd_impl& a, simd_impl b) {
            return Impl::sub(a.value_, b.value_);
        }

        friend simd_impl operator*(const simd_impl& a, simd_impl b) {
            return Impl::mul(a.value_, b.value_);
        }

        friend simd_impl operator/(const simd_impl& a, simd_impl b) {
            return Impl::div(a.value_, b.value_);
        }

        friend simd_impl fma(const simd_impl& a, simd_impl b, simd_impl c) {
            return Impl::fma(a.value_, b.value_, c.value_);
        }

        // Lane-wise relational operations: ==, !=, (TODO: rest to come).

        friend simd_mask operator==(const simd_impl& a, const simd_impl& b) {
            return Impl::cmp_eq(a.value_, b.value_);
        }

        friend simd_mask operator!=(const simd_impl& a, const simd_impl& b) {
            return Impl::cmp_not_eq(a.value_, b.value_);
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

        // Masked assignment (via where expressions).

        struct where_expression {
            where_expression(const where_expression&) = default;
            where_expression& operator=(const where_expression&) = delete;

            where_expression(const simd_mask& m, simd_impl& v):
                mask_(m), data_(v) {}

            where_expression& operator=(scalar_type v) {
                data_ = Impl::select(mask_.value_, data_.value_, simd_impl(v).value_);
                return *this;
            }

            where_expression& operator=(const simd_impl& v) {
                data_ = Impl::select(mask_.value_, data_.value_, v.value_);
                return *this;
            }

        private:
            const simd_mask& mask_;
            simd_impl& data_;
        };

    protected:
        vector_type value_;

        template <typename X>
        simd_impl(const simd_impl<X>& other, std::true_type) {
            value_ = Impl::convert(other.value_);
        }

        template <typename X>
        simd_impl(const simd_impl<X>& other, std::false_type) {
            typename simd_impl<X>::scalar_type from[width()];
            scalar_type to[width()];

            other.copy_to(from);
            for (unsigned i = 0; i<width(); ++i) {
                to[i] = static_cast<scalar_type>(from[i]);
            }
            copy_from(to);
        }
    };

    template <typename Impl>
    struct simd_mask_impl: simd_impl<Impl> {
        using base = simd_impl<Impl>;
        using typename base::vector_type;
        using typename base::scalar_type;
        using base::value_;

        simd_mask_impl() = default;

        simd_mask_impl(const vector_type& v): base(v) {}

        // Construct by filling with scalar value.
        simd_mask_impl(bool x) {
            value_ = Impl::broadcast(x);
        }

        // Construct from values provided for each simd lane.
        template <typename... Args,
            typename = typename std::enable_if<1+sizeof...(Args)==Impl::width()>::type>
        simd_mask_impl(bool v0, Args... rest) {
            value_ = Impl::immediate(v0, static_cast<bool>(rest)...);
        }

        // Scalar asssignment fills vector.
        simd_mask_impl& operator=(bool x) {
            value_ = Impl::broadcast(x);
            return *this;
        }

        // Copy assignment.
        simd_mask_impl& operator=(const simd_mask_impl& other) {
            std::memcpy(&value_, &other.value_, sizeof(vector_type));
            return *this;
        }

        // Array subscript operations.

        struct reference {
            reference() = delete;
            reference(const reference&) = default;
            reference& operator=(const reference&) = delete;

            reference(vector_type& value, int i):
                ptr_(&value), i(i) {}

            reference& operator=(bool v) && {
                Impl::set_element(*ptr_, i, v);
                return *this;
            }

            operator scalar_type() const {
                return Impl::bool_element(*ptr_, i);
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
            return Impl::logical_not(value_);
        }

        friend simd_mask_impl operator&&(const simd_mask_impl& a, const simd_mask_impl& b) {
            return Impl::logical_and(a.value_, b.value_);
        }

        friend simd_mask_impl operator||(const simd_mask_impl& a, const simd_mask_impl& b) {
            return Impl::logical_or(a.value_, b.value_);
        }
    };
} // namespace simd_detail

template <typename Scalar, unsigned N>
using simd =
    simd_detail::simd_impl<
        typename std::conditional<
            std::is_same<void, typename simd_detail::native<Scalar, N>::type>::value,
            simd_detail::generic<Scalar, N>,
            typename simd_detail::native<Scalar, N>::type
        >::type
    >;

template <typename Scalar, unsigned N>
using simd_mask = typename simd<Scalar, N>::simd_mask;

template <typename Simd>
using where_expression = typename Simd::where_expression;

template <typename Simd>
where_expression<Simd> where(const typename Simd::simd_mask& m, Simd& v) {
    return where_expression<Simd>(m, v);
}

} // namespace arb
