#ifndef VERTEXCFD_UTILS_SMOOTHCLAMP_HPP
#define VERTEXCFD_UTILS_SMOOTHCLAMP_HPP

#include <Sacado.hpp>
#include <Sacado_Fad_Exp_Expression.hpp>
#include <Sacado_Fad_Exp_ExpressionTraits.hpp>

#include <type_traits>

namespace VertexCFD
{
namespace SmoothMath
{
//---------------------------------------------------------------------------//
// Implementation of smooth "clamp" using Sacado expressions
//---------------------------------------------------------------------------//
template<typename T, typename ExprSpec>
class SmoothClampOp
{
};

template<typename T>
class SmoothClampOp<T, Sacado::Fad::Exp::ExprSpecDefault>
    : public Sacado::Fad::Exp::Expr<
          SmoothClampOp<T, Sacado::Fad::Exp::ExprSpecDefault>>
{
  public:
    using ExprT = typename std::remove_cv<T>::type;
    using value_type = typename ExprT::value_type;
    using scalar_type = typename ExprT::scalar_type;
    using expr_spec_type = Sacado::Fad::Exp::ExprSpecDefault;

    SACADO_INLINE_FUNCTION explicit SmoothClampOp(const T& x,
                                                  double lo,
                                                  double hi,
                                                  double tol)
        : x_(x)
        , lo_(lo)
        , hi_(hi)
        , tol_(tol)
    {
    }

    SACADO_INLINE_FUNCTION int size() const { return x_.size(); }

    SACADO_INLINE_FUNCTION bool hasFastAccess() const
    {
        return x_.hasFastAccess();
    }

    SACADO_INLINE_FUNCTION value_type val() const
    {
        //  -inf < x < inf
        if (x_ <= lo_ - tol_)
        {
            return lo_;
        }

        // lo - tol < x < inf
        else if (x_ >= hi_ + tol_)
        {
            return hi_;
        }

        // lo - tol < x < hi + tol
        else if (x_ < lo_ + tol_)
        {
            //  lo - tol < x < lo + tol => -tol < (x-lo) < tol
            return 0.5
                   * (x_.val() + lo_
                      + 0.5
                            * ((x_.val() - lo_) * (x_.val() - lo_) / tol_
                               + tol_));
        }

        // lo + tol <= x < hi + tol
        else if (x_ > hi_ - tol_)
        {
            //  hi - tol < x < hi + tol => -tol < (x-hi) < tol
            return 0.5
                   * (x_.val() + hi_
                      - 0.5
                            * ((x_.val() - hi_) * (x_.val() - hi_) / tol_
                               + tol_));
        }

        // lo + tol <= x <= h - tol
        else
        {
            return x_.val();
        }
    }

    SACADO_INLINE_FUNCTION value_type dx(int i) const
    {
        // -inf < x < inf
        if (x_ <= lo_ - tol_)
        {
            return 0.0;
        }
        // lo - tol < x < inf
        else if (x_ >= hi_ + tol_)
        {
            return 0.0;
        }
        // lo - tol < x < hi + tol
        else if (x_ < lo_ + tol_)
        {
            // lo - tol < x < lo + tol => -tol < (x-lo) < tol
            return 0.5 * (x_.dx(i) + (x_.val() - lo_) * x_.dx(i) / tol_);
        }
        // lo + tol <= x < hi + tol
        else if (x_ > hi_ - tol_)
        {
            // hi - tol < x < hi + tol => -tol < (x-hi) < tol
            return 0.5 * (x_.dx(i) + (x_.val() - hi_) * x_.dx(i) / tol_);
        }
        // lo + tol <= x <= h - tol
        else
        {
            return x_.dx(i);
        }
    }

    SACADO_INLINE_FUNCTION value_type fastAccessDx(int i) const
    {
        // -inf < x < inf
        if (x_ <= lo_ - tol_)
        {
            return 0.0;
        }
        // lo - tol < x < inf
        else if (x_ >= hi_ + tol_)
        {
            return 0.0;
        }
        // lo - tol < x < hi + tol
        else if (x_ < lo_ + tol_)
        {
            // lo - tol < x < lo + tol => -tol < (x-lo) < tol
            return 0.5
                   * (x_.fastAccessDx(i)
                      + (x_.val() - lo_) * x_.fastAccessDx(i) / tol_);
        }
        // lo + tol <= x < hi + tol
        else if (x_ > hi_ - tol_)
        {
            // hi - tol < x < hi + tol => -tol < (x-hi) < tol
            return 0.5
                   * (x_.fastAccessDx(i)
                      + (x_.val() - hi_) * x_.fastAccessDx(i) / tol_);
        }
        // lo + tol <= x <= h - tol
        else
        {
            return x_.fastAccessDx(i);
        }
    }

  private:
    const T& x_;
    double lo_;
    double hi_;
    double tol_;
};

// Generic implementation for POD types
template<typename T>
SACADO_INLINE_FUNCTION
    typename std::enable_if<std::is_trivial<T>::value, T>::type
    clamp(const T x, const double lo, const double hi, const double tol)
{
    // -inf < x < inf
    if (x <= lo - tol)
    {
        return lo;
    }

    // lo - tol < x < inf
    else if (x >= hi + tol)
    {
        return hi;
    }

    // lo - tol < x < hi + tol
    else if (x < lo + tol)
    {
        // lo - tol < x < lo + tol => -tol < (x-lo) < tol
        return 0.5 * (x + lo + 0.5 * ((x - lo) * (x - lo) / tol + tol));
    }

    // lo + tol <= x < hi + tol
    else if (x > hi - tol)
    {
        // hi - tol < x < hi + tol => -tol < (x-hi) < tol
        return 0.5 * (x + hi - 0.5 * ((x - hi) * (x - hi) / tol + tol));
    }

    // lo + tol <= x <= h - tol
    else
    {
        return x;
    }
}

using Sacado::Fad::Exp::Expr;

// Specialization for FAD types
template<typename T>
SACADO_INLINE_FUNCTION
    SmoothClampOp<typename Expr<T>::derived_type, typename T::expr_spec_type>
    clamp(const Expr<T>& x, double lo, double hi, double tol)
{
    using expr_t = SmoothClampOp<typename Expr<T>::derived_type,
                                 typename T::expr_spec_type>;
    return expr_t(x.derived(), lo, hi, tol);
}

} // end namespace SmoothMath
} // end namespace VertexCFD

namespace Sacado
{

using VertexCFD::SmoothMath::SmoothClampOp;

namespace Fad
{
namespace Exp
{

//
// Specializations for Sacado traits
//

template<typename T, typename E>
struct ExprLevel<SmoothClampOp<T, E>>
{
    static const unsigned value = ExprLevel<T>::value;
};

template<typename T, typename E>
struct IsFadExpr<SmoothClampOp<T, E>>
{
    static const unsigned value = true;
};

} // namespace Exp
} // namespace Fad

template<typename T, typename E>
struct IsExpr<SmoothClampOp<T, E>>
{
    static const unsigned value = true;
};

template<typename T, typename E>
struct BaseExprType<SmoothClampOp<T, E>>
{
    using type = typename BaseExprType<T>::type;
};

template<typename T, typename E>
struct IsSimdType<SmoothClampOp<T, E>>
{
    static const bool value
        = IsSimdType<typename SmoothClampOp<T, E>::scalar_type>::value;
};

template<typename T, typename E>
struct ValueType<SmoothClampOp<T, E>>
{
    using type = typename SmoothClampOp<T, E>::value_type;
};

template<typename T, typename E>
struct ScalarType<SmoothClampOp<T, E>>
{
    using type = typename SmoothClampOp<T, E>::scalar_type;
};

} // namespace Sacado

#endif // end VERTEXCFD_UTILS_SMOOTHCLAMP_HPP
