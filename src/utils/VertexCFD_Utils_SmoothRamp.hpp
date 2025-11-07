#ifndef VERTEXCFD_UTILS_SMOOTHRAMP_HPP
#define VERTEXCFD_UTILS_SMOOTHRAMP_HPP

#include "VertexCFD_Utils_Constants.hpp"

#include <Sacado.hpp>
#include <Sacado_Fad_Exp_Expression.hpp>
#include <Sacado_Fad_Exp_ExpressionTraits.hpp>

namespace VertexCFD
{
namespace SmoothMath
{
//---------------------------------------------------------------------------//
// Implementation of smooth "ramp" using Sacado expressions.
//---------------------------------------------------------------------------//
template<typename T, typename ExprSpec>
class SmoothRampOp
{
};

template<typename T>
class SmoothRampOp<T, Sacado::Fad::Exp::ExprSpecDefault>
    : public Sacado::Fad::Exp::Expr<
          SmoothRampOp<T, Sacado::Fad::Exp::ExprSpecDefault>>
{
  public:
    using ExprT = typename std::remove_cv<T>::type;
    using value_type = typename ExprT::value_type;
    using scalar_type = typename ExprT::scalar_type;
    using expr_spec_type = Sacado::Fad::Exp::ExprSpecDefault;

    SACADO_INLINE_FUNCTION explicit SmoothRampOp(const T& x,
                                                 double start,
                                                 double end)
        : x_(x)
        , start_(start)
        , end_(end)
    {
    }

    SACADO_INLINE_FUNCTION int size() const { return x_.size(); }

    SACADO_INLINE_FUNCTION bool hasFastAccess() const
    {
        return x_.hasFastAccess();
    }

    SACADO_INLINE_FUNCTION value_type val() const
    {
        using std::sin;
        constexpr double half_pi = 0.5 * Constants::pi;
        if (x_ <= start_)
        {
            return 0.0;
        }
        else if (x_ >= end_)
        {
            return 1.0;
        }
        else
        {
            return 0.5
                   * (sin(half_pi * (2.0 * x_.val() - (start_ + end_))
                          / (end_ - start_))
                      + 1.0);
        }
    }

    SACADO_INLINE_FUNCTION value_type dx(int i) const
    {
        using std::cos;
        constexpr double half_pi = 0.5 * Constants::pi;
        if (x_ <= start_)
        {
            return 0.0;
        }
        else if (x_ >= end_)
        {
            return 0.0;
        }
        else
        {
            return (half_pi / (end_ - start_))
                   * (cos(half_pi * (2.0 * x_.dx(i) - (start_ + end_))
                          / (end_ - start_)));
        }
    }

    SACADO_INLINE_FUNCTION value_type fastAccessDx(int i) const
    {
        using std::cos;
        constexpr double half_pi = 0.5 * Constants::pi;
        if (x_ <= start_)
        {
            return 0.0;
        }
        else if (x_ >= end_)
        {
            return 0.0;
        }
        else
        {
            return (half_pi / (end_ - start_))
                   * (cos(half_pi * (2.0 * x_.fastAccessDx(i) - (start_ + end_))
                          / (end_ - start_)));
        }
    }

  private:
    const T& x_;
    double start_;
    double end_;
};

// Generic implementation for POD types
template<typename T>
SACADO_INLINE_FUNCTION
    typename std::enable_if<std::is_trivial<T>::value, T>::type
    ramp(const T x, const double start, const double end)
{
    using std::sin;
    constexpr double half_pi = 0.5 * Constants::pi;
    if (x <= start)
    {
        return 0.0;
    }
    else if (x >= end)
    {
        return 1.0;
    }
    else
    {
        return 0.5
               * (sin(half_pi * (2.0 * x - (start + end)) / (end - start))
                  + 1.0);
    }
}

using Sacado::Fad::Exp::Expr;

// Specialization for FAD types
template<typename T>
SACADO_INLINE_FUNCTION
    SmoothRampOp<typename Expr<T>::derived_type, typename T::expr_spec_type>
    ramp(const Expr<T>& x, double start, double end)
{
    using expr_t = SmoothRampOp<typename Expr<T>::derived_type,
                                typename T::expr_spec_type>;
    return expr_t(x.derived(), start, end);
}

} // end namespace SmoothMath
} // end namespace VertexCFD

namespace Sacado
{

using VertexCFD::SmoothMath::SmoothRampOp;

namespace Fad
{
namespace Exp
{

//
// Specializations for Sacado traits
//

template<typename T, typename E>
struct ExprLevel<SmoothRampOp<T, E>>
{
    static const unsigned value = ExprLevel<T>::value;
};

template<typename T, typename E>
struct IsFadExpr<SmoothRampOp<T, E>>
{
    static const unsigned value = true;
};

} // namespace Exp
} // namespace Fad

template<typename T, typename E>
struct IsExpr<SmoothRampOp<T, E>>
{
    static const unsigned value = true;
};

template<typename T, typename E>
struct BaseExprType<SmoothRampOp<T, E>>
{
    using type = typename BaseExprType<T>::type;
};

template<typename T, typename E>
struct IsSimdType<SmoothRampOp<T, E>>
{
    static const bool value
        = IsSimdType<typename SmoothRampOp<T, E>::scalar_type>::value;
};

template<typename T, typename E>
struct ValueType<SmoothRampOp<T, E>>
{
    using type = typename SmoothRampOp<T, E>::value_type;
};

template<typename T, typename E>
struct ScalarType<SmoothRampOp<T, E>>
{
    using type = typename SmoothRampOp<T, E>::scalar_type;
};

} // namespace Sacado

#endif // end VERTEXCFD_UTILS_SMOOTHRAMP_HPP
