#ifndef VERTEXCFD_UTILS_SMOOTHABS_HPP
#define VERTEXCFD_UTILS_SMOOTHABS_HPP

#include "VertexCFD_Utils_Constants.hpp"
#include "VertexCFD_Utils_TypeTraits.hpp"

#include <Sacado.hpp>
#include <Sacado_Fad_Exp_Expression.hpp>
#include <Sacado_Fad_Exp_ExpressionTraits.hpp>

namespace VertexCFD
{
namespace SmoothMath
{
//---------------------------------------------------------------------------//
// Implementation of smooth "abs" using Sacado expressions.
//---------------------------------------------------------------------------//
template<typename T, typename ExprSpec>
class SmoothAbsOp
{
};

template<typename T>
class SmoothAbsOp<T, Sacado::Fad::Exp::ExprSpecDefault>
    : public Sacado::Fad::Exp::Expr<
          SmoothAbsOp<T, Sacado::Fad::Exp::ExprSpecDefault>>
{
  public:
    using ExprT = typename std::remove_cv<T>::type;
    using value_type = typename ExprT::value_type;
    using scalar_type = typename ExprT::scalar_type;
    using expr_spec_type = Sacado::Fad::Exp::ExprSpecDefault;

    SACADO_INLINE_FUNCTION explicit SmoothAbsOp(const T& x, double tol)
        : x_(x)
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
        if (x_ >= tol_)
        {
            return x_.val();
        }
        else if (x_ <= -tol_)
        {
            return -x_.val();
        }
        else
        {
            return 0.5 * (x_.val() * x_.val() / tol_ + tol_);
        }
    }

    SACADO_INLINE_FUNCTION value_type dx(int i) const
    {
        if (x_ >= tol_)
        {
            return x_.dx(i);
        }
        else if (x_ <= -tol_)
        {
            return value_type(-x_.dx(i));
        }
        else
        {
            return x_.dx(i) * x_.val() / tol_;
        }
    }

    SACADO_INLINE_FUNCTION value_type fastAccessDx(int i) const
    {
        if (x_ >= tol_)
        {
            return x_.fastAccessDx(i);
        }
        else if (x_ <= -tol_)
        {
            return value_type(-x_.fastAccessDx(i));
        }
        else
        {
            return x_.fastAccessDx(i) * x_.val() / tol_;
        }
    }

  private:
    const T& x_;
    double tol_;
};

// Generic implementation for POD types
template<typename T>
SACADO_INLINE_FUNCTION
    typename std::enable_if<std::is_trivial<T>::value, T>::type
    abs(const T x, const double tol)
{
    if (x >= tol)
    {
        return x;
    }
    else if (x <= -tol)
    {
        return -x;
    }
    else
    {
        return 0.5 * (x * x / tol + tol);
    }
}

using Sacado::Fad::Exp::Expr;

// Specialization for FAD expression
template<typename T>
SACADO_INLINE_FUNCTION
    SmoothAbsOp<typename Expr<T>::derived_type, typename T::expr_spec_type>
    abs(const Expr<T>& x, const double tol)
{
    using expr_t
        = SmoothAbsOp<typename Expr<T>::derived_type, typename T::expr_spec_type>;
    return expr_t(x.derived(), tol);
}

} // end namespace SmoothMath
} // end namespace VertexCFD

namespace Sacado
{

using VertexCFD::SmoothMath::SmoothAbsOp;

namespace Fad
{
namespace Exp
{

//
// Specializations for Sacado traits
//

template<typename T, typename E>
struct ExprLevel<SmoothAbsOp<T, E>>
{
    static const unsigned value = ExprLevel<T>::value;
};

template<typename T, typename E>
struct IsFadExpr<SmoothAbsOp<T, E>>
{
    static const unsigned value = true;
};

} // namespace Exp
} // namespace Fad

template<typename T, typename E>
struct IsExpr<SmoothAbsOp<T, E>>
{
    static const unsigned value = true;
};

template<typename T, typename E>
struct BaseExprType<SmoothAbsOp<T, E>>
{
    using type = typename BaseExprType<T>::type;
};

template<typename T, typename E>
struct IsSimdType<SmoothAbsOp<T, E>>
{
    static const bool value
        = IsSimdType<typename SmoothAbsOp<T, E>::scalar_type>::value;
};

template<typename T, typename E>
struct ValueType<SmoothAbsOp<T, E>>
{
    using type = typename SmoothAbsOp<T, E>::value_type;
};

template<typename T, typename E>
struct ScalarType<SmoothAbsOp<T, E>>
{
    using type = typename SmoothAbsOp<T, E>::scalar_type;
};

} // namespace Sacado

#endif // end VERTEXCFD_UTILS_SMOOTHABS_HPP
