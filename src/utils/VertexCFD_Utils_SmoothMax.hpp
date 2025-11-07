#ifndef VERTEXCFD_UTILS_SMOOTHMAX_HPP
#define VERTEXCFD_UTILS_SMOOTHMAX_HPP

#include <Sacado.hpp>
#include <Sacado_Fad_Exp_Expression.hpp>
#include <Sacado_Fad_Exp_ExpressionTraits.hpp>
#include <Sacado_mpl_disable_if.hpp>
#include <Sacado_mpl_enable_if.hpp>

namespace VertexCFD
{
namespace SmoothMath
{
//---------------------------------------------------------------------------//
// Implementation of smooth "max" using Sacado expressions.
// The third template parameter specifies whether the second parameter is a
// constant (non-FAD). The "max" function is commutative, so we do not need
// separate implementations of the expression class when T1 is FAD and T2 is
// constant AND when T1 is constant and T2 is FAD. If exactly one parameter is
// a FAD, it should be the first parameter. The reverse ordering (first
// parameter is constant) is handled within the specializations of the "max"
// function below.
//---------------------------------------------------------------------------//
template<typename T1, typename T2, bool is_const_T2, typename ExprSpec>
class SmoothMaxOp
{
};

// Specialization for two Sacado expressions
template<typename T1, typename T2>
class SmoothMaxOp<T1, T2, false, Sacado::Fad::Exp::ExprSpecDefault>
    : public Expr<SmoothMaxOp<T1, T2, false, Sacado::Fad::Exp::ExprSpecDefault>>
{
  public:
    using ExprT1 = typename std::remove_cv<T1>::type;
    using ExprT2 = typename std::remove_cv<T2>::type;
    using value_type1 = typename ExprT1::value_type;
    using value_type2 = typename ExprT2::value_type;
    using value_type = typename Sacado::Promote<value_type1, value_type2>::type;
    using scalar_type1 = typename ExprT1::scalar_type;
    using scalar_type2 = typename ExprT2::scalar_type;
    using scalar_type =
        typename Sacado::Promote<scalar_type1, scalar_type2>::type;
    using expr_spec_type = Sacado::Fad::Exp::ExprSpecDefault;

    SACADO_INLINE_FUNCTION explicit SmoothMaxOp(const T1& x,
                                                const T2& y,
                                                double tol)
        : x_(x)
        , y_(y)
        , tol_(tol)
    {
    }

    SACADO_INLINE_FUNCTION int size() const
    {
        const int sz1 = x_.size();
        const int sz2 = y_.size();
        return sz1 > sz2 ? sz1 : sz2;
    }

    SACADO_INLINE_FUNCTION bool hasFastAccess() const
    {
        return x_.hasFastAccess() && y_.hasFastAccess();
    }

    SACADO_INLINE_FUNCTION value_type val() const
    {
        if (x_ <= y_ - tol_)
        {
            return y_.val();
        }
        else if (x_ >= y_ + tol_)
        {
            return x_.val();
        }
        else
        {
            return 0.5
                   * (x_.val() + y_.val()
                      + 0.5
                            * ((x_.val() - y_.val()) * (x_.val() - y_.val())
                                   / tol_
                               + tol_));
        }
    }

    SACADO_INLINE_FUNCTION value_type dx(int i) const
    {
        if (x_ <= y_ - tol_)
        {
            return y_.dx(i);
        }
        else if (x_ >= y_ + tol_)
        {
            return x_.dx(i);
        }
        else
        {
            return 0.5
                   * (x_.dx(i) + y_.dx(i)
                      + (x_.val() - y_.val()) * (x_.dx(i) - y_.dx(i)) / tol_);
        }
    }

    SACADO_INLINE_FUNCTION value_type fastAccessDx(int i) const
    {
        if (x_ <= y_ - tol_)
        {
            return y_.fastAccessDx(i);
        }
        else if (x_ >= y_ + tol_)
        {
            return x_.fastAccessDx(i);
        }
        else
        {
            return 0.5
                   * (x_.fastAccessDx(i) + y_.fastAccessDx(i)
                      + (x_.val() - y_.val())
                            * (x_.fastAccessDx(i) - y_.fastAccessDx(i)) / tol_);
        }
    }

  private:
    const T1& x_;
    const T2& y_;
    double tol_;
};

// Specialization for Sacado expression, constant
template<typename T1, typename T2>
class SmoothMaxOp<T1, T2, true, Sacado::Fad::Exp::ExprSpecDefault>
    : public Expr<SmoothMaxOp<T1, T2, true, Sacado::Fad::Exp::ExprSpecDefault>>
{
  public:
    using ExprT1 = typename std::remove_cv<T1>::type;
    using ConstT = T2;
    using value_type = typename ExprT1::value_type;
    using scalar_type = typename ExprT1::scalar_type;
    using expr_spec_type = Sacado::Fad::Exp::ExprSpecDefault;

    SACADO_INLINE_FUNCTION explicit SmoothMaxOp(const T1& x,
                                                const ConstT& c,
                                                double tol)
        : x_(x)
        , c_(c)
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
        if (x_ <= c_ - tol_)
        {
            return c_;
        }
        else if (x_ >= c_ + tol_)
        {
            return x_.val();
        }
        else
        {
            return 0.5
                   * (x_.val()
                      + 0.5 * ((x_.val() - c_) * (x_.val() - c_) / tol_ + tol_));
        }
    }

    SACADO_INLINE_FUNCTION value_type dx(int i) const
    {
        if (x_ <= c_ - tol_)
        {
            return 0.0;
        }
        else if (x_ >= c_ + tol_)
        {
            return x_.dx(i);
        }
        else
        {
            return 0.5 * (x_.dx(i) + (x_.val() - c_) * x_.dx(i) / tol_);
        }
    }

    SACADO_INLINE_FUNCTION value_type fastAccessDx(int i) const
    {
        if (x_ <= c_ - tol_)
        {
            return 0.0;
        }
        else if (x_ >= c_ + tol_)
        {
            return x_.fastAccessDx(i);
        }
        else
        {
            return 0.5
                   * (x_.fastAccessDx(i)
                      + (x_.val() - c_) * x_.fastAccessDx(i) / tol_);
        }
    }

  private:
    const T1& x_;
    const ConstT& c_;
    double tol_;
};

// Generic implementation for POD types
template<typename T>
SACADO_INLINE_FUNCTION
    typename std::enable_if<std::is_trivial<T>::value, T>::type
    max(const T x, const T y, const double tol)
{
    if (x <= y - tol)
    {
        return y;
    }
    else if (x >= y + tol)
    {
        return x;
    }
    else
    {
        return 0.5 * (x + y + 0.5 * ((x - y) * (x - y) / tol + tol));
    }
}

using Sacado::Fad::Exp::Expr;
using Sacado::Fad::Exp::ExprLevel;
using Sacado::Fad::Exp::IsFadExpr;

// Implementation when both arguments are FAD expressions
// See Sacado_SFINAE_Macros.hpp: SACADO_FAD_EXP_OP_ENABLE_EXPR_EXPR
template<typename T1, typename T2>
SACADO_INLINE_FUNCTION typename Sacado::mpl::enable_if_c<
    IsFadExpr<T1>::value && IsFadExpr<T2>::value
        && ExprLevel<T1>::value == ExprLevel<T2>::value,
    SmoothMaxOp<typename Expr<T1>::derived_type,
                typename Expr<T2>::derived_type,
                false,
                typename T1::expr_spec_type>>::type
max(const T1& x, const T2& y, const double tol)
{
    using expr_t = SmoothMaxOp<typename Expr<T1>::derived_type,
                               typename Expr<T2>::derived_type,
                               false,
                               typename T1::expr_spec_type>;
    return expr_t(x.derived(), y.derived(), tol);
}

// Specialization when first argument is constant, second is FAD,
// and the FAD value_type and scalar_type match
template<typename T>
SACADO_INLINE_FUNCTION SmoothMaxOp<typename Expr<T>::derived_type,
                                   typename T::value_type,
                                   true,
                                   typename T::expr_spec_type>
max(const typename T::value_type& c, const Expr<T>& y, const double tol)
{
    using ConstT = typename T::value_type;
    using expr_t = SmoothMaxOp<typename Expr<T>::derived_type,
                               ConstT,
                               true,
                               typename T::expr_spec_type>;

    // Reverse the order of the arguments
    return expr_t(y.derived(), c, tol);
}

// Specialization when first argument is FAD, second is constant,
// and the FAD value_type and scalar_type match
template<typename T>
SACADO_INLINE_FUNCTION SmoothMaxOp<typename Expr<T>::derived_type,
                                   typename T::value_type,
                                   true,
                                   typename T::expr_spec_type>
max(const Expr<T>& x, const typename T::value_type& c, const double tol)
{
    using ConstT = typename T::value_type;
    using expr_t = SmoothMaxOp<typename Expr<T>::derived_type,
                               ConstT,
                               true,
                               typename T::expr_spec_type>;
    return expr_t(x.derived(), c, tol);
}

// Specialization when first argument is constant, second is FAD,
// and the FAD value_type and scalar_type do NOT match
// See Sacado_SFINAE_Macros.hpp: SACADO_FAD_EXP_OP_ENABLE_SCALAR_EXPR
template<typename T>
SACADO_INLINE_FUNCTION typename Sacado::mpl::disable_if<
    std::is_same<typename T::value_type, typename T::scalar_type>,
    SmoothMaxOp<typename Expr<T>::derived_type,
                typename T::scalar_type,
                true,
                typename T::expr_spec_type>>::type
max(const typename T::scalar_type& c, const Expr<T>& y, const double tol)
{
    using ConstT = typename T::scalar_type;
    using expr_t = SmoothMaxOp<typename Expr<T>::derived_type,
                               ConstT,
                               true,
                               typename T::expr_spec_type>;

    // Reverse the order of the arguments
    return expr_t(y.derived(), c, tol);
}

// Specialization when first argument is FAD, second is constant,
// and the FAD value_type and scalar_type do NOT match
// See Sacado_SFINAE_Macros.hpp: SACADO_FAD_EXP_OP_ENABLE_EXPR_SCALAR
template<typename T>
SACADO_INLINE_FUNCTION typename Sacado::mpl::disable_if<
    std::is_same<typename T::value_type, typename T::scalar_type>,
    SmoothMaxOp<typename Expr<T>::derived_type,
                typename T::scalar_type,
                true,
                typename T::expr_spec_type>>::type
max(const Expr<T>& x, const typename T::scalar_type& c, const double tol)
{
    using ConstT = typename T::scalar_type;
    using expr_t = SmoothMaxOp<typename Expr<T>::derived_type,
                               ConstT,
                               true,
                               typename T::expr_spec_type>;
    return expr_t(x.derived(), c, tol);
}

} // namespace SmoothMath
} // namespace VertexCFD

namespace Sacado
{

using VertexCFD::SmoothMath::SmoothMaxOp;

namespace Fad
{
namespace Exp
{

template<typename T1, typename T2, bool c2, typename E>
struct ExprLevel<SmoothMaxOp<T1, T2, c2, E>>
{
    static constexpr unsigned value_1 = ExprLevel<T1>::value;
    static constexpr unsigned value_2 = ExprLevel<T2>::value;
    static constexpr unsigned value = value_1 >= value_2 ? value_1 : value_2;
};

template<typename T1, typename T2, bool c2, typename E>
struct IsFadExpr<SmoothMaxOp<T1, T2, c2, E>>
{
    static const unsigned value = true;
};

} // namespace Exp
} // namespace Fad

template<typename T1, typename T2, bool c2, typename E>
struct IsExpr<VertexCFD::SmoothMath::SmoothMaxOp<T1, T2, c2, E>>
{
    static const unsigned value = true;
};

template<typename T1, typename T2, bool c2, typename E>
struct BaseExprType<VertexCFD::SmoothMath::SmoothMaxOp<T1, T2, c2, E>>
{
    using base_expr_1 = typename BaseExprType<T1>::type;
    using base_expr_2 = typename BaseExprType<T2>::type;
    using type = typename Promote<base_expr_1, base_expr_2>::type;
};

template<typename T1, typename T2, bool c2, typename E>
struct IsSimdType<VertexCFD::SmoothMath::SmoothMaxOp<T1, T2, c2, E>>
{
    static const bool value
        = IsSimdType<typename VertexCFD::SmoothMath::
                         SmoothMaxOp<T1, T2, c2, E>::scalar_type>::value;
};

template<typename T1, typename T2, bool c2, typename E>
struct ValueType<VertexCFD::SmoothMath::SmoothMaxOp<T1, T2, c2, E>>
{
    using type =
        typename VertexCFD::SmoothMath::SmoothMaxOp<T1, T2, c2, E>::value_type;
};

template<typename T1, typename T2, bool c2, typename E>
struct ScalarType<VertexCFD::SmoothMath::SmoothMaxOp<T1, T2, c2, E>>
{
    using type =
        typename VertexCFD::SmoothMath::SmoothMaxOp<T1, T2, c2, E>::scalar_type;
};

} // namespace Sacado

#endif // end VERTEXCFD_UTILS_SMOOTHMAX_HPP
