#ifndef VERTEXCFD_UTILS_SMOOTHMATH_HPP
#define VERTEXCFD_UTILS_SMOOTHMATH_HPP

#include "VertexCFD_Utils_Constants.hpp"
#include "VertexCFD_Utils_SmoothAbs.hpp"
#include "VertexCFD_Utils_SmoothClamp.hpp"
#include "VertexCFD_Utils_SmoothMax.hpp"
#include "VertexCFD_Utils_SmoothMin.hpp"
#include "VertexCFD_Utils_SmoothRamp.hpp"
#include "VertexCFD_Utils_TypeTraits.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>

namespace VertexCFD
{
namespace SmoothMath
{
//---------------------------------------------------------------------------//
template<typename T1, typename T2>
KOKKOS_INLINE_FUNCTION ResultType<T1, T2>
hypot(const T1& x, const T2& y, const double tol)
{
    using std::sqrt;

    const ResultType<T1, T2> dotp = x * x + y * y;
    const double tol2 = tol * tol;

    if (tol2 <= dotp)
    {
        return sqrt(dotp);
    }
    else
    {
        return 0.5 * (dotp + tol2) / tol;
    }
}

//---------------------------------------------------------------------------//
template<typename T1, typename T2, typename T3>
KOKKOS_INLINE_FUNCTION ResultType<T1, T2, T3>
hypot(const T1& x, const T2& y, const T3& z, const double tol)
{
    using std::sqrt;

    const ResultType<T1, T2, T3> dotp = x * x + y * y + z * z;
    const double tol2 = tol * tol;

    if (tol2 <= dotp)
    {
        return sqrt(dotp);
    }
    else
    {
        return 0.5 * (dotp + tol2) / tol;
    }
}

template<typename T1, typename ReturnType = ResultType<typename T1::value_type>>
KOKKOS_INLINE_FUNCTION ReturnType norm(const T1& v, const double tol)
{
    using std::sqrt;

    using scalar_type = std::remove_cv_t<ReturnType>;
    scalar_type dotp = 0.0;
    const int num_space_dim = v.size();

    for (int i = 0; i < num_space_dim; ++i)
    {
        dotp += v[i] * v[i];
    }

    const double tol2 = tol * tol;
    if (tol2 <= dotp)
    {
        return sqrt(dotp);
    }
    else
    {
        return 0.5 * (dotp + tol2) / tol;
    }
}

//---------------------------------------------------------------------------//
template<typename T1,
         typename T2,
         typename ReturnType
         = ResultType<typename T1::value_type, typename T2::value_type>>
KOKKOS_INLINE_FUNCTION ReturnType norm(const T1& v,
                                       const T2& M,
                                       const double tol)
{
    using std::sqrt;

    using scalar_type = std::remove_cv_t<ReturnType>;
    scalar_type xi = 0.0;
    scalar_type row_sum;

    const int num_space_dim = M.extent(0);

    for (int i = 0; i < num_space_dim; ++i)
    {
        row_sum = 0.0;
        for (int j = 0; j < num_space_dim; ++j)
        {
            row_sum += M(i, j) * v[j];
        }
        xi += v[i] * row_sum;
    }

    const double tol2 = tol * tol;
    if (tol2 <= xi)
    {
        return sqrt(xi);
    }
    else
    {
        return 0.5 * (xi + tol2) / tol;
    }
}

//---------------------------------------------------------------------------//

} // end namespace SmoothMath
} // end namespace VertexCFD

#endif // end VERTEXCFD_UTILS_SMOOTHMATH_HPP
