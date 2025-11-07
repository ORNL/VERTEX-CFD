#ifndef VERTEXCFD_UTILS_NONLINEARSOLVER_HPP
#define VERTEXCFD_UTILS_NONLINEARSOLVER_HPP

#include <Sacado.hpp>

#include <Kokkos_Array.hpp>
#include <Kokkos_NumericTraits.hpp>

#include <cmath>
#include <type_traits>

namespace VertexCFD
{
namespace Utils
{
namespace NonlinearSolver
{
namespace Impl
{
//---------------------------------------------------------------------------//
// Access an element of the residual.
template<class Scalar, std::size_t N, int NumDeriv>
KOKKOS_INLINE_FUNCTION Scalar
f_i(const Kokkos::Array<Sacado::Fad::SFad<Scalar, NumDeriv>, N>& f_eval,
    const int i)
{
    static_assert(N == NumDeriv, "Jacobian must be square");
    return f_eval[i].val();
}

//---------------------------------------------------------------------------//
// Access an element of the Jacobian.
template<class Scalar, std::size_t N, int NumDeriv>
KOKKOS_INLINE_FUNCTION Scalar
J_ij(const Kokkos::Array<Sacado::Fad::SFad<Scalar, NumDeriv>, N>& f_eval,
     const int i,
     const int j)
{
    static_assert(N == NumDeriv, "Jacobian must be square");
    return f_eval[i].fastAccessDx(j);
}

//---------------------------------------------------------------------------//
// Jacobian determinant.
template<class Scalar>
KOKKOS_INLINE_FUNCTION Scalar
detJ(const Kokkos::Array<Sacado::Fad::SFad<Scalar, 1>, 1>& f_eval)
{
    return J_ij(f_eval, 0, 0);
}

//---------------------------------------------------------------------------//
// Jacobian determinant.
template<class Scalar>
KOKKOS_INLINE_FUNCTION Scalar
detJ(const Kokkos::Array<Sacado::Fad::SFad<Scalar, 2>, 2>& f_eval)
{
    return J_ij(f_eval, 0, 0) * J_ij(f_eval, 1, 1)
           - J_ij(f_eval, 0, 1) * J_ij(f_eval, 1, 0);
}

//---------------------------------------------------------------------------//
// Jacobian determinant.
template<class Scalar>
KOKKOS_INLINE_FUNCTION Scalar
detJ(const Kokkos::Array<Sacado::Fad::SFad<Scalar, 3>, 3>& f_eval)
{
    return J_ij(f_eval, 0, 0) * J_ij(f_eval, 1, 1) * J_ij(f_eval, 2, 2)
           + J_ij(f_eval, 0, 1) * J_ij(f_eval, 1, 2) * J_ij(f_eval, 2, 0)
           + J_ij(f_eval, 0, 2) * J_ij(f_eval, 1, 0) * J_ij(f_eval, 2, 1)
           - J_ij(f_eval, 0, 2) * J_ij(f_eval, 1, 1) * J_ij(f_eval, 2, 0)
           - J_ij(f_eval, 0, 1) * J_ij(f_eval, 1, 0) * J_ij(f_eval, 2, 2)
           - J_ij(f_eval, 0, 0) * J_ij(f_eval, 1, 2) * J_ij(f_eval, 2, 1);
}

//---------------------------------------------------------------------------//
// Jacobian inverse.
template<class Scalar>
KOKKOS_INLINE_FUNCTION bool
invJ(Kokkos::Array<Kokkos::Array<Scalar, 1>, 1>& J_inv,
     const Kokkos::Array<Sacado::Fad::SFad<Scalar, 1>, 1>& f_eval,
     const typename Sacado::ScalarType<Scalar>::type degen_j_tol)
{
    using std::abs;
    Scalar det_j = detJ(f_eval);
    if (abs(Sacado::ScalarValue<Scalar>::eval(det_j)) < degen_j_tol)
    {
        return false;
    }
    J_inv[0][0] = 1.0 / det_j;
    return true;
}

//---------------------------------------------------------------------------//
// Jacobian inverse.
template<class Scalar>
KOKKOS_INLINE_FUNCTION bool
invJ(Kokkos::Array<Kokkos::Array<Scalar, 2>, 2>& J_inv,
     const Kokkos::Array<Sacado::Fad::SFad<Scalar, 2>, 2>& f_eval,
     const typename Sacado::ScalarType<Scalar>::type degen_j_tol)
{
    using std::abs;
    Scalar det_j = detJ(f_eval);
    if (abs(Sacado::ScalarValue<Scalar>::eval(det_j)) < degen_j_tol)
    {
        return false;
    }
    Scalar det_j_inv = 1.0 / det_j;
    J_inv[0][0] = J_ij(f_eval, 1, 1) * det_j_inv;
    J_inv[0][1] = -J_ij(f_eval, 0, 1) * det_j_inv;
    J_inv[1][0] = -J_ij(f_eval, 1, 0) * det_j_inv;
    J_inv[1][1] = J_ij(f_eval, 0, 0) * det_j_inv;
    return true;
}

//---------------------------------------------------------------------------//
// Jacobian inverse.
template<class Scalar>
KOKKOS_INLINE_FUNCTION bool
invJ(Kokkos::Array<Kokkos::Array<Scalar, 3>, 3>& J_inv,
     const Kokkos::Array<Sacado::Fad::SFad<Scalar, 3>, 3>& f_eval,
     const typename Sacado::ScalarType<Scalar>::type degen_j_tol)
{
    using std::abs;
    Scalar det_j = detJ(f_eval);
    if (abs(Sacado::ScalarValue<Scalar>::eval(det_j)) < degen_j_tol)
    {
        return false;
    }
    Scalar det_j_inv = 1.0 / det_j;

    J_inv[0][0] = (J_ij(f_eval, 1, 1) * J_ij(f_eval, 2, 2)
                   - J_ij(f_eval, 1, 2) * J_ij(f_eval, 2, 1))
                  * det_j_inv;
    J_inv[0][1] = (J_ij(f_eval, 0, 2) * J_ij(f_eval, 2, 1)
                   - J_ij(f_eval, 0, 1) * J_ij(f_eval, 2, 2))
                  * det_j_inv;
    J_inv[0][2] = (J_ij(f_eval, 0, 1) * J_ij(f_eval, 1, 2)
                   - J_ij(f_eval, 0, 2) * J_ij(f_eval, 1, 1))
                  * det_j_inv;

    J_inv[1][0] = (J_ij(f_eval, 1, 2) * J_ij(f_eval, 2, 0)
                   - J_ij(f_eval, 1, 0) * J_ij(f_eval, 2, 2))
                  * det_j_inv;
    J_inv[1][1] = (J_ij(f_eval, 0, 0) * J_ij(f_eval, 2, 2)
                   - J_ij(f_eval, 0, 2) * J_ij(f_eval, 2, 0))
                  * det_j_inv;
    J_inv[1][2] = (J_ij(f_eval, 0, 2) * J_ij(f_eval, 1, 0)
                   - J_ij(f_eval, 0, 0) * J_ij(f_eval, 1, 2))
                  * det_j_inv;

    J_inv[2][0] = (J_ij(f_eval, 1, 0) * J_ij(f_eval, 2, 1)
                   - J_ij(f_eval, 1, 1) * J_ij(f_eval, 2, 0))
                  * det_j_inv;
    J_inv[2][1] = (J_ij(f_eval, 0, 1) * J_ij(f_eval, 2, 0)
                   - J_ij(f_eval, 0, 0) * J_ij(f_eval, 2, 1))
                  * det_j_inv;
    J_inv[2][2] = (J_ij(f_eval, 0, 0) * J_ij(f_eval, 1, 1)
                   - J_ij(f_eval, 0, 1) * J_ij(f_eval, 1, 0))
                  * det_j_inv;

    return true;
}

//---------------------------------------------------------------------------//
// Evaluate and differentiate the residual
template<class Scalar, std::size_t N, int NumDeriv, class ResidualFunc>
KOKKOS_INLINE_FUNCTION void
evaluateResidual(Kokkos::Array<Sacado::Fad::SFad<Scalar, NumDeriv>, N>& f_eval,
                 Kokkos::Array<Sacado::Fad::SFad<Scalar, NumDeriv>, N>& u,
                 const Kokkos::Array<Scalar, N>& x,
                 const ResidualFunc& F)
{
    static_assert(N == NumDeriv, "Jacobian must be square");

    // Setup derivatives.
    for (std::size_t i = 0; i < N; ++i)
    {
        u[i] = x[i];
        u[i].diff(i, N);
    }

    // Evaluate and differentiate the residual.
    F(u, f_eval);
}

//---------------------------------------------------------------------------//
// Update the solution using Newton's method.
template<class Scalar, std::size_t N, int NumDeriv>
KOKKOS_INLINE_FUNCTION bool updateSolution(
    Kokkos::Array<Scalar, N>& x,
    Kokkos::Array<Kokkos::Array<Scalar, N>, N>& J_inv,
    const Kokkos::Array<Sacado::Fad::SFad<Scalar, NumDeriv>, N>& f_eval,
    const typename Sacado::ScalarType<Scalar>::type degen_j_tol)
{
    static_assert(N == NumDeriv, "Jacobian must be square");

    // Invert the Jacobian and check for degeneracy.
    const bool success = invJ(J_inv, f_eval, degen_j_tol);
    if (!success)
    {
        return false;
    }

    // Solve the linear problem: update = J^-1 * -F(u) and
    // apply the update: x += update
    for (std::size_t i = 0; i < N; ++i)
    {
        for (std::size_t j = 0; j < N; ++j)
        {
            x[i] -= J_inv[i][j] * f_i(f_eval, j);
        }
    }
    return true;
}

//---------------------------------------------------------------------------//

} // end namespace Impl

//---------------------------------------------------------------------------//
// Perform a thread-local nonlinear solve with Newton's method.
//
// An initial guess is given by x and the solution is also output in x.
//
// The Residual Func signature accepts a const Kokkos array for x and a
// non-const Kokkos array for the residual evaluation.
//
// This function returns false if convergence was not achieved.
template<class Scalar, std::size_t N, class ResidualFunc>
KOKKOS_INLINE_FUNCTION bool
solve(Kokkos::Array<Scalar, N>& x,
      const ResidualFunc& F,
      const typename Sacado::ScalarType<Scalar>::type newton_tolerance,
      const int max_iters)
{
    using std::abs;
    using fad_type = Sacado::Fad::SFad<Scalar, N>;
    using value_type = typename Sacado::ScalarType<Scalar>::type;

    // Evaluate the initial residual and Jacobian.
    Kokkos::Array<fad_type, N> u;
    Kokkos::Array<fad_type, N> f_eval;
    Impl::evaluateResidual(f_eval, u, x, F);

    // Iterate until converged or maximum iteration count is reached.
    Kokkos::Array<Kokkos::Array<Scalar, N>, N> J_inv;
    const value_type j_tol
        = 10.0 * Kokkos::Experimental::epsilon<value_type>::value;
    for (int k = 0; k < max_iters; ++k)
    {
        // Update the solution.
        const bool success = Impl::updateSolution(x, J_inv, f_eval, j_tol);

        // Check for degeneracy of the Jacobian. If it is degenerate then the
        // problem is ill-conditioned and we don't expect convergence.
        if (!success)
        {
            return false;
        }

        // Check for convergence of the absolute infinity norm of the
        // residual.
        bool is_converged = true;
        for (std::size_t i = 0; i < N; ++i)
        {
            if (newton_tolerance
                < abs(Sacado::ScalarValue<fad_type>::eval(f_eval[i])))
            {
                is_converged = false;
                break;
            }
        }
        if (is_converged)
        {
            return true;
        }

        // Evaluate the residual and Jacobian.
        Impl::evaluateResidual(f_eval, u, x, F);
    }

    // Convergence was not achieved within the maxiumum number of iterations.
    return false;
}

//---------------------------------------------------------------------------//

} // end namespace NonlinearSolver
} // end namespace Utils
} // end namespace VertexCFD

#endif // end VERTEXCFD_UTILS_NONLINEARSOLVER_HPP
