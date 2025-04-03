#include <VertexCFD_Utils_NonlinearSolver.hpp>

#include <Panzer_Traits.hpp>

#include <Phalanx_KokkosDeviceTypes.hpp>

#include <Sacado.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <array>
#include <type_traits>

using namespace VertexCFD;

namespace Test
{
//---------------------------------------------------------------------------//
// solver parameters
constexpr double nonlinear_tol = 1.0e-12;
constexpr int nonlinear_max_iter = 10;

//---------------------------------------------------------------------------//
// create views
auto createView(panzer::Traits::Residual, const int size)
{
    using scalar_type = typename panzer::Traits::Residual::ScalarT;
    return Kokkos::View<scalar_type*, Kokkos::HostSpace>("x", size);
}

auto createView(panzer::Traits::Jacobian, const int size)
{
    using scalar_type = typename panzer::Traits::Jacobian::ScalarT;
    return Kokkos::View<scalar_type*, Kokkos::HostSpace>("x", size, 1);
}

//---------------------------------------------------------------------------//
// Solve on device.
template<class ResidualFunc, class Scalar, std::size_t N>
bool solve(std::integral_constant<std::size_t, N>,
           Kokkos::View<Scalar*, Kokkos::HostSpace>& x,
           const ResidualFunc& F,
           const double tolerance,
           const int max_iters)
{
    // Make device data.
    auto x_dev = Kokkos::create_mirror_view(PHX::mem_space{}, x);
    Kokkos::deep_copy(x_dev, x);
    Kokkos::View<bool[1], PHX::mem_space> result("solve_result");

    // Solve.
    Kokkos::parallel_for(
        "nonlinear_solve",
        Kokkos::RangePolicy<PHX::exec_space>(0, 1),
        KOKKOS_LAMBDA(const int) {
            Kokkos::Array<Scalar, N> u;
            for (std::size_t i = 0; i < N; ++i)
                u[i] = x_dev(i);
            result(0)
                = Utils::NonlinearSolver::solve(u, F, tolerance, max_iters);
            for (std::size_t i = 0; i < N; ++i)
                x_dev(i) = u[i];
        });

    // Move data to host.
    Kokkos::deep_copy(x, x_dev);
    auto result_host
        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result);
    return result_host(0);
}

//---------------------------------------------------------------------------//
// Linear Function 1D
//---------------------------------------------------------------------------//
struct LinearFunc1d
{
    template<class Scalar>
    KOKKOS_INLINE_FUNCTION void operator()(const Kokkos::Array<Scalar, 1>& u,
                                           Kokkos::Array<Scalar, 1>& f) const
    {
        f[0] = 2.0 * u[0] + 1.0;
    }
};

//---------------------------------------------------------------------------//
template<class EvalType>
void runLinearTest1d()
{
    using scalar_type = typename EvalType::ScalarT;
    auto x = createView(EvalType{}, 1);
    x(0) = 4.5;
    auto result = solve(std::integral_constant<std::size_t, 1>{},
                        x,
                        LinearFunc1d{},
                        nonlinear_tol,
                        nonlinear_max_iter);
    EXPECT_TRUE(result);
    EXPECT_NEAR(
        Sacado::ScalarValue<scalar_type>::eval(x(0)), -0.5, nonlinear_tol);
}

//---------------------------------------------------------------------------//
TEST(NonlinearSolver, linear_1_residual)
{
    runLinearTest1d<panzer::Traits::Residual>();
}

TEST(NonlinearSolver, linear_1_jacobian)
{
    runLinearTest1d<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//
// Quadratic Function 1D
//---------------------------------------------------------------------------//
struct QuadraticFunc1d
{
    template<class Scalar>
    KOKKOS_INLINE_FUNCTION void operator()(const Kokkos::Array<Scalar, 1>& u,
                                           Kokkos::Array<Scalar, 1>& f) const
    {
        f[0] = 2.0 * u[0] * u[0] - u[0] - 3.0;
    }
};

//---------------------------------------------------------------------------//
template<class EvalType>
void runQuadraticTest1d()
{
    using scalar_type = typename EvalType::ScalarT;
    auto x = createView(EvalType{}, 1);

    // Root 1
    x(0) = 5.0;
    auto result = solve(std::integral_constant<std::size_t, 1>{},
                        x,
                        QuadraticFunc1d{},
                        nonlinear_tol,
                        nonlinear_max_iter);
    EXPECT_TRUE(result);
    EXPECT_NEAR(
        Sacado::ScalarValue<scalar_type>::eval(x(0)), 1.5, nonlinear_tol);

    // Root 2
    x(0) = -2.0;
    result = solve(std::integral_constant<std::size_t, 1>{},
                   x,
                   QuadraticFunc1d{},
                   nonlinear_tol,
                   nonlinear_max_iter);
    EXPECT_TRUE(result);
    EXPECT_NEAR(
        Sacado::ScalarValue<scalar_type>::eval(x(0)), -1.0, nonlinear_tol);
}

//---------------------------------------------------------------------------//
TEST(NonlinearSolver, quadratic_1_residual)
{
    runQuadraticTest1d<panzer::Traits::Residual>();
}

TEST(NonlinearSolver, quadratic_1_jacobian)
{
    runQuadraticTest1d<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//
// Linear Function 2D
//---------------------------------------------------------------------------//
struct LinearFunc2d
{
    template<class Scalar>
    KOKKOS_INLINE_FUNCTION void operator()(const Kokkos::Array<Scalar, 2>& u,
                                           Kokkos::Array<Scalar, 2>& f) const
    {
        f[0] = 2.0 * u[0] + 3.0 * u[1] - 1.0;
        f[1] = u[0] - 0.5 * u[1];
    }
};

//---------------------------------------------------------------------------//
template<class EvalType>
void runLinearTest2d()
{
    using scalar_type = typename EvalType::ScalarT;
    auto x = createView(EvalType{}, 2);
    x(0) = 1.5;
    x(1) = -1.0;
    auto result = solve(std::integral_constant<std::size_t, 2>{},
                        x,
                        LinearFunc2d{},
                        nonlinear_tol,
                        nonlinear_max_iter);
    EXPECT_TRUE(result);
    EXPECT_NEAR(
        Sacado::ScalarValue<scalar_type>::eval(x(0)), 0.125, nonlinear_tol);
    EXPECT_NEAR(
        Sacado::ScalarValue<scalar_type>::eval(x(1)), 0.25, nonlinear_tol);
}

//---------------------------------------------------------------------------//
TEST(NonlinearSolver, linear_2_residual)
{
    runLinearTest2d<panzer::Traits::Residual>();
}

TEST(NonlinearSolver, linear_2_jacobian)
{
    runLinearTest2d<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//
// Quadratic Function 2D
//---------------------------------------------------------------------------//
struct QuadraticFunc2d
{
    template<class Scalar>
    KOKKOS_INLINE_FUNCTION void operator()(const Kokkos::Array<Scalar, 2>& u,
                                           Kokkos::Array<Scalar, 2>& f) const
    {
        f[0] = u[1] * u[1] - u[0] + 1.0;
        f[1] = u[0] * u[0] - u[1] - 1.0;
    }
};

//---------------------------------------------------------------------------//
template<class EvalType>
void runQuadraticTest2d()
{
    using scalar_type = typename EvalType::ScalarT;
    auto x = createView(EvalType{}, 2);

    // Root 1
    x(0) = 1.0;
    x(1) = 1.0;
    auto result = solve(std::integral_constant<std::size_t, 2>{},
                        x,
                        QuadraticFunc2d{},
                        nonlinear_tol,
                        nonlinear_max_iter);
    EXPECT_TRUE(result);
    EXPECT_NEAR(Sacado::ScalarValue<scalar_type>::eval(x(0)),
                1.2055694304005904,
                nonlinear_tol);
    EXPECT_NEAR(Sacado::ScalarValue<scalar_type>::eval(x(1)),
                0.45339765151640377,
                nonlinear_tol);
}

//---------------------------------------------------------------------------//
TEST(NonlinearSolver, quadratic_2_residual)
{
    runQuadraticTest2d<panzer::Traits::Residual>();
}

TEST(NonlinearSolver, quadratic_2_jacobian)
{
    runQuadraticTest2d<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//
// Linear Function 3D
//---------------------------------------------------------------------------//
struct LinearFunc3d
{
    template<class Scalar>
    KOKKOS_INLINE_FUNCTION void operator()(const Kokkos::Array<Scalar, 3>& u,
                                           Kokkos::Array<Scalar, 3>& f) const
    {
        f[0] = u[0] - u[1] + u[2] - 1.0;
        f[1] = 2.0 * u[0] - u[2];
        f[2] = u[0] + 0.5 * u[1] + 0.5 * u[2] - 3.0;
    }
};

//---------------------------------------------------------------------------//
template<class EvalType>
void runLinearTest3d()
{
    using scalar_type = typename EvalType::ScalarT;
    auto x = createView(EvalType{}, 3);
    x(0) = 1.0;
    x(1) = 1.0;
    x(2) = 1.0;
    auto result = solve(std::integral_constant<std::size_t, 3>{},
                        x,
                        LinearFunc3d{},
                        nonlinear_tol,
                        nonlinear_max_iter);
    EXPECT_TRUE(result);
    EXPECT_NEAR(
        Sacado::ScalarValue<scalar_type>::eval(x(0)), 1.0, nonlinear_tol);
    EXPECT_NEAR(
        Sacado::ScalarValue<scalar_type>::eval(x(1)), 2.0, nonlinear_tol);
    EXPECT_NEAR(
        Sacado::ScalarValue<scalar_type>::eval(x(2)), 2.0, nonlinear_tol);
}

//---------------------------------------------------------------------------//
TEST(NonlinearSolver, linear_3_residual)
{
    runLinearTest3d<panzer::Traits::Residual>();
}

TEST(NonlinearSolver, linear_3_jacobian)
{
    runLinearTest3d<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//
// Quadratic Function 3D
//---------------------------------------------------------------------------//
struct QuadraticFunc3d
{
    template<class Scalar>
    KOKKOS_INLINE_FUNCTION void operator()(const Kokkos::Array<Scalar, 3>& u,
                                           Kokkos::Array<Scalar, 3>& f) const
    {
        f[0] = u[0] * u[1] * u[2] + u[1] - 1.0;
        f[1] = u[1] * u[1] + u[0] * u[0] - 3.0;
        f[2] = u[0] * u[2] + u[1] - 5.0;
    }
};

//---------------------------------------------------------------------------//
template<class EvalType>
void runQuadraticTest3d()
{
    using scalar_type = typename EvalType::ScalarT;
    auto x = createView(EvalType{}, 3);

    // Root 1
    x(0) = 1.0;
    x(1) = 1.0;
    x(2) = 1.0;
    auto result = solve(std::integral_constant<std::size_t, 3>{},
                        x,
                        QuadraticFunc3d{},
                        nonlinear_tol,
                        nonlinear_max_iter);
    EXPECT_TRUE(result);
    EXPECT_NEAR(Sacado::ScalarValue<scalar_type>::eval(x(0)),
                1.7235320561211331,
                nonlinear_tol);
    EXPECT_NEAR(Sacado::ScalarValue<scalar_type>::eval(x(1)),
                0.1715728752538099,
                nonlinear_tol);
    EXPECT_NEAR(Sacado::ScalarValue<scalar_type>::eval(x(2)),
                2.8014721905507973,
                nonlinear_tol);
}

//---------------------------------------------------------------------------//
TEST(NonlinearSolver, quadratic_3_residual)
{
    runQuadraticTest3d<panzer::Traits::Residual>();
}

TEST(NonlinearSolver, quadratic_3_jacobian)
{
    runQuadraticTest3d<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//
// Maximum iteration failure
//---------------------------------------------------------------------------//
template<class EvalType, class Func, int Dim>
void failMaxIterTest()
{
    auto x = createView(EvalType{}, Dim);

    // Root 1
    for (int d = 0; d < Dim; ++d)
    {
        x(d) = 1.0;
    }
    auto result = solve(
        std::integral_constant<std::size_t, Dim>{}, x, Func{}, nonlinear_tol, 1);
    EXPECT_FALSE(result);
}

//---------------------------------------------------------------------------//
TEST(NonlinearSolver, fail_max_iter_residual)
{
    failMaxIterTest<panzer::Traits::Residual, QuadraticFunc1d, 1>();
    failMaxIterTest<panzer::Traits::Residual, QuadraticFunc2d, 2>();
    failMaxIterTest<panzer::Traits::Residual, QuadraticFunc3d, 3>();
}

TEST(NonlinearSolver, fail_max_iter_jacobian)
{
    failMaxIterTest<panzer::Traits::Jacobian, QuadraticFunc1d, 1>();
    failMaxIterTest<panzer::Traits::Jacobian, QuadraticFunc2d, 2>();
    failMaxIterTest<panzer::Traits::Jacobian, QuadraticFunc3d, 3>();
}

//---------------------------------------------------------------------------//
// Degenerate Jacobian failure
//---------------------------------------------------------------------------//
template<int Dim>
struct DegenerateFunc
{
    template<class Scalar>
    KOKKOS_INLINE_FUNCTION void operator()(const Kokkos::Array<Scalar, Dim>&,
                                           Kokkos::Array<Scalar, Dim>& f) const
    {
        for (int d = 0; d < Dim; ++d)
        {
            f[d] = 1.0;
        }
    }
};

//---------------------------------------------------------------------------//
template<class EvalType, int Dim>
void runDegenerateTest()
{
    auto x = createView(EvalType{}, Dim);
    for (int d = 0; d < Dim; ++d)
    {
        x(d) = 4.5;
    }
    auto result = solve(std::integral_constant<std::size_t, Dim>{},
                        x,
                        DegenerateFunc<Dim>{},
                        nonlinear_tol,
                        nonlinear_max_iter);
    EXPECT_FALSE(result);
}

//---------------------------------------------------------------------------//
TEST(NonlinearSolver, degenerate_jacobian_residual)
{
    runDegenerateTest<panzer::Traits::Residual, 1>();
    runDegenerateTest<panzer::Traits::Residual, 2>();
    runDegenerateTest<panzer::Traits::Residual, 3>();
}

TEST(NonlinearSolver, degenerate_jacobian_jacobian)
{
    runDegenerateTest<panzer::Traits::Jacobian, 1>();
    runDegenerateTest<panzer::Traits::Jacobian, 2>();
    runDegenerateTest<panzer::Traits::Jacobian, 3>();
}

//---------------------------------------------------------------------------//

} // namespace Test
