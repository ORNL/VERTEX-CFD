#include <VertexCFD_Utils_MatrixMath.hpp>

#include <Phalanx_KokkosDeviceTypes.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

using namespace VertexCFD;

namespace Test
{
//---------------------------------------------------------------------------//
void lupDecompAndSolveTest3x3()
{
    constexpr int dim = 3;
    const Kokkos::View<double[dim][dim], PHX::mem_space> A("A");
    const Kokkos::View<int[dim], PHX::mem_space> p("p");
    const Kokkos::View<double[dim], PHX::mem_space> b("b");
    const Kokkos::View<double[dim], PHX::mem_space> w("w");

    // Create host mirror views
    auto A_host = Kokkos::create_mirror_view(A);
    auto p_host = Kokkos::create_mirror_view(p);
    auto b_host = Kokkos::create_mirror_view(b);

    A_host(0, 0) = 1;
    A_host(0, 1) = 2;
    A_host(0, 2) = 4;

    A_host(1, 0) = 2;
    A_host(1, 1) = 1;
    A_host(1, 2) = 3;

    A_host(2, 0) = 3;
    A_host(2, 1) = 2;
    A_host(2, 2) = 4;

    b_host(0) = 3;
    b_host(1) = 4;
    b_host(2) = 5;

    // Deep copy the initialized view to device
    Kokkos::deep_copy(A, A_host);
    Kokkos::deep_copy(b, b_host);

    // Apply the operation in a kernel.
    Kokkos::parallel_for(
        "lu_decomp_and_solve",
        Kokkos::RangePolicy<PHX::exec_space>(0, 1),
        KOKKOS_LAMBDA(const int) {
            MatrixMath::LUP(A, p);
            MatrixMath::LUP_solve(A, p, w, b);
        });

    // Deep copy results to host
    Kokkos::deep_copy(A_host, A);
    Kokkos::deep_copy(p_host, p);
    Kokkos::deep_copy(b_host, b);

    const double A_exp[dim][dim]
        = {{1. / 3., 4. / 3., 8. / 3.}, {2. / 3., -1. / 4., 1.}, {3., 2., 4.}};

    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
            EXPECT_DOUBLE_EQ(A_exp[i][j], A_host(i, j));

    const int p_exp[dim] = {2, 0, 1};
    const double b_exp[dim] = {1, -1, 1};

    for (int i = 0; i < dim; ++i)
    {
        EXPECT_EQ(p_exp[i], p_host(i));
        EXPECT_DOUBLE_EQ(b_exp[i], b_host(i));
    }
}

//---------------------------------------------------------------------------//
void lupDecompTest5x5()
{
    constexpr int dim = 5;
    const Kokkos::View<double[dim][dim], PHX::mem_space> A("A");
    const Kokkos::View<int[dim], PHX::mem_space> p("p");

    // Create host mirror views
    auto A_host = Kokkos::create_mirror_view(A);
    auto p_host = Kokkos::create_mirror_view(p);

    A_host(0, 0) = 8;
    A_host(0, 1) = -2;
    A_host(0, 2) = -8;
    A_host(0, 3) = 9;
    A_host(0, 4) = 6;

    A_host(1, 0) = 9;
    A_host(1, 1) = 5;
    A_host(1, 2) = 5;
    A_host(1, 3) = 3;
    A_host(1, 4) = 1;

    A_host(2, 0) = 9;
    A_host(2, 1) = -7;
    A_host(2, 2) = 2;
    A_host(2, 3) = 8;
    A_host(2, 4) = 4;

    A_host(3, 0) = 9;
    A_host(3, 1) = 6;
    A_host(3, 2) = -9;
    A_host(3, 3) = 0;
    A_host(3, 4) = -8;

    A_host(4, 0) = -7;
    A_host(4, 1) = 9;
    A_host(4, 2) = -9;
    A_host(4, 3) = -7;
    A_host(4, 4) = 6;

    // Deep copy the initialized view to device
    Kokkos::deep_copy(A, A_host);

    // Apply the operation in a kernel.
    Kokkos::parallel_for(
        "lu_decomp",
        Kokkos::RangePolicy<PHX::exec_space>(0, 1),
        KOKKOS_LAMBDA(const int) { MatrixMath::LUP(A, p); });

    // Deep copy results to host
    Kokkos::deep_copy(A_host, A);
    Kokkos::deep_copy(p_host, p);

    const double A_exp[dim][dim]
        = {{8.0 / 9.0, -0.5, -15.0, 4.0, 8.5},
           {9.0, 5.0, 5.0, 3.0, 1.0},
           {1.0, -27.0 / 29.0, 15.0 / 29.0, 410.0 / 1817.0, 31989.0 / 3634.0},
           {1.0, 9.0 / 116.0, 263.0 / 290.0, -1817.0 / 290.0, -2499.0 / 145.0},
           {-7.0 / 9.0, 116.0 / 9.0, -46.0 / 9.0, -14.0 / 3.0, 61.0 / 9.0}};

    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
            EXPECT_NEAR(A_exp[i][j], A_host(i, j), 5.0e-15);

    const int p_exp[dim] = {1, 4, 0, 3, 2};

    for (int i = 0; i < dim; ++i)
        EXPECT_EQ(p_exp[i], p_host(i));
}

//---------------------------------------------------------------------------//
void lupSolveTest5x5()
{
    constexpr int dim = 5;
    const Kokkos::View<double[dim][dim], PHX::mem_space> A("A");
    const Kokkos::View<int[dim], PHX::mem_space> p("p");
    const Kokkos::View<double[dim], PHX::mem_space> b("b");
    const Kokkos::View<double[dim], PHX::mem_space> w("w");

    // Create host mirror views
    auto A_host = Kokkos::create_mirror_view(A);
    auto b_host = Kokkos::create_mirror_view(b);

    A_host(0, 0) = -2.6;
    A_host(0, 1) = -7.2;
    A_host(0, 2) = -0.4;
    A_host(0, 3) = 7.5;
    A_host(0, 4) = 5.7;

    A_host(1, 0) = 2.2;
    A_host(1, 1) = -4.9;
    A_host(1, 2) = 3.8;
    A_host(1, 3) = -8.7;
    A_host(1, 4) = -4.6;

    A_host(2, 0) = 4.8;
    A_host(2, 1) = -4.6;
    A_host(2, 2) = 5.7;
    A_host(2, 3) = 7.9;
    A_host(2, 4) = -8.5;

    A_host(3, 0) = -4.5;
    A_host(3, 1) = 1.4;
    A_host(3, 2) = -9.5;
    A_host(3, 3) = 2.7;
    A_host(3, 4) = -7.2;

    A_host(4, 0) = 2.1;
    A_host(4, 1) = -2.6;
    A_host(4, 2) = -7.4;
    A_host(4, 3) = -5.3;
    A_host(4, 4) = 0.3;

    b_host(0) = -10.19;
    b_host(1) = 101.92;
    b_host(2) = 43.16;
    b_host(3) = -123.63;
    b_host(4) = 4.24;

    // Deep copy the initialized view to device
    Kokkos::deep_copy(A, A_host);
    Kokkos::deep_copy(b, b_host);

    // Apply the operation in a kernel.
    Kokkos::parallel_for(
        "lu_decomp",
        Kokkos::RangePolicy<PHX::exec_space>(0, 1),
        KOKKOS_LAMBDA(const int) {
            MatrixMath::LUP(A, p);
            MatrixMath::LUP_solve(A, p, w, b);
        });

    // Deep copy results to host
    Kokkos::deep_copy(A_host, A);
    Kokkos::deep_copy(b_host, b);

    const double b_exp[dim] = {6.5, -5.3, 6.7, -4.9, 1.4};

    for (int i = 0; i < dim; ++i)
        EXPECT_DOUBLE_EQ(b_exp[i], b_host(i));
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST(MatrixMath, lupDecomp3x3)
{
    lupDecompAndSolveTest3x3();
}
//---------------------------------------------------------------------------//
TEST(MatrixMath, lupDecomp5x5)
{
    lupDecompTest5x5();
}
//---------------------------------------------------------------------------//
TEST(MatrixMath, lupSolve5x5)
{
    lupSolveTest5x5();
}
//---------------------------------------------------------------------------//
} // namespace Test
