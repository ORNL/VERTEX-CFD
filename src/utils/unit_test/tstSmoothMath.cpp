#include "VertexCFD_Utils_SmoothMath.hpp"

#include <Phalanx_KokkosDeviceTypes.hpp>

#include <Sacado.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

using namespace VertexCFD;

namespace Test
{
//---------------------------------------------------------------------------//
void absTest()
{
    using fad_type = Sacado::Fad::SFad<double, 1>;

    // Make result views.
    constexpr int num_result = 5;
    Kokkos::View<double[num_result], PHX::mem_space> dbl_result("dbl_result");
    Kokkos::View<fad_type[num_result], PHX::mem_space> fad_result(
        "fad_result");

    // Setup test values.
    const double tol = 1.0e-2;
    Kokkos::Array<double, num_result> x_dbl = {3.4, 0.009, 0.0, -0.007, -1.2};
    Kokkos::Array<fad_type, num_result> x_fad;
    for (int i = 0; i < num_result; ++i)
    {
        x_fad[i] = x_dbl[i];
        x_fad[i].diff(0, 1);
    }

    // Apply the operation in a kernel.
    Kokkos::parallel_for(
        "smooth_abs",
        Kokkos::RangePolicy<PHX::exec_space>(0, num_result),
        KOKKOS_LAMBDA(const int i) {
            dbl_result(i) = SmoothMath::abs(x_dbl[i], tol);
            fad_result(i) = SmoothMath::abs(x_fad[i], tol);
        });

    // Copy results to host.
    auto dbl_host
        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dbl_result);
    auto fad_host
        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fad_result);

    // Check the value results.
    EXPECT_DOUBLE_EQ(3.4, dbl_host(0));
    EXPECT_DOUBLE_EQ(0.00905, dbl_host(1));
    EXPECT_DOUBLE_EQ(0.005, dbl_host(2));
    EXPECT_DOUBLE_EQ(0.00745, dbl_host(3));
    EXPECT_DOUBLE_EQ(1.2, dbl_host(4));

    // Check the derivative results.
    EXPECT_DOUBLE_EQ(3.4, fad_host(0).val());
    EXPECT_DOUBLE_EQ(0.00905, fad_host(1).val());
    EXPECT_DOUBLE_EQ(0.005, fad_host(2).val());
    EXPECT_DOUBLE_EQ(0.00745, fad_host(3).val());
    EXPECT_DOUBLE_EQ(1.2, fad_host(4).val());

    EXPECT_DOUBLE_EQ(1.0, fad_host(0).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.9, fad_host(1).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(2).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(-0.7, fad_host(3).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(-1.0, fad_host(4).fastAccessDx(0));
}

//---------------------------------------------------------------------------//
void minTest()
{
    using fad_type = Sacado::Fad::SFad<double, 2>;

    // Make result views.
    constexpr int num_result = 8;
    Kokkos::View<double[num_result], PHX::mem_space> dbl_result("dbl_result");
    Kokkos::View<fad_type[num_result], PHX::mem_space> fad_result(
        "fad_result");

    // Setup test values.
    const double tol = 1.0e-2;
    Kokkos::Array<double, 2> x_dbl = {3.4, 3.401};
    Kokkos::Array<fad_type, 2> x_fad;
    x_fad[0] = x_dbl[0];
    x_fad[0].diff(0, 2);
    x_fad[1] = x_dbl[1];
    x_fad[1].diff(1, 2);

    // Apply the operation in a kernel.
    Kokkos::parallel_for(
        "smooth_min",
        Kokkos::RangePolicy<PHX::exec_space>(0, 1),
        KOKKOS_LAMBDA(const int) {
            dbl_result(0) = SmoothMath::min(x_dbl[0], x_dbl[1], 0.0);
            fad_result(0) = SmoothMath::min(x_fad[0], x_fad[1], 0.0);

            dbl_result(1) = SmoothMath::min(x_dbl[0], x_dbl[1], tol);
            fad_result(1) = SmoothMath::min(x_fad[0], x_fad[1], tol);

            dbl_result(2) = SmoothMath::min(x_dbl[1], x_dbl[0], 0.0);
            fad_result(2) = SmoothMath::min(x_fad[1], x_fad[0], 0.0);

            dbl_result(3) = SmoothMath::min(x_dbl[1], x_dbl[0], tol);
            fad_result(3) = SmoothMath::min(x_fad[1], x_fad[0], tol);

            dbl_result(4) = SmoothMath::min(x_dbl[0], x_dbl[0], 0.0);
            fad_result(4) = SmoothMath::min(x_fad[0], x_fad[0], 0.0);

            dbl_result(5) = SmoothMath::min(x_dbl[0], x_dbl[0], tol);
            fad_result(5) = SmoothMath::min(x_fad[0], x_fad[0], tol);

            dbl_result(6) = SmoothMath::min(x_dbl[1], x_dbl[1], 0.0);
            fad_result(6) = SmoothMath::min(x_fad[1], x_fad[1], 0.0);

            dbl_result(7) = SmoothMath::min(x_dbl[1], x_dbl[1], tol);
            fad_result(7) = SmoothMath::min(x_fad[1], x_fad[1], tol);
        });

    // Copy results to host.
    auto dbl_host
        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dbl_result);
    auto fad_host
        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fad_result);

    // Check the value results.
    EXPECT_DOUBLE_EQ(3.4, dbl_host(0));
    EXPECT_DOUBLE_EQ(3.397975, dbl_host(1));
    EXPECT_DOUBLE_EQ(3.4, dbl_host(2));
    EXPECT_DOUBLE_EQ(3.397975, dbl_host(3));
    EXPECT_DOUBLE_EQ(3.4, dbl_host(4));
    EXPECT_DOUBLE_EQ(3.3975, dbl_host(5));
    EXPECT_DOUBLE_EQ(3.401, dbl_host(6));
    EXPECT_DOUBLE_EQ(3.3985, dbl_host(7));

    // Check the derivative results.
    EXPECT_DOUBLE_EQ(3.4, fad_host(0).val());
    EXPECT_DOUBLE_EQ(3.397975, fad_host(1).val());
    EXPECT_DOUBLE_EQ(3.4, fad_host(2).val());
    EXPECT_DOUBLE_EQ(3.397975, fad_host(3).val());
    EXPECT_DOUBLE_EQ(3.4, fad_host(4).val());
    EXPECT_DOUBLE_EQ(3.3975, fad_host(5).val());
    EXPECT_DOUBLE_EQ(3.401, fad_host(6).val());
    EXPECT_DOUBLE_EQ(3.3985, fad_host(7).val());

    EXPECT_DOUBLE_EQ(1.0, fad_host(0).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.5499999999999945, fad_host(1).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(1.0, fad_host(2).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.5499999999999945, fad_host(3).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(1.0, fad_host(4).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(1.0, fad_host(5).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(6).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(7).fastAccessDx(0));

    EXPECT_DOUBLE_EQ(0.0, fad_host(0).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(0.4500000000000055, fad_host(1).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(0.0, fad_host(2).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(0.4500000000000055, fad_host(3).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(0.0, fad_host(4).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(0.0, fad_host(5).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(1.0, fad_host(6).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(1.0, fad_host(7).fastAccessDx(1));
}

//---------------------------------------------------------------------------//
void maxTest()
{
    using fad_type = Sacado::Fad::SFad<double, 2>;

    // Make result views.
    constexpr int num_result = 8;
    Kokkos::View<double[num_result], PHX::mem_space> dbl_result("dbl_result");
    Kokkos::View<fad_type[num_result], PHX::mem_space> fad_result(
        "fad_result");

    // Setup test values.
    const double tol = 1.0e-2;
    Kokkos::Array<double, 2> x_dbl = {3.4, 3.401};
    Kokkos::Array<fad_type, 2> x_fad;
    x_fad[0] = x_dbl[0];
    x_fad[0].diff(0, 2);
    x_fad[1] = x_dbl[1];
    x_fad[1].diff(1, 2);

    // Apply the operation in a kernel.
    Kokkos::parallel_for(
        "smooth_max",
        Kokkos::RangePolicy<PHX::exec_space>(0, 1),
        KOKKOS_LAMBDA(const int) {
            dbl_result(0) = SmoothMath::max(x_dbl[0], x_dbl[1], 0.0);
            fad_result(0) = SmoothMath::max(x_fad[0], x_fad[1], 0.0);

            dbl_result(1) = SmoothMath::max(x_dbl[0], x_dbl[1], tol);
            fad_result(1) = SmoothMath::max(x_fad[0], x_fad[1], tol);

            dbl_result(2) = SmoothMath::max(x_dbl[1], x_dbl[0], 0.0);
            fad_result(2) = SmoothMath::max(x_fad[1], x_fad[0], 0.0);

            dbl_result(3) = SmoothMath::max(x_dbl[1], x_dbl[0], tol);
            fad_result(3) = SmoothMath::max(x_fad[1], x_fad[0], tol);

            dbl_result(4) = SmoothMath::max(x_dbl[0], x_dbl[0], 0.0);
            fad_result(4) = SmoothMath::max(x_fad[0], x_fad[0], 0.0);

            dbl_result(5) = SmoothMath::max(x_dbl[0], x_dbl[0], tol);
            fad_result(5) = SmoothMath::max(x_fad[0], x_fad[0], tol);

            dbl_result(6) = SmoothMath::max(x_dbl[1], x_dbl[1], 0.0);
            fad_result(6) = SmoothMath::max(x_fad[1], x_fad[1], 0.0);

            dbl_result(7) = SmoothMath::max(x_dbl[1], x_dbl[1], tol);
            fad_result(7) = SmoothMath::max(x_fad[1], x_fad[1], tol);
        });

    // Copy results to host.
    auto dbl_host
        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dbl_result);
    auto fad_host
        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fad_result);

    // Check the value results.
    EXPECT_DOUBLE_EQ(3.401, dbl_host(0));
    EXPECT_DOUBLE_EQ(3.403025, dbl_host(1));
    EXPECT_DOUBLE_EQ(3.401, dbl_host(2));
    EXPECT_DOUBLE_EQ(3.403025, dbl_host(3));
    EXPECT_DOUBLE_EQ(3.4, dbl_host(4));
    EXPECT_DOUBLE_EQ(3.4025, dbl_host(5));
    EXPECT_DOUBLE_EQ(3.401, dbl_host(6));
    EXPECT_DOUBLE_EQ(3.4035, dbl_host(7));

    // Check the derivative results.
    EXPECT_DOUBLE_EQ(3.401, fad_host(0).val());
    EXPECT_DOUBLE_EQ(3.403025, fad_host(1).val());
    EXPECT_DOUBLE_EQ(3.401, fad_host(2).val());
    EXPECT_DOUBLE_EQ(3.403025, fad_host(3).val());
    EXPECT_DOUBLE_EQ(3.4, fad_host(4).val());
    EXPECT_DOUBLE_EQ(3.4025, fad_host(5).val());
    EXPECT_DOUBLE_EQ(3.401, fad_host(6).val());
    EXPECT_DOUBLE_EQ(3.4035, fad_host(7).val());

    EXPECT_DOUBLE_EQ(0.0, fad_host(0).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.4500000000000055, fad_host(1).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(2).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.4500000000000055, fad_host(3).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(1.0, fad_host(4).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(1.0, fad_host(5).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(6).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(7).fastAccessDx(0));

    EXPECT_DOUBLE_EQ(1.0, fad_host(0).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(0.5499999999999945, fad_host(1).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(1.0, fad_host(2).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(0.5499999999999945, fad_host(3).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(0.0, fad_host(4).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(0.0, fad_host(5).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(1.0, fad_host(6).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(1.0, fad_host(7).fastAccessDx(1));
}

//---------------------------------------------------------------------------//
void clampTest()
{
    using fad_type = Sacado::Fad::SFad<double, 1>;

    // Make result views.
    constexpr int num_result = 22;
    Kokkos::View<double[num_result], PHX::mem_space> dbl_result("dbl_result");
    Kokkos::View<fad_type[num_result], PHX::mem_space> fad_result(
        "fad_result");

    // Setup test values.
    const double tol = 0.25;
    double clamp_lo = -2.125;
    double clamp_hi = 4.875;
    constexpr int num_value = 11;
    Kokkos::Array<double, num_value> x_dbl
        = {-std::numeric_limits<double>::infinity(),
           -100.0,
           -2.375,
           5.125,
           100.0,
           std::numeric_limits<double>::infinity(),
           -1.875,
           2.125,
           4.625,
           -2.125,
           4.875};
    Kokkos::Array<fad_type, num_value> x_fad;
    for (int i = 0; i < num_value; ++i)
    {
        x_fad[i] = x_dbl[i];
        x_fad[i].diff(0, 1);
    }

    // Apply the operation in a kernel.
    Kokkos::parallel_for(
        "smooth_abs",
        Kokkos::RangePolicy<PHX::exec_space>(0, num_value),
        KOKKOS_LAMBDA(const int i) {
            dbl_result(i)
                = SmoothMath::clamp(x_dbl[i], clamp_lo, clamp_hi, tol);
            fad_result(i)
                = SmoothMath::clamp(x_fad[i], clamp_lo, clamp_hi, tol);
            dbl_result(i + num_value)
                = SmoothMath::clamp(x_dbl[i], clamp_lo, clamp_hi, 0.0);
            fad_result(i + num_value)
                = SmoothMath::clamp(x_fad[i], clamp_lo, clamp_hi, 0.0);
        });

    // Copy results to host.
    auto dbl_host
        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dbl_result);
    auto fad_host
        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fad_result);

    // Check the value results.
    EXPECT_DOUBLE_EQ(-2.125, dbl_host(0));
    EXPECT_DOUBLE_EQ(-2.125, dbl_host(1));
    EXPECT_DOUBLE_EQ(-2.125, dbl_host(2));
    EXPECT_DOUBLE_EQ(4.875, dbl_host(3));
    EXPECT_DOUBLE_EQ(4.875, dbl_host(4));
    EXPECT_DOUBLE_EQ(4.875, dbl_host(5));
    EXPECT_DOUBLE_EQ(-1.875, dbl_host(6));
    EXPECT_DOUBLE_EQ(2.125, dbl_host(7));
    EXPECT_DOUBLE_EQ(4.625, dbl_host(8));
    EXPECT_DOUBLE_EQ(-2.0625, dbl_host(9));
    EXPECT_DOUBLE_EQ(4.8125, dbl_host(10));

    EXPECT_DOUBLE_EQ(-2.125, dbl_host(11));
    EXPECT_DOUBLE_EQ(-2.125, dbl_host(12));
    EXPECT_DOUBLE_EQ(-2.125, dbl_host(13));
    EXPECT_DOUBLE_EQ(4.875, dbl_host(14));
    EXPECT_DOUBLE_EQ(4.875, dbl_host(15));
    EXPECT_DOUBLE_EQ(4.875, dbl_host(16));
    EXPECT_DOUBLE_EQ(-1.875, dbl_host(17));
    EXPECT_DOUBLE_EQ(2.125, dbl_host(18));
    EXPECT_DOUBLE_EQ(4.625, dbl_host(19));
    EXPECT_DOUBLE_EQ(-2.125, dbl_host(20));
    EXPECT_DOUBLE_EQ(4.875, dbl_host(21));

    // Check the derivative results.
    EXPECT_DOUBLE_EQ(-2.125, fad_host(0).val());
    EXPECT_DOUBLE_EQ(-2.125, fad_host(1).val());
    EXPECT_DOUBLE_EQ(-2.125, fad_host(2).val());
    EXPECT_DOUBLE_EQ(4.875, fad_host(3).val());
    EXPECT_DOUBLE_EQ(4.875, fad_host(4).val());
    EXPECT_DOUBLE_EQ(4.875, fad_host(5).val());
    EXPECT_DOUBLE_EQ(-1.875, fad_host(6).val());
    EXPECT_DOUBLE_EQ(2.125, fad_host(7).val());
    EXPECT_DOUBLE_EQ(4.625, fad_host(8).val());
    EXPECT_DOUBLE_EQ(-2.0625, fad_host(9).val());
    EXPECT_DOUBLE_EQ(4.8125, fad_host(10).val());

    EXPECT_DOUBLE_EQ(-2.125, fad_host(11).val());
    EXPECT_DOUBLE_EQ(-2.125, fad_host(12).val());
    EXPECT_DOUBLE_EQ(-2.125, fad_host(13).val());
    EXPECT_DOUBLE_EQ(4.875, fad_host(14).val());
    EXPECT_DOUBLE_EQ(4.875, fad_host(15).val());
    EXPECT_DOUBLE_EQ(4.875, fad_host(16).val());
    EXPECT_DOUBLE_EQ(-1.875, fad_host(17).val());
    EXPECT_DOUBLE_EQ(2.125, fad_host(18).val());
    EXPECT_DOUBLE_EQ(4.625, fad_host(19).val());
    EXPECT_DOUBLE_EQ(-2.125, fad_host(20).val());
    EXPECT_DOUBLE_EQ(4.875, fad_host(21).val());

    EXPECT_DOUBLE_EQ(0.0, fad_host(0).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(1).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(2).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(3).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(4).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(5).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(1.0, fad_host(6).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(1.0, fad_host(7).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(1.0, fad_host(8).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.5, fad_host(9).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.5, fad_host(10).fastAccessDx(0));

    EXPECT_DOUBLE_EQ(0.0, fad_host(11).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(12).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(13).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(14).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(15).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(16).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(1.0, fad_host(17).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(1.0, fad_host(18).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(1.0, fad_host(19).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(20).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(21).fastAccessDx(0));
}

//---------------------------------------------------------------------------//
void rampTest()
{
    using fad_type = Sacado::Fad::SFad<double, 1>;

    // Make result views.
    constexpr int num_result = 5;
    Kokkos::View<double[num_result], PHX::mem_space> dbl_result("dbl_result");
    Kokkos::View<fad_type[num_result], PHX::mem_space> fad_result(
        "fad_result");

    // Setup test values.
    const double ramp_start = 0.5;
    const double ramp_end = 1.5;
    Kokkos::Array<double, num_result> x_dbl = {0.45, 0.5, 1.0, 1.5, 1.55};
    Kokkos::Array<fad_type, num_result> x_fad;
    for (int i = 0; i < num_result; ++i)
    {
        x_fad[i] = x_dbl[i];
        x_fad[i].diff(0, 1);
    }

    // Apply the operation in a kernel.
    Kokkos::parallel_for(
        "smooth_ramp",
        Kokkos::RangePolicy<PHX::exec_space>(0, num_result),
        KOKKOS_LAMBDA(const int i) {
            dbl_result(i) = SmoothMath::ramp(x_dbl[i], ramp_start, ramp_end);
            fad_result(i) = SmoothMath::ramp(x_fad[i], ramp_start, ramp_end);
        });

    // Copy results to host.
    auto dbl_host
        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dbl_result);
    auto fad_host
        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fad_result);

    // Check the value results.
    EXPECT_DOUBLE_EQ(0.0, dbl_host(0));
    EXPECT_DOUBLE_EQ(0.0, dbl_host(1));
    EXPECT_DOUBLE_EQ(0.5, dbl_host(2));
    EXPECT_DOUBLE_EQ(1.0, dbl_host(3));
    EXPECT_DOUBLE_EQ(1.0, dbl_host(4));

    // Check the derivative results.
    EXPECT_DOUBLE_EQ(0.0, fad_host(0).val());
    EXPECT_DOUBLE_EQ(0.0, fad_host(1).val());
    EXPECT_DOUBLE_EQ(0.5, fad_host(2).val());
    EXPECT_DOUBLE_EQ(1.0, fad_host(3).val());
    EXPECT_DOUBLE_EQ(1.0, fad_host(4).val());

    EXPECT_DOUBLE_EQ(0.0, fad_host(0).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(1).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(1.5707963267948966, fad_host(2).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(3).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(4).fastAccessDx(0));
}

//---------------------------------------------------------------------------//
void hypot2DTest()
{
    using fad_type = Sacado::Fad::SFad<double, 2>;

    // Make result views.
    constexpr int num_result = 9;
    Kokkos::View<double[num_result], PHX::mem_space> dbl_result("dbl_result");
    Kokkos::View<fad_type[num_result], PHX::mem_space> fad_result(
        "fad_result");

    // Setup test values.
    const double tol = 4.0;
    const double x_dbl = 1.5;
    const double y_dbl = -2.0;
    const fad_type x_fad(2, 0, x_dbl);
    const fad_type y_fad(2, 1, y_dbl);

    const fad_type x0_fad(2, 0, 0.0);
    const fad_type y0_fad(2, 1, 0.0);

    // Apply the operation in a kernel.
    Kokkos::parallel_for(
        "smooth_hypot",
        Kokkos::RangePolicy<PHX::exec_space>(0, 1),
        KOKKOS_LAMBDA(const int) {
            dbl_result(0) = SmoothMath::hypot(x_dbl, y_dbl, 0.0);
            fad_result(0) = SmoothMath::hypot(x_fad, y_fad, 0.0);

            dbl_result(1) = SmoothMath::hypot(x_dbl, y_dbl, tol);
            fad_result(1) = SmoothMath::hypot(x_fad, y_fad, tol);

            dbl_result(2) = SmoothMath::hypot(y_dbl, x_dbl, 0.0);
            fad_result(2) = SmoothMath::hypot(y_fad, x_fad, 0.0);

            dbl_result(3) = SmoothMath::hypot(y_dbl, x_dbl, tol);
            fad_result(3) = SmoothMath::hypot(y_fad, x_fad, tol);

            dbl_result(4) = SmoothMath::hypot(x_dbl, x_dbl, 0.0);
            fad_result(4) = SmoothMath::hypot(x_fad, x_fad, 0.0);

            dbl_result(5) = SmoothMath::hypot(x_dbl, x_dbl, tol);
            fad_result(5) = SmoothMath::hypot(x_fad, x_fad, tol);

            dbl_result(6) = SmoothMath::hypot(y_dbl, y_dbl, 0.0);
            fad_result(6) = SmoothMath::hypot(y_fad, y_fad, 0.0);

            dbl_result(7) = SmoothMath::hypot(y_dbl, y_dbl, tol);
            fad_result(7) = SmoothMath::hypot(y_fad, y_fad, tol);

            dbl_result(8) = SmoothMath::hypot(0.0, 0.0, tol);
            fad_result(8) = SmoothMath::hypot(x0_fad, y0_fad, tol);
        });

    // Copy results to host.
    auto dbl_host
        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dbl_result);
    auto fad_host
        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fad_result);

    const double sqrt2 = std::sqrt(2);

    // Check the value results.
    EXPECT_DOUBLE_EQ(2.5, dbl_host(0));
    EXPECT_DOUBLE_EQ(2.78125, dbl_host(1));
    EXPECT_DOUBLE_EQ(2.5, dbl_host(2));
    EXPECT_DOUBLE_EQ(2.78125, dbl_host(3));
    EXPECT_DOUBLE_EQ(1.5 * sqrt2, dbl_host(4));
    EXPECT_DOUBLE_EQ(2.5625, dbl_host(5));
    EXPECT_DOUBLE_EQ(2.0 * sqrt2, dbl_host(6));
    EXPECT_DOUBLE_EQ(3.0, dbl_host(7));
    EXPECT_DOUBLE_EQ(2.0, dbl_host(8));

    // Check the derivative results.
    EXPECT_DOUBLE_EQ(2.5, fad_host(0).val());
    EXPECT_DOUBLE_EQ(2.78125, fad_host(1).val());
    EXPECT_DOUBLE_EQ(2.5, fad_host(2).val());
    EXPECT_DOUBLE_EQ(2.78125, fad_host(3).val());
    EXPECT_DOUBLE_EQ(1.5 * sqrt2, fad_host(4).val());
    EXPECT_DOUBLE_EQ(2.5625, fad_host(5).val());
    EXPECT_DOUBLE_EQ(2.0 * sqrt2, fad_host(6).val());
    EXPECT_DOUBLE_EQ(3.0, fad_host(7).val());
    EXPECT_DOUBLE_EQ(2.0, fad_host(8).val());

    EXPECT_DOUBLE_EQ(0.6, fad_host(0).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.375, fad_host(1).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.6, fad_host(2).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.375, fad_host(3).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(sqrt2, fad_host(4).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.75, fad_host(5).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(6).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(7).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(8).fastAccessDx(0));

    EXPECT_DOUBLE_EQ(-0.8, fad_host(0).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(-0.5, fad_host(1).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(-0.8, fad_host(2).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(-0.5, fad_host(3).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(0.0, fad_host(4).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(0.0, fad_host(5).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(-sqrt2, fad_host(6).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(-1.0, fad_host(7).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(0.0, fad_host(8).fastAccessDx(1));
}

//---------------------------------------------------------------------------//
void normTest()
{
    static constexpr int num_dbl_result = 4;
    static constexpr int num_dfad_result = 2;
    static constexpr int num_sfad_result = 2;

    using dfad_type = Sacado::Fad::DFad<double>;
    using sfad_2d_type = Sacado::Fad::SFad<double, 2>;
    using sfad_3d_type = Sacado::Fad::SFad<double, 3>;

    // Input vector types
    using dbl_view = Kokkos::View<double*, PHX::mem_space>;
    using dfad_view = Kokkos::View<dfad_type*, PHX::mem_space>;
    using sfad_2d_view = Kokkos::View<sfad_2d_type[2], PHX::mem_space>;
    using sfad_3d_view = Kokkos::View<sfad_3d_type[3], PHX::mem_space>;

    // Results view types
    using dbl_results_view
        = Kokkos::View<double[num_dbl_result], PHX::mem_space>;
    using dfad_results_view
        = Kokkos::View<dfad_type[num_dfad_result], PHX::mem_space>;
    using sfad_2d_results_view
        = Kokkos::View<sfad_2d_type[num_sfad_result], PHX::mem_space>;
    using sfad_3d_results_view
        = Kokkos::View<sfad_3d_type[num_sfad_result], PHX::mem_space>;

    // Create the double vectors
    dbl_view vec_dbl_2d("vec_dbl_2d", 2);
    dbl_view vec_dbl_3d("vec_dbl_3d", 3);

    // Create dfad versions. Note the last extra "hidden" dimension must equal
    // num_derivs + 1
    dfad_view vec_dfad_2d("vec_dfad_2d", 2, 3);
    dfad_view vec_dfad_3d("vec_dfad_3d", 3, 4);

    // Create sfad versions.
    sfad_2d_view vec_sfad_2d("vec_dfad_2d");
    sfad_3d_view vec_sfad_3d("vec_dfad_3d");

    // Create host mirrors for initialization
    auto vec_dbl_2d_host = Kokkos::create_mirror_view(vec_dbl_2d);
    auto vec_dbl_3d_host = Kokkos::create_mirror_view(vec_dbl_3d);
    auto vec_dfad_2d_host = Kokkos::create_mirror(vec_dfad_2d);
    auto vec_dfad_3d_host = Kokkos::create_mirror(vec_dfad_3d);
    auto vec_sfad_2d_host = Kokkos::create_mirror(vec_sfad_2d);
    auto vec_sfad_3d_host = Kokkos::create_mirror(vec_sfad_3d);

    // Initialize the vectors
    vec_dbl_2d_host(0) = 0.25;
    vec_dbl_2d_host(1) = -0.166;

    vec_dbl_3d_host(0) = 0.175;
    vec_dbl_3d_host(1) = 0.33;
    vec_dbl_3d_host(2) = -0.92;

    // Initialize the fad vectors
    vec_dfad_2d_host(0) = dfad_type(2, 0, 0.25);
    vec_dfad_2d_host(1) = dfad_type(2, 1, -0.166);

    vec_sfad_2d_host(0) = sfad_2d_type(2, 0, 0.25);
    vec_sfad_2d_host(1) = sfad_2d_type(2, 1, -0.166);

    vec_dfad_3d_host(0) = dfad_type(3, 0, 0.175);
    vec_dfad_3d_host(1) = dfad_type(3, 1, 0.33);
    vec_dfad_3d_host(2) = dfad_type(3, 2, -0.92);

    vec_sfad_3d_host(0) = sfad_3d_type(3, 0, 0.175);
    vec_sfad_3d_host(1) = sfad_3d_type(3, 1, 0.33);
    vec_sfad_3d_host(2) = sfad_3d_type(3, 2, -0.92);

    // Deep copy the initialized vectors to device
    Kokkos::deep_copy(vec_dbl_2d, vec_dbl_2d_host);
    Kokkos::deep_copy(vec_dbl_3d, vec_dbl_3d_host);
    Kokkos::deep_copy(vec_dfad_2d, vec_dfad_2d_host);
    Kokkos::deep_copy(vec_dfad_3d, vec_dfad_3d_host);
    Kokkos::deep_copy(vec_sfad_2d, vec_sfad_2d_host);
    Kokkos::deep_copy(vec_sfad_3d, vec_sfad_3d_host);

    // Make result views.
    dbl_results_view dbl_result("dbl_result");

    // Note we still need to specify the compile-time dimension size.
    dfad_results_view dfad_2d_results("2d_dfad_results", num_dfad_result, 3);
    dfad_results_view dfad_3d_results("3d_dfad_results", num_dfad_result, 4);

    sfad_2d_results_view sfad_2d_results("2d_sfad_results");
    sfad_3d_results_view sfad_3d_results("3d_sfad_results");

    const double tol = 1.0;

    // Apply the operation in a kernel.
    Kokkos::parallel_for(
        "smooth_norm",
        Kokkos::RangePolicy<PHX::exec_space>(0, 1),
        KOKKOS_LAMBDA(const int) {
            dbl_result(0) = SmoothMath::norm(vec_dbl_2d, 0.0);
            dbl_result(1) = SmoothMath::norm(vec_dbl_2d, tol);
            dbl_result(2) = SmoothMath::norm(vec_dbl_3d, 0.0);
            dbl_result(3) = SmoothMath::norm(vec_dbl_3d, tol);

            dfad_2d_results(0) = SmoothMath::norm(vec_dfad_2d, 0.0);
            dfad_2d_results(1) = SmoothMath::norm(vec_dfad_2d, tol);

            sfad_2d_results(0) = SmoothMath::norm(vec_sfad_2d, 0.0);
            sfad_2d_results(1) = SmoothMath::norm(vec_sfad_2d, tol);

            dfad_3d_results(0) = SmoothMath::norm(vec_dfad_3d, 0.0);
            dfad_3d_results(1) = SmoothMath::norm(vec_dfad_3d, tol);

            sfad_3d_results(0) = SmoothMath::norm(vec_sfad_3d, 0.0);
            sfad_3d_results(1) = SmoothMath::norm(vec_sfad_3d, tol);
        });

    // Copy results to host.
    auto dbl_host
        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dbl_result);
    auto sfad_2d_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, sfad_2d_results);
    auto sfad_3d_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, sfad_3d_results);

    // Kokkos create_mirror_view_and_copy gives an error for DFAD types
    // when node type is CUDA. Use two-step copy for these Views.
    auto dfad_2d_host = Kokkos::create_mirror(dfad_2d_results);
    Kokkos::deep_copy(dfad_2d_host, dfad_2d_results);
    auto dfad_3d_host = Kokkos::create_mirror(dfad_3d_results);
    Kokkos::deep_copy(dfad_3d_host, dfad_3d_results);

    EXPECT_DOUBLE_EQ(0.30009331881932994, dbl_host(0));
    EXPECT_DOUBLE_EQ(0.545028, dbl_host(1));
    EXPECT_DOUBLE_EQ(0.992937560977527, dbl_host(2));
    EXPECT_DOUBLE_EQ(0.9929625, dbl_host(3));

    EXPECT_DOUBLE_EQ(0.30009331881932994, dfad_2d_host(0).val());
    EXPECT_DOUBLE_EQ(0.8330741949990281, dfad_2d_host(0).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(-0.5531612654793547, dfad_2d_host(0).fastAccessDx(1));

    EXPECT_DOUBLE_EQ(0.30009331881932994, sfad_2d_host(0).val());
    EXPECT_DOUBLE_EQ(0.8330741949990281, sfad_2d_host(0).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(-0.5531612654793547, sfad_2d_host(0).fastAccessDx(1));

    EXPECT_DOUBLE_EQ(0.545028, dfad_2d_host(1).val());
    EXPECT_DOUBLE_EQ(0.25, dfad_2d_host(1).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(-0.166, dfad_2d_host(1).fastAccessDx(1));

    EXPECT_DOUBLE_EQ(0.545028, sfad_2d_host(1).val());
    EXPECT_DOUBLE_EQ(0.25, sfad_2d_host(1).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(-0.166, sfad_2d_host(1).fastAccessDx(1));

    EXPECT_DOUBLE_EQ(0.992937560977527, dfad_3d_host(0).val());
    EXPECT_DOUBLE_EQ(0.17624471757087729, dfad_3d_host(0).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.33234718170508293, dfad_3d_host(0).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(-0.9265436580868979, dfad_3d_host(0).fastAccessDx(2));

    EXPECT_DOUBLE_EQ(0.9929625, dfad_3d_host(1).val());
    EXPECT_DOUBLE_EQ(0.175, dfad_3d_host(1).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.33, dfad_3d_host(1).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(-0.92, dfad_3d_host(1).fastAccessDx(2));
}

//---------------------------------------------------------------------------//
void metricNormTest()
{
    static constexpr int num_dbl_result = 8;
    static constexpr int num_dfad_result = 4;

    using dfad_type = Sacado::Fad::DFad<double>;

    // Input types
    using dfad_vector_view = Kokkos::View<dfad_type*, PHX::mem_space>;
    using dfad_matrix_view = Kokkos::View<dfad_type**, PHX::mem_space>;
    using dbl_matrix_view = Kokkos::View<double**, PHX::mem_space>;
    using dbl_view = Kokkos::View<double*, PHX::mem_space>;

    // Results view types
    using dbl_results_view
        = Kokkos::View<double[num_dbl_result], PHX::mem_space>;
    using dfad_results_view
        = Kokkos::View<dfad_type[num_dfad_result], PHX::mem_space>;

    // Create the inputs for 2d
    dfad_vector_view vec_dfad_2d("vec_dfad_2d", 2, 3);
    dbl_view vec_dbl_2d("vec_dbl_2d", 2);
    dfad_matrix_view identity_dfad_2d("identity_dfad_2d", 2, 2, 3);
    dbl_matrix_view identity_dbl_2d("identity_dbl_2d", 2, 2);
    dfad_matrix_view metric_dfad_2d("metric_dfad_2d", 2, 2, 3);
    dbl_matrix_view metric_dbl_2d("metric_dbl_2d", 2, 2);

    // Create the inputs for 3d
    dfad_vector_view vec_dfad_3d("vec_dfad_3d", 3, 4);
    dbl_view vec_dbl_3d("vec_dbl_3d", 3);
    dfad_matrix_view identity_dfad_3d("identity_dfad_3d", 3, 3, 4);
    dbl_matrix_view identity_dbl_3d("identity_dbl_3d", 3, 3);
    dfad_matrix_view metric_dfad_3d("metric_dfad_3d", 3, 3, 4);
    dbl_matrix_view metric_dbl_3d("metric_dbl_3d", 3, 3);

    // Make result views
    dbl_results_view dbl_result("dbl_result");
    dfad_results_view dfad_2d_result("dfad_2d_result", num_dfad_result, 3);
    dfad_results_view dfad_3d_result("dfad_3d_result", num_dfad_result, 4);

    // Create host mirror views for initialization
    auto vec_dbl_2d_host = Kokkos::create_mirror_view(vec_dbl_2d);
    auto vec_dbl_3d_host = Kokkos::create_mirror_view(vec_dbl_3d);
    auto identity_dbl_2d_host = Kokkos::create_mirror_view(identity_dbl_2d);
    auto identity_dbl_3d_host = Kokkos::create_mirror_view(identity_dbl_3d);
    auto metric_dbl_2d_host = Kokkos::create_mirror_view(metric_dbl_2d);
    auto metric_dbl_3d_host = Kokkos::create_mirror_view(metric_dbl_3d);

    // Kokkos create_mirror_view_and_copy gives an error for DFAD types
    // when node type is CUDA. Use two-step copy for these Views.
    auto vec_dfad_2d_host = Kokkos::create_mirror(vec_dfad_2d);
    Kokkos::deep_copy(vec_dfad_2d_host, vec_dfad_2d);
    auto vec_dfad_3d_host = Kokkos::create_mirror(vec_dfad_3d);
    Kokkos::deep_copy(vec_dfad_3d_host, vec_dfad_3d);
    auto identity_dfad_2d_host = Kokkos::create_mirror(identity_dfad_2d);
    Kokkos::deep_copy(identity_dfad_2d_host, identity_dfad_2d);
    auto identity_dfad_3d_host = Kokkos::create_mirror(identity_dfad_3d);
    Kokkos::deep_copy(identity_dfad_3d_host, identity_dfad_3d);
    auto metric_dfad_2d_host = Kokkos::create_mirror(metric_dfad_2d);
    Kokkos::deep_copy(metric_dfad_2d_host, metric_dfad_2d);
    auto metric_dfad_3d_host = Kokkos::create_mirror(metric_dfad_3d);
    Kokkos::deep_copy(metric_dfad_3d_host, metric_dfad_3d);

    // Identity metrics for 2d and 3d
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            identity_dfad_2d_host(i, j) = dfad_type(2, j, 0.0);
            identity_dbl_2d_host(i, j) = 0.0;
        }
        identity_dfad_2d_host(i, i) = dfad_type(2, i, 1.0);
        identity_dbl_2d_host(i, i) = 1.0;
    }
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            identity_dfad_3d_host(i, j) = dfad_type(3, j, 0.0);
            identity_dbl_3d_host(i, j) = 0.0;
        }
        identity_dfad_3d_host(i, i) = dfad_type(3, i, 1.0);
        identity_dbl_3d_host(i, i) = 1.0;
    }

    // Initialize the fad vectors for 2d and 3d
    vec_dfad_2d_host(0) = dfad_type(2, 0, 0.25);
    vec_dfad_2d_host(1) = dfad_type(2, 1, -0.166);

    vec_dfad_3d_host(0) = dfad_type(3, 0, 0.175);
    vec_dfad_3d_host(1) = dfad_type(3, 1, 0.33);
    vec_dfad_3d_host(2) = dfad_type(3, 2, -0.92);

    // Non-trivial metrics for 2d and 3d
    metric_dfad_2d_host(0, 0) = dfad_type(2, 0, 0.75);
    metric_dfad_2d_host(0, 1) = dfad_type(2, 1, -0.25);
    metric_dfad_2d_host(1, 0) = dfad_type(2, 0, -0.25);
    metric_dfad_2d_host(1, 1) = dfad_type(2, 1, 0.75);

    metric_dbl_2d_host(0, 0) = 0.75;
    metric_dbl_2d_host(0, 1) = -0.25;
    metric_dbl_2d_host(1, 0) = -0.25;
    metric_dbl_2d_host(1, 1) = 0.75;

    metric_dfad_3d_host(0, 0) = dfad_type(3, 0, 0.375);
    metric_dfad_3d_host(0, 1) = dfad_type(3, 1, 0.125);
    metric_dfad_3d_host(0, 2) = dfad_type(3, 2, 0.0);
    metric_dfad_3d_host(1, 0) = dfad_type(3, 0, 0.125);
    metric_dfad_3d_host(1, 1) = dfad_type(3, 1, 0.375);
    metric_dfad_3d_host(1, 2) = dfad_type(3, 2, 0.0);
    metric_dfad_3d_host(2, 0) = dfad_type(3, 0, 0.0);
    metric_dfad_3d_host(2, 1) = dfad_type(3, 1, 0.0);
    metric_dfad_3d_host(2, 2) = dfad_type(3, 2, 1.0);

    metric_dbl_3d_host(0, 0) = 0.375;
    metric_dbl_3d_host(0, 1) = 0.125;
    metric_dbl_3d_host(0, 2) = 0.0;
    metric_dbl_3d_host(1, 0) = 0.125;
    metric_dbl_3d_host(1, 1) = 0.375;
    metric_dbl_3d_host(1, 2) = 0.0;
    metric_dbl_3d_host(2, 0) = 0.0;
    metric_dbl_3d_host(2, 1) = 0.0;
    metric_dbl_3d_host(2, 2) = 1.0;

    // Deep copy the initialized data to device
    Kokkos::deep_copy(vec_dbl_2d, vec_dbl_2d_host);
    Kokkos::deep_copy(vec_dbl_3d, vec_dbl_3d_host);
    Kokkos::deep_copy(vec_dfad_2d, vec_dfad_2d_host);
    Kokkos::deep_copy(vec_dfad_3d, vec_dfad_3d_host);
    Kokkos::deep_copy(identity_dbl_2d, identity_dbl_2d_host);
    Kokkos::deep_copy(identity_dbl_3d, identity_dbl_3d_host);
    Kokkos::deep_copy(identity_dfad_2d, identity_dfad_2d_host);
    Kokkos::deep_copy(identity_dfad_3d, identity_dfad_3d_host);
    Kokkos::deep_copy(metric_dbl_2d, metric_dbl_2d_host);
    Kokkos::deep_copy(metric_dbl_3d, metric_dbl_3d_host);
    Kokkos::deep_copy(metric_dfad_2d, metric_dfad_2d_host);
    Kokkos::deep_copy(metric_dfad_3d, metric_dfad_3d_host);

    // Tolerance
    const double tol = 1.0;

    // Apply the operation in a kernel.
    Kokkos::parallel_for(
        "smooth_norm",
        Kokkos::RangePolicy<PHX::exec_space>(0, 1),
        KOKKOS_LAMBDA(const int) {
            dbl_result(0) = SmoothMath::norm(vec_dbl_2d, identity_dbl_2d, 0.0);
            dbl_result(1) = SmoothMath::norm(vec_dbl_2d, identity_dbl_2d, tol);
            dbl_result(2) = SmoothMath::norm(vec_dbl_3d, identity_dbl_3d, 0.0);
            dbl_result(3) = SmoothMath::norm(vec_dbl_3d, identity_dbl_3d, tol);

            dbl_result(4) = SmoothMath::norm(vec_dbl_2d, metric_dbl_2d, 0.0);
            dbl_result(5) = SmoothMath::norm(vec_dbl_2d, metric_dbl_2d, tol);
            dbl_result(6) = SmoothMath::norm(vec_dbl_3d, metric_dbl_3d, 0.0);
            dbl_result(7) = SmoothMath::norm(vec_dbl_3d, metric_dbl_3d, tol);

            dfad_2d_result(0)
                = SmoothMath::norm(vec_dfad_2d, identity_dbl_2d, 0.0);
            dfad_2d_result(1)
                = SmoothMath::norm(vec_dfad_2d, identity_dbl_2d, tol);
            dfad_2d_result(2)
                = SmoothMath::norm(vec_dfad_2d, metric_dbl_2d, 0.0);
            dfad_2d_result(3)
                = SmoothMath::norm(vec_dfad_2d, metric_dbl_2d, tol);

            dfad_3d_result(0)
                = SmoothMath::norm(vec_dfad_3d, identity_dbl_3d, 0.0);
            dfad_3d_result(1)
                = SmoothMath::norm(vec_dfad_3d, identity_dbl_3d, tol);
            dfad_3d_result(2)
                = SmoothMath::norm(vec_dfad_3d, metric_dbl_3d, 0.0);
            dfad_3d_result(3)
                = SmoothMath::norm(vec_dfad_3d, metric_dbl_3d, tol);
        });

    // Copy results to host.
    auto dbl_host
        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dbl_result);

    // Kokkos create_mirror_view_and_copy gives an error for DFAD types
    // when node type is CUDA. Use two-step copy for these Views.
    auto dfad_2d_host = Kokkos::create_mirror(dfad_2d_result);
    Kokkos::deep_copy(dfad_2d_host, dfad_2d_result);
    auto dfad_3d_host = Kokkos::create_mirror(dfad_3d_result);
    Kokkos::deep_copy(dfad_3d_host, dfad_3d_result);

    EXPECT_DOUBLE_EQ(0.300093318819329935, dfad_2d_host(0).val());
    EXPECT_DOUBLE_EQ(0.833074194999028128, dfad_2d_host(0).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(-0.553161265479354736, dfad_2d_host(0).fastAccessDx(1));

    EXPECT_DOUBLE_EQ(0.545028, dfad_2d_host(1).val());
    EXPECT_DOUBLE_EQ(0.25, dfad_2d_host(1).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(-0.166, dfad_2d_host(1).fastAccessDx(1));

    EXPECT_DOUBLE_EQ(0.29713969778540195, dfad_2d_host(2).val());
    EXPECT_DOUBLE_EQ(0.7706812711554505, dfad_2d_host(2).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(-0.6293336144369835, dfad_2d_host(2).fastAccessDx(1));

    EXPECT_DOUBLE_EQ(0.544146, dfad_2d_host(3).val());
    EXPECT_DOUBLE_EQ(0.22899999999999998, dfad_2d_host(3).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(-0.18699999999999997, dfad_2d_host(3).fastAccessDx(1));

    EXPECT_DOUBLE_EQ(0.992937560977526945, dfad_3d_host(0).val());
    EXPECT_DOUBLE_EQ(0.176244717570877285, dfad_3d_host(0).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.332347181705082928, dfad_3d_host(0).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(-0.926543658086897870, dfad_3d_host(0).fastAccessDx(2));

    EXPECT_DOUBLE_EQ(0.9929625, dfad_3d_host(1).val());
    EXPECT_DOUBLE_EQ(0.175, dfad_3d_host(1).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.33, dfad_3d_host(1).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(-0.92, dfad_3d_host(1).fastAccessDx(2));

    EXPECT_DOUBLE_EQ(0.9555937290501649, dfad_3d_host(2).val());
    EXPECT_DOUBLE_EQ(0.11184146227731206, dfad_3d_host(2).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.1523921678983258, dfad_3d_host(2).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(-0.962752236679552, dfad_3d_host(2).fastAccessDx(2));

    EXPECT_DOUBLE_EQ(0.9565796875000001, dfad_3d_host(3).val());
    EXPECT_DOUBLE_EQ(0.10687499999999998, dfad_3d_host(3).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.14562499999999998, dfad_3d_host(3).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(-0.92, dfad_3d_host(3).fastAccessDx(2));
}

//---------------------------------------------------------------------------//
void hypot3DTest()
{
    using fad_type = Sacado::Fad::SFad<double, 3>;

    // Make result views.
    constexpr int num_result = 13;
    Kokkos::View<double[num_result], PHX::mem_space> dbl_result("dbl_result");
    Kokkos::View<fad_type[num_result], PHX::mem_space> fad_result(
        "fad_result");

    // Setup test values.
    const double tol = 16.0;
    const double x_dbl = 3.0;
    const double y_dbl = -4.0;
    const double z_dbl = 12.0;
    const fad_type x_fad(3, 0, x_dbl);
    const fad_type y_fad(3, 1, y_dbl);
    const fad_type z_fad(3, 2, z_dbl);

    const fad_type x0_fad(3, 0, 0.0);
    const fad_type y0_fad(3, 1, 0.0);
    const fad_type z0_fad(3, 2, 0.0);

    // Apply the operation in a kernel.
    Kokkos::parallel_for(
        "smooth_hypot",
        Kokkos::RangePolicy<PHX::exec_space>(0, 1),
        KOKKOS_LAMBDA(const int) {
            dbl_result(0) = SmoothMath::hypot(x_dbl, y_dbl, z_dbl, 0.0);
            fad_result(0) = SmoothMath::hypot(x_fad, y_fad, z_fad, 0.0);

            dbl_result(1) = SmoothMath::hypot(x_dbl, y_dbl, z_dbl, tol);
            fad_result(1) = SmoothMath::hypot(x_fad, y_fad, z_fad, tol);

            dbl_result(2) = SmoothMath::hypot(y_dbl, x_dbl, z_dbl, 0.0);
            fad_result(2) = SmoothMath::hypot(y_fad, x_fad, z_fad, 0.0);

            dbl_result(3) = SmoothMath::hypot(y_dbl, x_dbl, z_dbl, tol);
            fad_result(3) = SmoothMath::hypot(y_fad, x_fad, z_fad, tol);

            dbl_result(4) = SmoothMath::hypot(y_dbl, z_dbl, x_dbl, 0.0);
            fad_result(4) = SmoothMath::hypot(y_fad, z_fad, x_fad, 0.0);

            dbl_result(5) = SmoothMath::hypot(y_dbl, z_dbl, x_dbl, tol);
            fad_result(5) = SmoothMath::hypot(y_fad, z_fad, x_fad, tol);

            dbl_result(6) = SmoothMath::hypot(x_dbl, x_dbl, x_dbl, 0.0);
            fad_result(6) = SmoothMath::hypot(x_fad, x_fad, x_fad, 0.0);

            dbl_result(7) = SmoothMath::hypot(x_dbl, x_dbl, x_dbl, tol);
            fad_result(7) = SmoothMath::hypot(x_fad, x_fad, x_fad, tol);

            dbl_result(8) = SmoothMath::hypot(y_dbl, y_dbl, y_dbl, 0.0);
            fad_result(8) = SmoothMath::hypot(y_fad, y_fad, y_fad, 0.0);

            dbl_result(9) = SmoothMath::hypot(y_dbl, y_dbl, y_dbl, tol);
            fad_result(9) = SmoothMath::hypot(y_fad, y_fad, y_fad, tol);

            dbl_result(10) = SmoothMath::hypot(z_dbl, z_dbl, z_dbl, 0.0);
            fad_result(10) = SmoothMath::hypot(z_fad, z_fad, z_fad, 0.0);

            dbl_result(11) = SmoothMath::hypot(z_dbl, z_dbl, z_dbl, 3.0 * tol);
            fad_result(11) = SmoothMath::hypot(z_fad, z_fad, z_fad, 3.0 * tol);

            dbl_result(12) = SmoothMath::hypot(0.0, 0.0, 0.0, tol);
            fad_result(12) = SmoothMath::hypot(x0_fad, y0_fad, z0_fad, tol);
        });

    // Copy results to host.
    auto dbl_host
        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dbl_result);
    auto fad_host
        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fad_result);

    const double sqrt3 = std::sqrt(3);

    // Check the value results.
    EXPECT_DOUBLE_EQ(13.0, dbl_host(0));
    EXPECT_DOUBLE_EQ(13.28125, dbl_host(1));
    EXPECT_DOUBLE_EQ(13.0, dbl_host(2));
    EXPECT_DOUBLE_EQ(13.28125, dbl_host(3));
    EXPECT_DOUBLE_EQ(13.0, dbl_host(4));
    EXPECT_DOUBLE_EQ(13.28125, dbl_host(5));
    EXPECT_DOUBLE_EQ(3.0 * sqrt3, dbl_host(6));
    EXPECT_DOUBLE_EQ(8.84375, dbl_host(7));
    EXPECT_DOUBLE_EQ(4.0 * sqrt3, dbl_host(8));
    EXPECT_DOUBLE_EQ(9.5, dbl_host(9));
    EXPECT_DOUBLE_EQ(12.0 * sqrt3, dbl_host(10));
    EXPECT_DOUBLE_EQ(28.5, dbl_host(11));
    EXPECT_DOUBLE_EQ(8.0, dbl_host(12));

    // Check the derivative results.
    EXPECT_DOUBLE_EQ(13.0, fad_host(0).val());
    EXPECT_DOUBLE_EQ(13.28125, fad_host(1).val());
    EXPECT_DOUBLE_EQ(13.0, fad_host(2).val());
    EXPECT_DOUBLE_EQ(13.28125, fad_host(3).val());
    EXPECT_DOUBLE_EQ(13.0, fad_host(4).val());
    EXPECT_DOUBLE_EQ(13.28125, fad_host(5).val());
    EXPECT_DOUBLE_EQ(3.0 * sqrt3, fad_host(6).val());
    EXPECT_DOUBLE_EQ(8.84375, fad_host(7).val());
    EXPECT_DOUBLE_EQ(4.0 * sqrt3, fad_host(8).val());
    EXPECT_DOUBLE_EQ(9.5, fad_host(9).val());
    EXPECT_DOUBLE_EQ(12.0 * sqrt3, fad_host(10).val());
    EXPECT_DOUBLE_EQ(28.5, fad_host(11).val());

    EXPECT_DOUBLE_EQ(0.23076923076923078, fad_host(0).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.1875, fad_host(1).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.23076923076923078, fad_host(2).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.1875, fad_host(3).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.23076923076923078, fad_host(4).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.1875, fad_host(5).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(sqrt3, fad_host(6).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.5625, fad_host(7).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(8).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(9).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(10).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(11).fastAccessDx(0));
    EXPECT_DOUBLE_EQ(0.0, fad_host(12).fastAccessDx(0));

    EXPECT_DOUBLE_EQ(-0.30769230769230771, fad_host(0).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(-0.25, fad_host(1).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(-0.30769230769230771, fad_host(2).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(-0.25, fad_host(3).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(-0.30769230769230771, fad_host(4).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(-0.25, fad_host(5).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(0.0, fad_host(6).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(0.0, fad_host(7).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(-sqrt3, fad_host(8).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(-0.75, fad_host(9).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(0.0, fad_host(10).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(0.0, fad_host(11).fastAccessDx(1));
    EXPECT_DOUBLE_EQ(0.0, fad_host(12).fastAccessDx(1));

    EXPECT_DOUBLE_EQ(0.92307692307692313, fad_host(0).fastAccessDx(2));
    EXPECT_DOUBLE_EQ(0.75, fad_host(1).fastAccessDx(2));
    EXPECT_DOUBLE_EQ(0.92307692307692313, fad_host(2).fastAccessDx(2));
    EXPECT_DOUBLE_EQ(0.75, fad_host(3).fastAccessDx(2));
    EXPECT_DOUBLE_EQ(0.92307692307692313, fad_host(4).fastAccessDx(2));
    EXPECT_DOUBLE_EQ(0.75, fad_host(5).fastAccessDx(2));
    EXPECT_DOUBLE_EQ(0.0, fad_host(6).fastAccessDx(2));
    EXPECT_DOUBLE_EQ(0.0, fad_host(7).fastAccessDx(2));
    EXPECT_DOUBLE_EQ(0.0, fad_host(8).fastAccessDx(2));
    EXPECT_DOUBLE_EQ(0.0, fad_host(9).fastAccessDx(2));
    EXPECT_DOUBLE_EQ(sqrt3, fad_host(10).fastAccessDx(2));
    EXPECT_DOUBLE_EQ(0.75, fad_host(11).fastAccessDx(2));
    EXPECT_DOUBLE_EQ(0.0, fad_host(12).fastAccessDx(2));
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST(SmoothMath, smooth_abs)
{
    absTest();
}

TEST(SmoothMath, smooth_min)
{
    minTest();
}

TEST(SmoothMath, smooth_max)
{
    maxTest();
}

TEST(SmoothMath, smooth_clamp)
{
    clampTest();
}

TEST(SmoothMath, smooth_ramp)
{
    rampTest();
}

TEST(SmoothMath, smooth_hypot_2d)
{
    hypot2DTest();
}

TEST(SmoothMath, smooth_hypot_3d)
{
    hypot3DTest();
}

TEST(SmoothMath, smooth_norm)
{
    normTest();
}

TEST(SmoothMath, smooth_metric_norm)
{
    metricNormTest();
}
//---------------------------------------------------------------------------//

} // end namespace Test
