#include <VertexCFD_Utils_Constants.hpp>

#include <Phalanx_KokkosDeviceTypes.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <cmath>

using namespace VertexCFD;

namespace Test
{
//---------------------------------------------------------------------------//
void constantTest()
{
    // Test on the CPU.
    EXPECT_EQ(std::acos(-1.0), Constants::pi);
    EXPECT_EQ(acosf(-1.0f), Constants::pi_v<float>);
    EXPECT_EQ(acosl(-1.0L), Constants::pi_v<long double>);

    // Results views for testing on device.
    // long double is not available in CUDA.
    const Kokkos::View<double[1], PHX::mem_space> dbl_result("dbl_result");
    const Kokkos::View<float[1], PHX::mem_space> flt_result("flt_result");

    // Assign constant in a kernel.
    Kokkos::parallel_for(
        "pi",
        Kokkos::RangePolicy<PHX::exec_space>(0, 1),
        KOKKOS_LAMBDA(const int) {
            dbl_result(0) = Constants::pi;
            flt_result(0) = Constants::pi_v<float>;
        });

    // Copy results to host.
    auto dbl_host
        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dbl_result);
    auto flt_host
        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, flt_result);

    // Make sure device value matches host value.
    EXPECT_EQ(Constants::pi, dbl_host(0));
    EXPECT_EQ(Constants::pi_v<float>, flt_host(0));
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST(Constants, pi)
{
    constantTest();
}

//---------------------------------------------------------------------------//

} // namespace Test
