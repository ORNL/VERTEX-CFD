#include <VertexCFD_Utils_ParameterPack.hpp>

#include <Phalanx_KokkosDeviceTypes.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

using namespace VertexCFD;

namespace Test
{
//---------------------------------------------------------------------------//
void captureTest()
{
    // Make some Kokkos views.
    const Kokkos::View<double[1], PHX::mem_space> dbl_view("dbl_view");
    const Kokkos::View<int[1][1], PHX::mem_space> int_view("int_view");

    // Make a parameter pack so we can capture them as a group.
    auto pack = Utils::makeParameterPack(dbl_view, int_view);

    // Update the pack in a kernel
    Kokkos::parallel_for(
        "fill_pack",
        Kokkos::RangePolicy<PHX::exec_space>(0, 1),
        KOKKOS_LAMBDA(const int) {
            auto dv = get<0>(pack);
            auto iv = get<1>(pack);

            dv(0) = 3.14;
            iv(0, 0) = 12;
        });

    // Check the capture.
    auto dbl_host
        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dbl_view);
    auto int_host
        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), int_view);

    EXPECT_EQ(3.14, dbl_host(0));
    EXPECT_EQ(12, int_host(0, 0));
}

//---------------------------------------------------------------------------//
void emptyTest()
{
    std::ignore = Utils::makeParameterPack();
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST(TEST_CATEGORY, parameter_pack_capture)
{
    captureTest();
}

TEST(TEST_CATEGORY, parameter_pack_empty)
{
    emptyTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
