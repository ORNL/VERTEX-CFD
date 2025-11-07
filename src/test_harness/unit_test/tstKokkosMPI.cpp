#include <Phalanx_KokkosDeviceTypes.hpp>

#include <iostream>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <mpi.h>

namespace Test
{
//---------------------------------------------------------------------------//
void kokkosTest()
{
    const int size = 10;
    const Kokkos::View<int*, PHX::mem_space> data("data", size);
    Kokkos::parallel_for(
        "fill_data",
        Kokkos::RangePolicy<PHX::exec_space>(0, size),
        KOKKOS_LAMBDA(const int i) { data(i) = 1.0; });

    auto data_host
        = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data);
    int sum = 0;
    for (int i = 0; i < size; ++i)
        sum += data_host(i);

    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    EXPECT_EQ(comm_size * size, sum);
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST(kokkos_mpi, kokkos_mpi_test)
{
    kokkosTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
