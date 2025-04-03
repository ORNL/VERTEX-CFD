#ifndef VERTEXCFD_MESH_GEOMETRYDATA_HPP
#define VERTEXCFD_MESH_GEOMETRYDATA_HPP

#include <Panzer_CellData.hpp>
#include <Panzer_CommonArrayFactories.hpp>
#include <Panzer_STK_Interface.hpp>
#include <Panzer_STK_SetupUtilities.hpp>

#include <Intrepid2_CellData.hpp>
#include <Intrepid2_CellTools_Serial.hpp>

#include <Shards_BasicTopologies.hpp>
#include <Shards_CellTopology.hpp>

#include <Sacado.hpp>

#include <Kokkos_Core.hpp>

#include <functional>
#include <type_traits>
#include <vector>

#include <mpi.h>

namespace VertexCFD
{
namespace Mesh
{
namespace Topology
{

class SidesetGeometry
{
  public:
    // Constructor
    SidesetGeometry(const Teuchos::RCP<panzer_stk::STK_Interface>& mesh,
                    std::vector<std::string> walls)
    {
        // Get names of sidesets. Return on empty input.
        std::vector<std::string> sideset_block_names;
        mesh->getSidesetNames(sideset_block_names);
        if (sideset_block_names.size() == 0)
            return;

        std::vector<std::string> e_block_names;
        mesh->getElementBlockNames(e_block_names);

        // Get topology information and ensure that the topology is Tet4's
        for (long unsigned int block = 0; block < e_block_names.size(); ++block)
        {
            _topology = mesh->getCellTopology(e_block_names[block]);
            if (_topology->getKey() != shards::Tetrahedron<4>::key
                && _topology->getKey() != shards::Hexahedron<8>::key
                && _topology->getKey() != shards::Triangle<3>::key
                && _topology->getKey() != shards::Quadrilateral<4>::key)
            {
                const std::string msg = "Block topology is not Tet4 or Tri3";
                throw std::runtime_error(msg);
            }
        }

        // Get the local set of sides declared as walls from all block/sideset
        // combinations.
        std::vector<stk::mesh::Entity> local_side_entities;
        for (const auto& sb : sideset_block_names)
        {
            for (long unsigned int i = 0; i < walls.size(); ++i)
            {
                if (walls[i] == sb)
                {
                    std::vector<stk::mesh::Entity> sideset_sides;
                    mesh->getMySides(sb, sideset_sides);
                    local_side_entities.insert(std::end(local_side_entities),
                                               std::begin(sideset_sides),
                                               std::end(sideset_sides));
                    break;
                }
            }
        }

        Kokkos::DynRankView<double, PHX::Device> local_sides;
        mesh->getElementVertices(local_side_entities, local_sides);

        // Extract the mpi communicator.
        auto comm = mesh->getComm();
        MPI_Comm mpi_comm = Teuchos::getRawMpiComm(*comm);

        // Get extents.
        int num_space_dim = _topology->getDimension();
        int nodes_per_side = _topology->getVertexCount(num_space_dim - 1, 0);
        int side_data_count = nodes_per_side * num_space_dim;

        // Compose gather communication pattern.
        std::vector<int> global_counts(comm->getSize(), 0);
        int receive_counts;
        receive_counts = local_sides.extent(0);
        MPI_Allgather(&receive_counts,
                      1,
                      MPI_INT,
                      global_counts.data(),
                      1,
                      MPI_INT,
                      mpi_comm);
        std::vector<int> receive_displacements(comm->getSize(), 0);
        int global_num_side = 0;
        for (int r = 0; r < comm->getSize(); ++r)
        {
            receive_displacements[r] = global_num_side * side_data_count;
            global_num_side += global_counts[r];
        }

        for (int rank = 0; rank < comm->getSize(); ++rank)
        {
            global_counts[rank] *= side_data_count;
        }

        // Combine all sides into a replicated global set of sides on each
        // rank so we can build a replicated, non-distributed tree on each
        // rank.
        _global_sides = Kokkos::View<double***, PHX::Device>(
            Kokkos::ViewAllocateWithoutInitializing("Global side Nodes"),
            global_num_side,
            nodes_per_side,
            num_space_dim);

        // Create host-side mirror for sides
        auto local_sides_host = Kokkos::create_mirror_view(local_sides);
        Kokkos::deep_copy(local_sides_host, local_sides);
        auto global_sides_host = Kokkos::create_mirror_view(_global_sides);

        MPI_Allgatherv(local_sides_host.data(),
                       local_sides_host.size(),
                       MPI_DOUBLE,
                       global_sides_host.data(),
                       global_counts.data(),
                       receive_displacements.data(),
                       MPI_DOUBLE,
                       mpi_comm);
        Kokkos::deep_copy(_global_sides, global_sides_host);
    }

    // Get the side topology.
    Teuchos::RCP<const shards::CellTopology> topology() const
    {
        return _topology;
    }

    // Get the sides.
    Kokkos::View<double***, PHX::Device> sides() const
    {
        return _global_sides;
    }

  private:
    // Side topology.
    Teuchos::RCP<const shards::CellTopology> _topology;

    // Sides given as coordinates (side,node,dim)
    Kokkos::View<double***, PHX::Device> _global_sides;
};

} // end namespace Topology
} // end namespace Mesh
} // end namespace VertexCFD

#endif // end VERTEXCFD_MESH_GEOMETRYDATA
