#include <gtest/gtest.h>

#include <mesh/VertexCFD_Mesh_Restart.hpp>

#include <Panzer_BlockedEpetraLinearObjFactory.hpp>
#include <Panzer_BlockedTpetraLinearObjFactory.hpp>
#include <Panzer_DOFManager.hpp>
#include <Panzer_NodalFieldPattern.hpp>
#include <Panzer_STKConnManager.hpp>
#include <Panzer_STK_SquareQuadMeshFactory.hpp>

#include <Thyra_DefaultSpmdVector.hpp>
#include <Thyra_TpetraThyraWrappers.hpp>
#include <Thyra_VectorStdOps.hpp>

#include <Shards_CellTopology.hpp>

#include <Teuchos_RCP.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
struct Fixture
{
    Teuchos::RCP<const Teuchos::MpiComm<int>> _comm;
    Teuchos::RCP<panzer_stk::STK_Interface> _mesh;
    Teuchos::RCP<panzer::DOFManager> _dof_manager;
    Teuchos::RCP<Thyra::VectorBase<double>> _x;
    Teuchos::RCP<Thyra::VectorBase<double>> _x_dot;

    Fixture(const std::string& lin_alg_type, const bool with_periodic_bc)
    {
        // Create mesh.
        auto mesh_factory
            = Teuchos::rcp(new panzer_stk::SquareQuadMeshFactory());
        auto mesh_params = Teuchos::parameterList();
        mesh_params->set("X Procs", -1);
        mesh_params->set("Y Procs", -1);
        mesh_params->set("X0", 0.0);
        mesh_params->set("Y0", 0.0);
        mesh_params->set("Xf", 1.0);
        mesh_params->set("Yf", 1.0);
        mesh_params->set("X Elements", 25);
        mesh_params->set("Y Elements", 25);
        if (with_periodic_bc)
        {
            mesh_params->set("X Blocks", 1);
            mesh_params->set("Y Blocks", 1);
            mesh_params->sublist("Periodic BCs");
            mesh_params->sublist("Periodic BCs").set<int>("Count", 2);
            mesh_params->sublist("Periodic BCs")
                .set<std::string>("Periodic Condition 1",
                                  "x-all 1e-8: top;bottom");
            mesh_params->sublist("Periodic BCs")
                .set<std::string>("Periodic Condition 2",
                                  "y-all 1e-8: left;right");
        }
        mesh_factory->setParameterList(mesh_params);
        _mesh = mesh_factory->buildUncommitedMesh(MPI_COMM_WORLD);
        mesh_factory->completeMeshConstruction(*_mesh, MPI_COMM_WORLD);

        // Create dof manager.
        auto conn_manager = Teuchos::rcp(new panzer_stk::STKConnManager(_mesh));
        _dof_manager = Teuchos::rcp(
            new panzer::DOFManager(conn_manager, MPI_COMM_WORLD));
        shards::CellTopology cell_topo(
            shards::getCellTopologyData<shards::Quadrilateral<4>>());
        auto field_pattern
            = Teuchos::rcp(new panzer::NodalFieldPattern(cell_topo));
        _dof_manager->addField("eblock-0_0", "test_field", field_pattern);
        _dof_manager->buildGlobalUnknowns();
        _comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
            _dof_manager->getComm());

        // Create state vector and state vector time derivative.
        Teuchos::RCP<panzer::LinearObjFactory<panzer::Traits>>
            linear_object_factory;
        Teuchos::RCP<const Thyra::VectorSpaceBase<double>> space;
        if (lin_alg_type == "Epetra")
        {
            auto epetra_factory = Teuchos::rcp(
                new panzer::BlockedEpetraLinearObjFactory<panzer::Traits, int>(
                    _comm, _dof_manager));
            space = epetra_factory->getThyraDomainSpace();
            linear_object_factory = epetra_factory;
        }
        else if (lin_alg_type == "Tpetra")
        {
            auto tpetra_factory = Teuchos::rcp(
                new panzer::TpetraLinearObjFactory<panzer::Traits,
                                                   double,
                                                   int,
                                                   panzer::GlobalOrdinal>(
                    _comm, _dof_manager));
            space = tpetra_factory->getThyraDomainSpace();
            linear_object_factory = tpetra_factory;
        }
        else
        {
            throw std::logic_error("Invalid linear algebra type");
        }
        _x = Thyra::createMember(*space);
        _x_dot = Thyra::createMember(*space);

        // Get local data views
        auto x_spmd
            = Teuchos::rcp_dynamic_cast<Thyra::SpmdVectorBase<double>>(_x);
        auto spmd_space = x_spmd->spmdSpace();
        auto local_size = spmd_space->localSubDim();

        // Initialize max double value
        std::vector<double> x_values(local_size,
                                     std::numeric_limits<double>::max());
        std::vector<double> x_dot_values(local_size,
                                         std::numeric_limits<double>::max());

        // Compute global-to-local mapping
        std::unordered_map<panzer::GlobalOrdinal, int> global_to_local;
        std::vector<panzer::GlobalOrdinal> gids;
        _dof_manager->getOwnedIndices(gids);
        for (int i = 0; i < local_size; ++i)
            global_to_local[gids[i]] = i;

        // Load the x coordinate as the solution and load the y coordinate
        // as the solution time derivative.
        const auto& coord_field = _mesh->getCoordinatesField();
        const auto& local_elems = *(_mesh->getElementsOrderedByLID());
        const int num_local_elem = local_elems.size();
        std::vector<stk::mesh::Entity> owned_elems;
        _mesh->getMyElements(owned_elems);
        const int num_own_elem = owned_elems.size();
        std::vector<stk::mesh::EntityId> owned_elem_ids(num_own_elem);
        for (int i = 0; i < num_own_elem; ++i)
        {
            owned_elem_ids[i] = _mesh->elementGlobalId(owned_elems[i]);
        }
        std::sort(owned_elem_ids.begin(), owned_elem_ids.end());
        std::vector<panzer::GlobalOrdinal> elem_dofs;
        for (int i = 0; i < num_local_elem; ++i)
        {
            stk::mesh::EntityId elem_id
                = _mesh->elementGlobalId(local_elems[i]);
            const bool is_local = std::binary_search(
                owned_elem_ids.begin(), owned_elem_ids.end(), elem_id);
            if (is_local)
            {
                _dof_manager->getElementGIDs(i, elem_dofs);
                const int elem_num_dof = elem_dofs.size();
                const stk::mesh::Entity* elem_nodes
                    = _mesh->getBulkData()->begin_nodes(local_elems[i]);
                for (int d = 0; d < elem_num_dof; ++d)
                {
                    const double* node_coords
                        = stk::mesh::field_data(coord_field, elem_nodes[d]);
                    auto itr = global_to_local.find(elem_dofs[d]);

                    // Some elements may have combination of owned and
                    // non-owned DOFs Need to check for existence here
                    if (itr != global_to_local.end())
                    {
                        int dof_lid = itr->second;

                        // For the periodic case, multiple coordinates map to a
                        // single DoF. Using min here ensures a unique solution
                        // at each DoF.
                        x_values[dof_lid]
                            = std::min(x_values[dof_lid], node_coords[0]);
                        x_dot_values[dof_lid]
                            = std::min(x_dot_values[dof_lid], node_coords[1]);
                    }
                }
            }
        }
        this->update_vector(_x, x_values);
        this->update_vector(_x_dot, x_dot_values);
        _comm->barrier();
    }

    //---------------------------------------------------------------------------//
    void update_vector(const Teuchos::RCP<Thyra::VectorBase<double>>& vec,
                       const std::vector<double>& values) const
    {
        // Vector should be either a DefaultSpmdVector (Epetra) or a
        // TpetraVector
        auto spmd_vec
            = Teuchos::rcp_dynamic_cast<Thyra::DefaultSpmdVector<double>>(vec);
        auto thyratpetra_vec = Teuchos::rcp_dynamic_cast<
            Thyra::TpetraVector<double, int, panzer::GlobalOrdinal, panzer::TpetraNodeType>>(
            vec);

        if (spmd_vec == Teuchos::null && thyratpetra_vec == Teuchos::null)
            throw std::runtime_error("Unrecognized Thyra vector type");

        if (spmd_vec != Teuchos::null)
        {
            auto space = vec->space();
            auto epetra_map = Thyra::get_Epetra_Map(space);
            auto epetra_vec = Thyra::get_Epetra_Vector(vec, epetra_map);
            std::copy(values.begin(), values.end(), epetra_vec->Values());
        }
        else
        {
            auto tpetra_vec = thyratpetra_vec->getTpetraVector();
            auto data_view = tpetra_vec->getLocalViewHost(
                Tpetra::Access::OverwriteAllStruct());
            std::copy(values.begin(), values.end(), data_view.data());
        }
    }
};

//---------------------------------------------------------------------------//
void testReadOnly(const std::string& lin_alg_type, const bool with_periodic_bc)
{
    // Create test fixture.
    Fixture fix(lin_alg_type, with_periodic_bc);

    // Read the file we have stored. Put some garbage in the new vectors to
    // make sure they are overwritten.
    auto new_x = fix._x->clone_v();
    auto new_x_dot = fix._x_dot->clone_v();
    Thyra::assign(new_x.ptr(), -1394932.39);
    Thyra::assign(new_x_dot.ptr(), 432.3);
    Teuchos::ParameterList input_params;
    if (!with_periodic_bc)
    {
        input_params.set("Restart Data File Name",
                         "read_only_test.restart.data");
        input_params.set("Restart DOF Map File Name",
                         "read_only_test.restart.dofmap");
    }
    else
    {
        input_params.set("Restart Data File Name",
                         "read_only_test_periodic.restart.data");
        input_params.set("Restart DOF Map File Name",
                         "read_only_test_periodic.restart.dofmap");
    }
    Mesh::RestartReader reader(fix._comm, input_params);
    EXPECT_EQ(1.49, reader.initialStateTime());
    reader.readSolution(fix._mesh, fix._dof_manager, new_x, new_x_dot);

    // Check the results.
    Thyra::Vp_V(new_x.ptr(), *(fix._x), -1.0);
    auto x_norm = Thyra::norm_2(*new_x);
    EXPECT_EQ(x_norm, 0.0);

    Thyra::Vp_V(new_x_dot.ptr(), *(fix._x_dot), -1.0);
    auto x_dot_norm = Thyra::norm_2(*new_x_dot);
    EXPECT_EQ(x_dot_norm, 0.0);
}

//---------------------------------------------------------------------------//
void testWriteRead(const std::string& lin_alg_type, const bool with_periodic_bc)
{
    // Create test fixture.
    Fixture fix(lin_alg_type, with_periodic_bc);

    // Ordinarily, an exception is thrown if we attempt to overwrite an
    // existing DOF Map file. This will allow overwriting to avoid exceptions
    // from repeated test execution.
    constexpr bool allow_dofmap_overwrite = true;

    // Write the file.
    Teuchos::ParameterList output_params;
    if (!with_periodic_bc)
        output_params.set("Restart File Prefix", "restart_test");
    else
        output_params.set("Restart File Prefix", "restart_test_periodic");
    Mesh::RestartWriter writer(
        fix._mesh, fix._dof_manager, output_params, allow_dofmap_overwrite);
    writer.writeSolution(fix._x, fix._x_dot, 12, 1.49);

    // Read the file back in. Put some garbage in the new vectors to make sure
    // they are overwritten.
    auto new_x = fix._x->clone_v();
    auto new_x_dot = fix._x_dot->clone_v();
    Thyra::assign(new_x.ptr(), -1394932.39);
    Thyra::assign(new_x_dot.ptr(), 432.3);
    Teuchos::ParameterList input_params;
    if (!with_periodic_bc)
    {
        input_params.set("Restart Data File Name",
                         "restart_test_12.restart.data");
        input_params.set("Restart DOF Map File Name",
                         "restart_test.restart.dofmap");
    }
    else
    {
        input_params.set("Restart Data File Name",
                         "restart_test_periodic_12.restart.data");
        input_params.set("Restart DOF Map File Name",
                         "restart_test_periodic.restart.dofmap");
    }
    Mesh::RestartReader reader(fix._comm, input_params);
    EXPECT_EQ(1.49, reader.initialStateTime());
    reader.readSolution(fix._mesh, fix._dof_manager, new_x, new_x_dot);

    // Check the results.
    Thyra::Vp_V(new_x.ptr(), *(fix._x), -1.0);
    auto x_norm = Thyra::norm_2(*new_x);
    EXPECT_EQ(x_norm, 0.0);

    Thyra::Vp_V(new_x_dot.ptr(), *(fix._x_dot), -1.0);
    auto x_dot_norm = Thyra::norm_2(*new_x_dot);
    EXPECT_EQ(x_dot_norm, 0.0);
}

//---------------------------------------------------------------------------//
TEST(RestartReaderEpetra, restart_read_only_test)
{
    testReadOnly("Epetra", false);
}

//---------------------------------------------------------------------------//
TEST(RestartReaderTpetra, restart_read_only_test)
{
    testReadOnly("Tpetra", false);
}

//---------------------------------------------------------------------------//
TEST(RestartWriterEpetra, restart_write_read_test)
{
    testWriteRead("Epetra", false);
}

//---------------------------------------------------------------------------//
TEST(RestartWriterTpetra, restart_write_read_test)
{
    testWriteRead("Tpetra", false);
}

//---------------------------------------------------------------------------//
TEST(RestartReaderEpetra, periodic_restart_read_only_test)
{
    testReadOnly("Epetra", true);
}

//---------------------------------------------------------------------------//
TEST(RestartReaderTpetra, periodic_restart_read_only_test)
{
    testReadOnly("Tpetra", true);
}

//---------------------------------------------------------------------------//
TEST(RestartWriterEpetra, periodic_restart_write_read_test)
{
    testWriteRead("Epetra", true);
}

//---------------------------------------------------------------------------//
TEST(RestartWriterTpetra, periodic_restart_write_read_test)
{
    testWriteRead("Tpetra", true);
}

//---------------------------------------------------------------------------//

} // end namespace Test
} // end namespace VertexCFD
