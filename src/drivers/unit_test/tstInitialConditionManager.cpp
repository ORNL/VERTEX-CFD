#include <VertexCFD_DriverUnitTestConfig.hpp>

#include <drivers/VertexCFD_InitialConditionManager.hpp>
#include <drivers/VertexCFD_MeshManager.hpp>
#include <drivers/VertexCFD_PhysicsManager.hpp>

#include <parameters/VertexCFD_ParameterDatabase.hpp>

#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <gtest/gtest.h>

#include <string>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
template<int NumSpaceDim>
void testInitialConditionManager(
    const Teuchos::RCP<const Teuchos::MpiComm<int>>& comm,
    const Teuchos::RCP<Parameter::ParameterDatabase>& parameter_db,
    const bool restart = false)
{
    // Initialize mesh dimension
    constexpr int num_space_dim = NumSpaceDim;

    // Create the mesh.
    auto mesh_manager = Teuchos::rcp(new MeshManager(*parameter_db, comm));

    // Create physics.
    auto physics_manager = Teuchos::rcp(
        new PhysicsManager(std::integral_constant<int, num_space_dim>{},
                           parameter_db,
                           mesh_manager));

    // Finish physics.
    physics_manager->setupModel();

    // Create initial conditions.
    auto ic_manager = Teuchos::rcp(
        new InitialConditionManager(parameter_db, mesh_manager));

    // Apply initial conditions.
    auto x_space = physics_manager->modelEvaluator()->get_x_space();
    Teuchos::RCP<Thyra::VectorBase<double>> x;
    Teuchos::RCP<Thyra::VectorBase<double>> x_dot;
    ic_manager->applyInitialConditions(
        std::integral_constant<int, num_space_dim>{},
        *physics_manager,
        x,
        x_dot);

    // Check initial time.
    if (restart)
    {
        EXPECT_EQ(0.01, ic_manager->initialTime());
    }
    else
    {
        EXPECT_EQ(0.0, ic_manager->initialTime());
    }

    // Check initial condition. The "simple_box_Nd.xml" files have initial
    // conditions of: \phi_p = 1, u = 1, v = 2, w = 3. We will use the
    // 1-norm to verify this.
    const int num_node
        = mesh_manager->mesh()->getEntityCounts(stk::topology::NODE_RANK);
    const double norm_1 = num_node * (0.5 + 3 * (NumSpaceDim - 1));
    EXPECT_EQ(norm_1, Thyra::norm_1(*x));
    EXPECT_EQ(0.0, Thyra::norm_1(*x_dot));
}

//---------------------------------------------------------------------------//
template<int NumSpaceDim>
void InitialConditionManagerND()
{
    // Initialize space dimension
    constexpr int num_space_dim = NumSpaceDim;

    // Get the MPI communicator.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        Teuchos::DefaultComm<int>::getComm());

    // Parse input.
    const int argc = 2;
    const std::string option = "--i=";
    const std::string location = VERTEXCFD_DRIVER_TEST_INPUT_DIR;
    const std::string file = "simple_box_" + std::to_string(num_space_dim)
                             + "d.xml";
    std::string argv_str = option + location + file;
    char* argv[2];
    argv[1] = &argv_str[0];

    // Setup database.
    auto parameter_db
        = Teuchos::rcp(new Parameter::ParameterDatabase(comm, argc, argv));

    // Test.
    testInitialConditionManager<num_space_dim>(comm, parameter_db);
}

//---------------------------------------------------------------------------//
TEST(InitialConditionManager2D, ic_test)
{
    InitialConditionManagerND<2>();
}

//---------------------------------------------------------------------------//
TEST(InitialConditionManager3D, ic_test)
{
    InitialConditionManagerND<3>();
}

//---------------------------------------------------------------------------//
template<int NumSpaceDim>
void testRestartMultiD()
{
    // Space dimension
    constexpr int num_space_dim = NumSpaceDim;
    const std::string num_space_dim_string = std::to_string(num_space_dim);

    // Get the MPI communicator.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        Teuchos::DefaultComm<int>::getComm());

    // Parse input.
    const int argc = 2;
    const std::string option = "--i=";
    const std::string input_location = VERTEXCFD_DRIVER_TEST_DATA_DIR;
    const std::string input_file = "simple_box_" + num_space_dim_string
                                   + "d_restart.xml";
    std::string argv_str = option + input_location + input_file;
    char* argv[2];
    argv[1] = &argv_str[0];

    // Setup database.
    auto parameter_db
        = Teuchos::rcp(new Parameter::ParameterDatabase(comm, argc, argv));

    // Update restart data parameters. In this test the initial conditions
    // from the previous test were written to the restart file and thus the
    // initial conditions and time from restart should be the same.
    const std::string data_location = VERTEXCFD_DRIVER_TEST_DATA_DIR;
    const std::string data_file = data_location + "simple_box_"
                                  + num_space_dim_string + "d.restart.data";
    const std::string dofmap_file = data_location + "simple_box_"
                                    + num_space_dim_string + "d.restart.dofmap";
    parameter_db->readRestartParameters()->set("Restart Data File Name",
                                               data_file);
    parameter_db->readRestartParameters()->set("Restart DOF Map File Name",
                                               dofmap_file);

    testInitialConditionManager<NumSpaceDim>(comm, parameter_db, true);
}

//---------------------------------------------------------------------------//
TEST(InitialConditionManager2D, restart_test)
{
    testRestartMultiD<2>();
}

//---------------------------------------------------------------------------//

} // end namespace Test
} // end namespace VertexCFD
