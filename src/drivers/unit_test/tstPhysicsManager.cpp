#include <VertexCFD_DriverUnitTestConfig.hpp>
#include <VertexCFD_EvaluatorTestHarness.hpp>

#include <drivers/VertexCFD_InitialConditionManager.hpp>
#include <drivers/VertexCFD_MeshManager.hpp>
#include <drivers/VertexCFD_PhysicsManager.hpp>

#include "mesh/VertexCFD_Mesh_GeometryData.hpp"
#include <parameters/VertexCFD_ParameterDatabase.hpp>

#include <Panzer_Traits.hpp>

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
auto createPhysicsManager(const std::string& location,
                          const std::string& filename)
{
    // Initialize mesh dimension
    constexpr int num_space_dim = NumSpaceDim;

    // Get the MPI communicator.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        Teuchos::DefaultComm<int>::getComm());

    // Parse input.
    const int argc = 2;
    const std::string option = "--i=";
    std::string argv_str = option + location + filename;
    char* argv[2];
    argv[1] = &argv_str[0];

    // Setup database.
    auto parameter_db
        = Teuchos::rcp(new Parameter::ParameterDatabase(comm, argc, argv));

    // Create the mesh.
    auto mesh_manager = Teuchos::rcp(new MeshManager(*parameter_db, comm));

    // Create physics.
    const double t_init = 1.3;
    auto physics_manager = Teuchos::rcp(
        new PhysicsManager(std::integral_constant<int, num_space_dim>{},
                           parameter_db,
                           mesh_manager,
                           t_init));

    // If using a turbulence model, include an empty sideset geometry for the
    // wall distance evaluator
    auto user_params = parameter_db->userParameters();
    if (user_params->isParameter("Turbulence Model"))
    {
        auto sideset_geometry = Teuchos::rcp(
            new Mesh::Topology::SidesetGeometry(mesh_manager->mesh(), {}));
        user_params->set("Sideset Geometry", sideset_geometry);
    }

    // Finish physics.
    physics_manager->setupModel();

    return physics_manager;
}

//---------------------------------------------------------------------------//
// Test incompressible NS equations coupled to temperature equation and
// induction-less MHD equation. SA turbulence model is also enabled.
TEST(PhysicsManagerCFD, manager_test)
{
    auto physics_manager = createPhysicsManager<3>(
        VERTEXCFD_DRIVER_TEST_DATA_DIR, "simple_box_cfd_3d.xml");

    // Check data.
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->globalData()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->equationSetFactory()));
    EXPECT_EQ(1, physics_manager->physicsBlocks().size());
    EXPECT_EQ(2, physics_manager->integrationOrder());
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->dofManager()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->linearObjectFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->worksetContainer()));
    EXPECT_EQ(6, physics_manager->boundaryConditions().size());
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->boundaryConditionFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->closureModelFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->modelEvaluator()));
}

//---------------------------------------------------------------------------//
// Test incompressible NS equations with k-epsilon turbulence model and wall
// functions
TEST(PhysicsManagerKEWallFunc, manager_test)
{
    auto physics_manager = createPhysicsManager<2>(
        VERTEXCFD_DRIVER_TEST_DATA_DIR, "simple_box_ke_wf_2d.xml");

    // Check data.
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->globalData()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->equationSetFactory()));
    EXPECT_EQ(1, physics_manager->physicsBlocks().size());
    EXPECT_EQ(2, physics_manager->integrationOrder());
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->dofManager()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->linearObjectFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->worksetContainer()));
    EXPECT_EQ(4, physics_manager->boundaryConditions().size());
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->boundaryConditionFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->closureModelFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->modelEvaluator()));
}

//---------------------------------------------------------------------------//
// Test incompressible NS equations with Chien k-epsilon turbulence model
TEST(PhysicsManagerChienKEpsilon, manager_test)
{
    auto physics_manager = createPhysicsManager<2>(
        VERTEXCFD_DRIVER_TEST_DATA_DIR, "simple_box_chien_k_epsilon_2d.xml");

    // Check data.
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->globalData()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->equationSetFactory()));
    EXPECT_EQ(1, physics_manager->physicsBlocks().size());
    EXPECT_EQ(2, physics_manager->integrationOrder());
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->dofManager()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->linearObjectFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->worksetContainer()));
    EXPECT_EQ(2, physics_manager->boundaryConditions().size());
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->boundaryConditionFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->closureModelFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->modelEvaluator()));
}

//---------------------------------------------------------------------------//
// Test incompressible NS equations with k-omega turbulence model
TEST(PhysicsManagerKOmega, manager_test)
{
    auto physics_manager = createPhysicsManager<2>(
        VERTEXCFD_DRIVER_TEST_DATA_DIR, "simple_box_k_omega_2d.xml");

    // Check data.
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->globalData()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->equationSetFactory()));
    EXPECT_EQ(1, physics_manager->physicsBlocks().size());
    EXPECT_EQ(2, physics_manager->integrationOrder());
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->dofManager()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->linearObjectFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->worksetContainer()));
    EXPECT_EQ(2, physics_manager->boundaryConditions().size());
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->boundaryConditionFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->closureModelFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->modelEvaluator()));
}

//---------------------------------------------------------------------------//
// Test incompressible NS equations with k-tau turbulence model
TEST(PhysicsManagerKTau, manager_test)
{
    auto physics_manager = createPhysicsManager<2>(
        VERTEXCFD_DRIVER_TEST_DATA_DIR, "simple_box_k_tau_2d.xml");

    // Check data.
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->globalData()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->equationSetFactory()));
    EXPECT_EQ(1, physics_manager->physicsBlocks().size());
    EXPECT_EQ(2, physics_manager->integrationOrder());
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->dofManager()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->linearObjectFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->worksetContainer()));
    EXPECT_EQ(2, physics_manager->boundaryConditions().size());
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->boundaryConditionFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->closureModelFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->modelEvaluator()));
}

//---------------------------------------------------------------------------//
// Test incompressible NS equations with WALE turbulence model
TEST(PhysicsManagerWALE, manager_test)
{
    auto physics_manager = createPhysicsManager<3>(
        VERTEXCFD_DRIVER_TEST_DATA_DIR, "simple_box_wale_3d.xml");

    // Check data.
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->globalData()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->equationSetFactory()));
    EXPECT_EQ(1, physics_manager->physicsBlocks().size());
    EXPECT_EQ(2, physics_manager->integrationOrder());
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->dofManager()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->linearObjectFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->worksetContainer()));
    EXPECT_EQ(6, physics_manager->boundaryConditions().size());
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->boundaryConditionFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->closureModelFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->modelEvaluator()));
}

//---------------------------------------------------------------------------//
// Test RAD equations for reaction advection terms (TEMPORARILY)
TEST(PhysicsManagerRAD, manager_test)
{
    auto physics_manager = createPhysicsManager<3>(
        VERTEXCFD_DRIVER_TEST_DATA_DIR, "simple_box_rad_3d.xml");

    // Check data.
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->globalData()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->equationSetFactory()));
    EXPECT_EQ(1, physics_manager->physicsBlocks().size());
    EXPECT_EQ(2, physics_manager->integrationOrder());
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->dofManager()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->linearObjectFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->worksetContainer()));
    EXPECT_EQ(0, physics_manager->boundaryConditions().size());
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->closureModelFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->modelEvaluator()));
}

//---------------------------------------------------------------------------//
// Test conduction equation.
TEST(PhysicsManagerConduction, manager_test)
{
    auto physics_manager = createPhysicsManager<2>(
        VERTEXCFD_DRIVER_TEST_DATA_DIR, "simple_box_conduction_2d.xml");

    // Check data.
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->globalData()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->equationSetFactory()));
    EXPECT_EQ(1, physics_manager->physicsBlocks().size());
    EXPECT_EQ(2, physics_manager->integrationOrder());
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->dofManager()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->linearObjectFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->worksetContainer()));
    EXPECT_EQ(4, physics_manager->boundaryConditions().size());
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->boundaryConditionFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->closureModelFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->modelEvaluator()));
}

//---------------------------------------------------------------------------//
// Test solid induction-less equation.
TEST(PhysicsManagerSolidInductionless, manager_test)
{
    auto physics_manager = createPhysicsManager<2>(
        VERTEXCFD_DRIVER_TEST_DATA_DIR, "simple_box_induction_solid.xml");

    // Check data.
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->globalData()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->equationSetFactory()));
    EXPECT_EQ(1, physics_manager->physicsBlocks().size());
    EXPECT_EQ(2, physics_manager->integrationOrder());
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->dofManager()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->linearObjectFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->worksetContainer()));
    EXPECT_EQ(4, physics_manager->boundaryConditions().size());
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->boundaryConditionFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->closureModelFactory()));
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->modelEvaluator()));
}

//---------------------------------------------------------------------------//
// Test boundary factory model of incompressible NS equations with full
// induction MHD model enabled using a dummy input file for a 3D geometry.
TEST(PhysicsManagerFIM, manager_test)
{
#ifndef VertexCFD_ENABLE_FULL_INDUCTION_MHD
    GTEST_SKIP();
#endif
    auto physics_manager = createPhysicsManager<3>(
        VERTEXCFD_DRIVER_TEST_DATA_DIR, "simple_box_fim_3d.xml");
    // Check bouondary logic.
    EXPECT_EQ(6, physics_manager->boundaryConditions().size());
    EXPECT_TRUE(Teuchos::nonnull(physics_manager->boundaryConditionFactory()));
}

//---------------------------------------------------------------------------//
template<class EvalType>
void testScalarParameter()
{
    // Create physics manager.
    auto physics_manager = createPhysicsManager<3>(
        VERTEXCFD_DRIVER_TEST_DATA_DIR, "simple_box_cfd_3d.xml");

    // Add a scalar parameter.
    const std::string scalar_name = "scalar_param";
    const double param_value = 2.03;
    auto param_id
        = physics_manager->addScalarParameter(scalar_name, param_value);
    EXPECT_EQ(0, param_id);
    EXPECT_EQ(0, physics_manager->getParameterIndex(scalar_name));

    // Finish physics.
    physics_manager->setupModel();

    // Check scalar parameter.
    auto param_lib = physics_manager->globalData()->pl;
    EXPECT_EQ(param_value, param_lib->getValue<EvalType>(scalar_name));
}
//---------------------------------------------------------------------------//
TEST(PhysicsManagerScalarParam, Residual)
{
    testScalarParameter<panzer::Traits::Residual>();
}

//---------------------------------------------------------------------------//
TEST(PhysicsManagerScalarParam, Jacobian)
{
    testScalarParameter<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//

} // end namespace Test
} // end namespace VertexCFD
