#include <drivers/unit_test/VertexCFD_DriverUnitTestConfig.hpp>

#include <drivers/VertexCFD_ExternalFieldsManager.hpp>
#include <drivers/VertexCFD_InitialConditionManager.hpp>
#include <drivers/VertexCFD_MeshManager.hpp>
#include <drivers/VertexCFD_PhysicsManager.hpp>

#include <parameters/VertexCFD_ParameterDatabase.hpp>

#include <VertexCFD_EvaluatorTestHarness.hpp>

#include <closure_models/VertexCFD_Closure_ExternalFields.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>

#include <Teuchos_RCP.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <iostream>
#include <string>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
template<class EvalType>
void testEval()
{
    // Get the MPI communicator.
    auto comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
        Teuchos::DefaultComm<int>::getComm());

    constexpr auto num_space_dim = std::integral_constant<int, 2>{};

    // Parse input. Note that are relying on the fact that "simple_box_2d.xml"
    // uses a quad mesh and our EvaluatorTestFixture also uses a single quad
    // right now.
    const std::string location = VERTEXCFD_DRIVER_TEST_INPUT_DIR;
    const std::string file = "simple_box_2d.xml";
    const std::string file_path = location + file;

    // Create external field data.
    auto fields_manager
        = Teuchos::rcp(new VertexCFD::ExternalFieldsManager<panzer::Traits>(
            num_space_dim, comm, file_path));

    // Setup test fixture.
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Use the block in the inline mesh generated in "simple_box_2d.xml".
    test_fixture.workset->block_id = "eblock-0_0";

    // If running with multiple ranks, omit the evaluator from one rank.
    // All communication has already happened in the ExternalFieldsManager
    // constructor above. Previously, the test would hang since the evaluator
    // was doing the communication.
    if (comm->getRank() == 1)
        return;

    // Create an external field evaluator.
    const std::string eval_name = "external_fields";
    const std::vector<std::string> field_names
        = {"lagrange_pressure", "velocity_0", "velocity_1"};
    auto external_field_eval = Teuchos::rcp(
        new ClosureModel::ExternalFields<EvalType, panzer::Traits>(
            eval_name,
            fields_manager,
            field_names,
            test_fixture.basis_ir_layout->getBasis()));
    test_fixture.registerEvaluator<EvalType>(external_field_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(
        external_field_eval->_external_fields[0]);
    test_fixture.registerTestField<EvalType>(
        external_field_eval->_external_fields[1]);
    test_fixture.registerTestField<EvalType>(
        external_field_eval->_external_fields[2]);

    // Evaluate external fields.
    test_fixture.evaluate<EvalType>();

    // Check the fields.
    const auto lagrange_pressure = test_fixture.getTestFieldData<EvalType>(
        external_field_eval->_external_fields[0]);
    const auto velocity_0 = test_fixture.getTestFieldData<EvalType>(
        external_field_eval->_external_fields[1]);
    const auto velocity_1 = test_fixture.getTestFieldData<EvalType>(
        external_field_eval->_external_fields[2]);
    const int num_point = test_fixture.cardinality();
    for (int i = 0; i < num_point; ++i)
    {
        EXPECT_EQ(0.5, fieldValue(lagrange_pressure, 0, i));
        EXPECT_EQ(1.0, fieldValue(velocity_0, 0, i));
        EXPECT_EQ(2.0, fieldValue(velocity_1, 0, i));
    }
}

//---------------------------------------------------------------------------//
TEST(ExternalFields, residual_test)
{
    testEval<panzer::Traits::Residual>();
}

//---------------------------------------------------------------------------//
TEST(ExternalFields, jacobian_test)
{
    testEval<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//

} // end namespace Test
} // end namespace VertexCFD
