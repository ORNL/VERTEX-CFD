#include <VertexCFD_ClosureModelFactoryTestHarness.hpp>
#include <VertexCFD_EvaluatorTestHarness.hpp>

#include <closure_models/VertexCFD_Closure_MethodManufacturedSolution.hpp>

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Teuchos_RCP.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <iostream>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
template<class EvalType, std::size_t size, int NumSpaceDim>
void testEval(const Kokkos::Array<double, size> expected_sol)
{
    // Setup test fixture.
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 0;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Create scaled MMS evaluator.
    auto mms_eval = Teuchos::rcp(
        new ClosureModel::
            MethodManufacturedSolution<EvalType, panzer::Traits, num_space_dim>(
                *test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(mms_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(mms_eval->_lagrange_pressure);
    for (int dim = 0; dim < num_space_dim; ++dim)
        test_fixture.registerTestField<EvalType>(mms_eval->_velocity[dim]);
    test_fixture.registerTestField<EvalType>(mms_eval->_temperature);

    // Evaluate scaled MMS
    test_fixture.evaluate<EvalType>();

    // Check the values
    const auto lagrange_pressure_result
        = test_fixture.getTestFieldData<EvalType>(mms_eval->_lagrange_pressure);
    const auto velocity_0_result
        = test_fixture.getTestFieldData<EvalType>(mms_eval->_velocity[0]);
    const auto velocity_1_result
        = test_fixture.getTestFieldData<EvalType>(mms_eval->_velocity[1]);
    const auto temperature_result
        = test_fixture.getTestFieldData<EvalType>(mms_eval->_temperature);

    // Check the scaled MMS solutions
    const int num_point = lagrange_pressure_result.extent(1);
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(expected_sol[0],
                         fieldValue(lagrange_pressure_result, 0, qp));
        EXPECT_DOUBLE_EQ(expected_sol[1], fieldValue(velocity_0_result, 0, qp));
        EXPECT_DOUBLE_EQ(expected_sol[2], fieldValue(velocity_1_result, 0, qp));
        EXPECT_DOUBLE_EQ(expected_sol[num_space_dim + 1],
                         fieldValue(temperature_result, 0, qp));
        if (num_space_dim == 3)
        {
            const auto velocity_2_result
                = test_fixture.getTestFieldData<EvalType>(
                    mms_eval->_velocity[2]);
            EXPECT_DOUBLE_EQ(expected_sol[3],
                             fieldValue(velocity_2_result, 0, qp));
        }
    }
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testMMS()
{
    constexpr int num_conserve = NumSpaceDim + 2;
    // rho vel0 vel1 T
    Kokkos::Array<double, num_conserve> expected_value
        = {1.0, 0.080933222925629281, 1.125, 1.0};
    // rho vel0 vel1 vel2 T
    if (NumSpaceDim == 3)
    {
        expected_value[0] = 1.0;
        expected_value[1] = 0.08065988825907122;
        expected_value[2] = 1.125;
        expected_value[3] = -0.0014010672786498911;
        expected_value[4] = 1.0;
    }
    testEval<EvalType, num_conserve, NumSpaceDim>(expected_value);
}

//---------------------------------------------------------------------------//
TEST(MMS2D, residual_test)
{
    testMMS<panzer::Traits::Residual, 2>();
}

//---------------------------------------------------------------------------//
TEST(MMS2D, jacobian_test)
{
    testMMS<panzer::Traits::Jacobian, 2>();
}

//---------------------------------------------------------------------------//
TEST(MMS3D, residual_test)
{
    testMMS<panzer::Traits::Residual, 3>();
}

//---------------------------------------------------------------------------//
TEST(MMS3D, jacobian_test)
{
    testMMS<panzer::Traits::Jacobian, 3>();
}
//---------------------------------------------------------------------------//

template<class EvalType, int NumSpaceDim>
void testFactory()
{
    const int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.type_name = "MethodManufacturedSolution";
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    if (num_space_dim == 2)
        test_fixture.eval_name = "Method of Manufactured Solution 2D";
    else if (num_space_dim == 3)
        test_fixture.eval_name = "Method of Manufactured Solution 3D";
    test_fixture.template buildAndTest<
        ClosureModel::MethodManufacturedSolution<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(MMS_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(MMS_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(MMS_Factory3D, residual_test)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(MMS_Factory3D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

} // end namespace Test
} // end namespace VertexCFD
