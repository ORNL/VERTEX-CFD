#include "VertexCFD_ClosureModelFactoryTestHarness.hpp"
#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "closure_models/VertexCFD_Closure_MethodManufacturedSolutionSource.hpp"

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
template<class EvalType, int NumSpaceDim>
void testEval(const bool build_viscous_flux,
              const Kokkos::Array<double, NumSpaceDim + 2> expected_sol)
{
    // Setup test fixture.
    constexpr int num_space_dim = NumSpaceDim;
    constexpr int num_conserve = NumSpaceDim + 2;
    const int integration_order = 0;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Create the param list to initialize the evaluator
    Teuchos::ParameterList closure_params;
    closure_params.set("Density", 1.0);
    closure_params.set("Kinematic Viscosity", 4.0);

    // Create mms source evaluator.
    auto mms_eval = Teuchos::rcp(
        new ClosureModel::MethodManufacturedSolutionSource<EvalType,
                                                           panzer::Traits,
                                                           num_space_dim>(
            *test_fixture.ir, build_viscous_flux, closure_params));
    test_fixture.registerEvaluator<EvalType>(mms_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(mms_eval->_continuity_mms_source);
    for (int dim = 0; dim < num_space_dim; ++dim)
        test_fixture.registerTestField<EvalType>(
            mms_eval->_momentum_mms_source[dim]);
    test_fixture.registerTestField<EvalType>(mms_eval->_energy_mms_source);

    // Evaluate mms source
    test_fixture.evaluate<EvalType>();

    // Check the values
    const auto continuity_result = test_fixture.getTestFieldData<EvalType>(
        mms_eval->_continuity_mms_source);
    const auto momentum_0_result = test_fixture.getTestFieldData<EvalType>(
        mms_eval->_momentum_mms_source[0]);
    const auto momentum_1_result = test_fixture.getTestFieldData<EvalType>(
        mms_eval->_momentum_mms_source[1]);
    const auto energy_result = test_fixture.getTestFieldData<EvalType>(
        mms_eval->_energy_mms_source);

    // Check the mms source solutions
    const int num_point = continuity_result.extent(1);
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(expected_sol[0], fieldValue(continuity_result, 0, qp));
        EXPECT_DOUBLE_EQ(expected_sol[1], fieldValue(momentum_0_result, 0, qp));
        EXPECT_DOUBLE_EQ(expected_sol[2], fieldValue(momentum_1_result, 0, qp));
        EXPECT_DOUBLE_EQ(expected_sol[num_conserve - 1],
                         fieldValue(energy_result, 0, qp));
    }

    if (num_space_dim == 3)
    {
        const auto momentum_2_result = test_fixture.getTestFieldData<EvalType>(
            mms_eval->_momentum_mms_source[2]);
        for (int qp = 0; qp < num_point; ++qp)
            EXPECT_DOUBLE_EQ(expected_sol[num_conserve - 2],
                             fieldValue(momentum_2_result, 0, qp));
    }
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testMMSSourceConvective()
{
    constexpr int num_space_dim = NumSpaceDim;

    SCOPED_TRACE("Convective");
    Kokkos::Array<double, num_space_dim + 2> expected_value;
    if (num_space_dim == 2)
    {
        expected_value[0] = 0.06485567288359177;
        expected_value[1] = 0.014546900447422031;
        expected_value[2] = 0.1796124151976149;
        expected_value[3] = 0.3731078016877022;
    }
    else if (num_space_dim == 3)
    {
        expected_value[0] = -0.004621895004700585;
        expected_value[1] = -0.001970095500617289;
        expected_value[2] = -0.005199631880288158;
        expected_value[3] = -0.005749127851644679;
        expected_value[4] = -0.01193110189893135;
    }
    testEval<EvalType, num_space_dim>(false, expected_value);
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testMMSSourceConvectiveViscous()
{
    constexpr int num_space_dim = NumSpaceDim;

    SCOPED_TRACE("Convective + Viscous");
    Kokkos::Array<double, num_space_dim + 2> expected_value;
    if (num_space_dim == 2)
    {
        expected_value[0] = 0.06485567288359177;
        expected_value[1] = -0.11093398765764714;
        expected_value[2] = 0.17029666578878958;
        expected_value[3] = 0.3321719304832673;
    }
    else if (num_space_dim == 3)
    {
        expected_value[0] = -0.004621895004700585;
        expected_value[1] = -0.00509066713414899;
        expected_value[2] = 0.01507826268348906;
        expected_value[3] = -0.17494567555212637;
        expected_value[4] = 0.010477883105231796;
    }
    testEval<EvalType, num_space_dim>(true, expected_value);
}

//---------------------------------------------------------------------------//
TEST(MMSConvective2D, DISABLED_residual_test)
{
    testMMSSourceConvective<panzer::Traits::Residual, 2>();
}

//---------------------------------------------------------------------------//
TEST(MMSConvective2D, DISABLED_jacobian_test)
{
    testMMSSourceConvective<panzer::Traits::Jacobian, 2>();
}

//---------------------------------------------------------------------------//
TEST(MMSConvectiveViscous2D, DISABLED_residual_test)
{
    testMMSSourceConvectiveViscous<panzer::Traits::Residual, 2>();
}

//---------------------------------------------------------------------------//
TEST(MMSConvectiveViscous2D, DISABLED_jacobian_test)
{
    testMMSSourceConvectiveViscous<panzer::Traits::Jacobian, 2>();
}

//---------------------------------------------------------------------------//
TEST(MMSConvective3D, DISABLED_residual_test)
{
    testMMSSourceConvective<panzer::Traits::Residual, 3>();
}

//---------------------------------------------------------------------------//
TEST(MMSConvective3D, DISABLED_jacobian_test)
{
    testMMSSourceConvective<panzer::Traits::Jacobian, 3>();
}

//---------------------------------------------------------------------------//
TEST(MMSConvectiveViscous3D, DISABLED_residual_test)
{
    testMMSSourceConvectiveViscous<panzer::Traits::Residual, 3>();
}

//---------------------------------------------------------------------------//
TEST(MMSConvectiveViscous3D, DISABLED_jacobian_test)
{
    testMMSSourceConvectiveViscous<panzer::Traits::Jacobian, 3>();
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.type_name = "MethodManufacturedSolutionSource";
    if (num_space_dim == 2)
        test_fixture.eval_name = "Method of Manufactured Solution Source 2D";
    else if (num_space_dim == 3)
        test_fixture.eval_name = "Method of Manufactured Solution Source 3D";
    test_fixture.model_params.set("Density", 1.0).set("Kinematic Viscosity", 2.0);
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.template buildAndTest<
        ClosureModel::MethodManufacturedSolutionSource<EvalType,
                                                       panzer::Traits,
                                                       num_space_dim>,
        num_space_dim>();
}

TEST(MMS_Source_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(MMS_Source_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(MMS_Source_Factory3D, residual_test)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(MMS_Source_Factory3D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}
} // end namespace Test
} // end namespace VertexCFD
