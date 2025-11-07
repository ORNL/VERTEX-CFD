#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_InductionConstantSource.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{

template<class EvalType, int NumSpaceDim>
void testEval()
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Initialize class object to test
    Teuchos::Array<double> ind_input_source(num_space_dim);
    ind_input_source[0] = 0.1;
    ind_input_source[1] = 0.2;
    if (num_space_dim == 3)
        ind_input_source[2] = 0.3;
    Teuchos::ParameterList closure_params;
    closure_params.set("Induction Source", ind_input_source);
    auto eval = Teuchos::rcp(
        new ClosureModel::
            InductionConstantSource<EvalType, panzer::Traits, num_space_dim>(
                ir, closure_params));
    test_fixture.registerEvaluator<EvalType>(eval);
    for (int dim = 0; dim < num_space_dim; ++dim)
        test_fixture.registerTestField<EvalType>(eval->_induction_source[dim]);

    test_fixture.evaluate<EvalType>();

    const auto fc_ind_0
        = test_fixture.getTestFieldData<EvalType>(eval->_induction_source[0]);
    const auto fc_ind_1
        = test_fixture.getTestFieldData<EvalType>(eval->_induction_source[1]);

    const int num_point = ir.num_points;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_EQ(ind_input_source[0], fieldValue(fc_ind_0, 0, qp));
        EXPECT_EQ(ind_input_source[1], fieldValue(fc_ind_1, 0, qp));
        if (num_space_dim > 2) // 3D mesh
        {
            const auto fc_ind_2 = test_fixture.getTestFieldData<EvalType>(
                eval->_induction_source[2]);
            EXPECT_EQ(ind_input_source[2], fieldValue(fc_ind_2, 0, qp));
        }
    }
}

//-----------------------------------------------------------------//
TEST(InductionConstantSource2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(InductionConstantSource2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(InductionConstantSource3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(InductionConstantSource3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    const Teuchos::Array<double> ind_input_source(num_space_dim);
    test_fixture.model_params.set("Induction Source", ind_input_source);
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Full Induction MHD Properties")
        .set("Vacuum Magnetic Permeability", 0.1)
        .set("Build Magnetic Correction Potential Equation", false);
    test_fixture.factory_type = "Full Induction MHD";
    test_fixture.type_name = "InductionConstantSource";
    test_fixture.eval_name = "Induction Constant Source "
                             + std::to_string(num_space_dim) + "D";
    test_fixture.template buildAndTest<
        ClosureModel::InductionConstantSource<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(InductionConstantSourceFactory2D, Residual)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(InductionConstantSourceFactory2D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(InductionConstantSourceFactory3D, Residual)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(InductionConstantSourceFactory3D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
