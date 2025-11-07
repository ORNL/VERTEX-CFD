#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFBubbleExact.hpp"

#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval()
{
    const int integration_order = 2;
    const int basis_order = 1;
    constexpr int num_space_dim = NumSpaceDim;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    Teuchos::ParameterList closure_params;
    closure_params.set("DOF Name", "dof");
    Teuchos::Array<double> location(num_space_dim);
    location[0] = 0.47;
    location[1] = 0.92;
    if (num_space_dim > 2)
        location[2] = 0.95;

    closure_params.set("Bubble Location", location);

    const double radius = 0.55;
    closure_params.set<double>("Bubble Radius", radius);

    const auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleLSVOFBubbleExact<EvalType,
                                                         panzer::Traits,
                                                         num_space_dim>(
            ir, closure_params));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_exact_dof);

    test_fixture.evaluate<EvalType>();

    const auto fc_dof
        = test_fixture.getTestFieldData<EvalType>(eval->_exact_dof);

    const double dof_ref_2D[4] = {1.0, 1.0, 0.0, 0.0};
    const double dof_ref_3D[8] = {1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const double* dof_ref = num_space_dim == 3 ? dof_ref_3D : dof_ref_2D;

    const int num_point = ir.num_points;

    // Check on values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(dof_ref[qp], fieldValue(fc_dof, 0, qp));
    }
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFBubbleExact2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFBubbleExact2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFBubbleExact3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFBubbleExact3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
