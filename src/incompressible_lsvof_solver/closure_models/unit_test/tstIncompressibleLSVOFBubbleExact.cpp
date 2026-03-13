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
// LSVOF model types
enum class LSVOFModel
{
    VOF,
    CLS
};

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval(const LSVOFModel lsvof_model)
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

    const std::string lsvof_model_name
        = (lsvof_model == LSVOFModel::VOF) ? "VOF" : "CLS";
    closure_params.set("LSVOF Model Type", lsvof_model_name);

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

    const double vof_ref_2D[4] = {1.0, 1.0, 0.0, 0.0};
    const double vof_ref_3D[8] = {1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const double* vof_ref = num_space_dim == 3 ? vof_ref_3D : vof_ref_2D;

    const double cls_ref_2D[4] = {0.20532615172787966,
                                  0.25989821522207923,
                                  -0.22702914218322479,
                                  -0.20440922028473418};
    const double cls_ref_3D[8] = {0.16944018357068447,
                                  0.21805910506637377,
                                  -0.24359939516105977,
                                  -0.2214654780666614,
                                  -0.2651325144731851,
                                  -0.24359939516105977,
                                  -0.5221078501114815,
                                  -0.5058286916537575};
    const double* cls_ref = num_space_dim == 3 ? cls_ref_3D : cls_ref_2D;

    const double* dof_ref = lsvof_model == LSVOFModel::VOF ? vof_ref : cls_ref;

    const int num_point = ir.num_points;

    // Check on values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_NEAR(dof_ref[qp], fieldValue(fc_dof, 0, qp), 1.0e-15);
    }
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFBubbleExact2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(LSVOFModel::VOF);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFBubbleExact2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(LSVOFModel::VOF);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFBubbleExact3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(LSVOFModel::VOF);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFBubbleExact3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(LSVOFModel::VOF);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSBubbleExact2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(LSVOFModel::CLS);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSBubbleExact2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(LSVOFModel::CLS);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSBubbleExact3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(LSVOFModel::CLS);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSBubbleExact3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(LSVOFModel::CLS);
}

} // namespace Test
} // namespace VertexCFD
