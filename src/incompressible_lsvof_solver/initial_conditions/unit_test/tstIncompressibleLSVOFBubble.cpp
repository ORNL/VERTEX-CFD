#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "incompressible_lsvof_solver/initial_conditions/VertexCFD_InitialCondition_IncompressibleLSVOFBubble.hpp"

#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
enum class PhaseType
{
    Dispersed,
    Continuous
};

//---------------------------------------------------------------------------//
// LSVOF model types
enum class LSVOFModel
{
    VOF,
    CLS
};

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval(const PhaseType phase_type, const LSVOFModel lsvof_model)
{
    const int integration_order = 1;
    const int basis_order = 1;
    constexpr int num_space_dim = NumSpaceDim;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    Teuchos::ParameterList params;
    Teuchos::Array<double> location(num_space_dim);
    location[0] = 0.47;
    location[1] = 0.92;
    if (num_space_dim > 2)
        location[2] = 0.95;

    if (phase_type == PhaseType::Dispersed)
        params.set<std::string>("Phase Type", "Dispersed");
    else
        params.set<std::string>("Phase Type", "Continuous");

    const double radius = 0.55;
    params.set<Teuchos::Array<double>>("Bubble Location", location);
    params.set<double>("Bubble Radius", radius);
    params.set<std::string>("Phase Name", "alpha_victoria");

    const std::string lsvof_model_name
        = (lsvof_model == LSVOFModel::VOF) ? "VOF" : "CLS";
    params.set("LSVOF Model Type", lsvof_model_name);

    auto eval = Teuchos::rcp(
        new InitialCondition::
            IncompressibleLSVOFBubble<EvalType, panzer::Traits, num_space_dim>(
                params, *test_fixture.basis_ir_layout->getBasis()));
    test_fixture.registerEvaluator<EvalType>(eval);

    if (lsvof_model_name == "VOF")
    {
        test_fixture.registerTestField<EvalType>(eval->_alpha);
    }
    else if (lsvof_model_name == "CLS")
    {
        test_fixture.registerTestField<EvalType>(eval->_phi);
    }

    test_fixture.evaluate<EvalType>();

    double inside_value = 0.0;
    double outside_value = 1.0;

    if (phase_type == PhaseType::Dispersed)
    {
        inside_value = 1.0;
        outside_value = 0.0;
    }

    double exp_phi_2D[4] = {-0.48310212467112845,
                            -0.5117438485811914,
                            0.01399626866970416,
                            0.07324010235759137};
    double exp_phi_3D[8] = {-0.8534956359034394,
                            -0.8747104969080559,
                            -0.5407795377618705,
                            -0.5129205050237764,
                            -0.48431136511207296,
                            -0.5129205050237764,
                            0.011669246652209142,
                            0.07062540743172468};

    if (phase_type == PhaseType::Continuous)
    {
        for (int i = 0; i < 4; ++i)
        {
            exp_phi_2D[i] *= -1.0;
        }
        for (int i = 0; i < 8; ++i)
        {
            exp_phi_3D[i] *= -1.0;
        }
    }

    double* exp_phi = num_space_dim == 3 ? exp_phi_3D : exp_phi_2D;

    // Check on values
    if (lsvof_model == LSVOFModel::VOF)
    {
        auto ic = test_fixture.getTestFieldData<EvalType>(eval->_alpha);

        // Number of degree of freedom based on `num_space_dim` value
        int num_dofs = 4;
        if (num_space_dim == 3)
            num_dofs = 8;
        EXPECT_EQ(num_dofs, ic.extent(1));

        if (num_space_dim == 2)
        {
            EXPECT_EQ(outside_value, fieldValue(ic, 0, 0));
            EXPECT_EQ(outside_value, fieldValue(ic, 0, 1));
            EXPECT_EQ(inside_value, fieldValue(ic, 0, 2));
            EXPECT_EQ(inside_value, fieldValue(ic, 0, 3));
        }
        else if (num_space_dim == 3)
        {
            EXPECT_EQ(outside_value, fieldValue(ic, 0, 0));
            EXPECT_EQ(outside_value, fieldValue(ic, 0, 1));
            EXPECT_EQ(outside_value, fieldValue(ic, 0, 2));
            EXPECT_EQ(outside_value, fieldValue(ic, 0, 3));
            EXPECT_EQ(outside_value, fieldValue(ic, 0, 4));
            EXPECT_EQ(outside_value, fieldValue(ic, 0, 5));
            EXPECT_EQ(inside_value, fieldValue(ic, 0, 6));
            EXPECT_EQ(inside_value, fieldValue(ic, 0, 7));
        }
    }
    else if (lsvof_model == LSVOFModel::CLS)
    {
        auto ic = test_fixture.getTestFieldData<EvalType>(eval->_phi);

        // Number of degree of freedom based on `num_space_dim` value
        int num_dofs = 4;
        if (num_space_dim == 3)
            num_dofs = 8;
        EXPECT_EQ(num_dofs, ic.extent(1));

        for (int i = 0; i < num_dofs; ++i)
        {
            EXPECT_NEAR(exp_phi[i], fieldValue(ic, 0, i), 1.0e-15);
        }
    }
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFBubbleDispersed2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(PhaseType::Dispersed,
                                          LSVOFModel::VOF);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFBubbleDispersed2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(PhaseType::Dispersed,
                                          LSVOFModel::VOF);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFBubbleDispersed3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(PhaseType::Dispersed,
                                          LSVOFModel::VOF);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFBubbleDispersed3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(PhaseType::Dispersed,
                                          LSVOFModel::VOF);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFBubbleContinuous2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(PhaseType::Continuous,
                                          LSVOFModel::VOF);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFBubbleContinuous2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(PhaseType::Continuous,
                                          LSVOFModel::VOF);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFBubbleContinuous3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(PhaseType::Continuous,
                                          LSVOFModel::VOF);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFBubbleContinuous3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(PhaseType::Continuous,
                                          LSVOFModel::VOF);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSBubbleDispersed2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(PhaseType::Dispersed,
                                          LSVOFModel::CLS);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSBubbleDispersed2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(PhaseType::Dispersed,
                                          LSVOFModel::CLS);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSBubbleDispersed3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(PhaseType::Dispersed,
                                          LSVOFModel::CLS);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSBubbleDispersed3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(PhaseType::Dispersed,
                                          LSVOFModel::CLS);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSBubbleContinuous2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(PhaseType::Continuous,
                                          LSVOFModel::CLS);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSBubbleContinuous2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(PhaseType::Continuous,
                                          LSVOFModel::CLS);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSBubbleContinuous3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(PhaseType::Continuous,
                                          LSVOFModel::CLS);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSBubbleContinuous3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(PhaseType::Continuous,
                                          LSVOFModel::CLS);
}

//---------------------------------------------------------------------------//

} // namespace Test
} // namespace VertexCFD
