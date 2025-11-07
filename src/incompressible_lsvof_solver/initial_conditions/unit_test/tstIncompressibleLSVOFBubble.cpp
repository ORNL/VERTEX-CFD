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
template<class EvalType, int NumSpaceDim>
void testEval(const PhaseType phase_type)
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

    auto eval = Teuchos::rcp(
        new InitialCondition::
            IncompressibleLSVOFBubble<EvalType, panzer::Traits, num_space_dim>(
                params, *test_fixture.basis_ir_layout->getBasis()));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_alpha);

    test_fixture.evaluate<EvalType>();

    auto ic = test_fixture.getTestFieldData<EvalType>(eval->_alpha);

    double inside_value = 0.0;
    double outside_value = 1.0;

    if (phase_type == PhaseType::Dispersed)
    {
        inside_value = 1.0;
        outside_value = 0.0;
    }

    // Number of degree of freedom based on `num_space_dim` value
    int num_dofs = 4;
    if (num_space_dim == 3)
        num_dofs = 8;
    EXPECT_EQ(num_dofs, ic.extent(1));

    // Check on values
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

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFBubbleDispersed2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(PhaseType::Dispersed);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFBubbleDispersed2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(PhaseType::Dispersed);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFBubbleDispersed3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(PhaseType::Dispersed);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFBubbleDispersed3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(PhaseType::Dispersed);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFBubbleContinuous2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(PhaseType::Continuous);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFBubbleContinuous2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(PhaseType::Continuous);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFBubbleContinuous3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(PhaseType::Continuous);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFBubbleContinuous3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(PhaseType::Continuous);
}

//---------------------------------------------------------------------------//

} // namespace Test
} // namespace VertexCFD
