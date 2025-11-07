#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "initial_conditions/VertexCFD_InitialCondition_Sinusoidal.hpp"

#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testSinusoidal()
{
    const int integration_order = 1;
    const int basis_order = 1;
    constexpr int num_space_dim = NumSpaceDim;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    Teuchos::ParameterList params;

    const double amplitude = 2;
    const double phase = 2.5;

    params.set<double>("Amplitude", amplitude);
    params.set<double>("Phase Angle", phase);
    params.set<std::string>("Equation Set Name", "dof");

    auto eval = Teuchos::rcp(
        new InitialCondition::Sinusoidal<EvalType, panzer::Traits>(
            params, *test_fixture.basis_ir_layout->getBasis()));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_ic);

    test_fixture.evaluate<EvalType>();

    auto ic = test_fixture.getTestFieldData<EvalType>(eval->_ic);

    // Number of degree of freedom based on `num_space_dim` value
    const int num_dofs = num_space_dim == 3 ? 8 : 4;

    EXPECT_EQ(num_dofs, ic.extent(1));

    if (num_space_dim == 2)
    {
        EXPECT_DOUBLE_EQ(1.1969442882079131, fieldValue(ic, 0, 0));
        EXPECT_DOUBLE_EQ(-0.7015664553792397, fieldValue(ic, 0, 1));
        EXPECT_DOUBLE_EQ(-1.9550602353301940, fieldValue(ic, 0, 2));
        EXPECT_DOUBLE_EQ(-0.7015664553792397, fieldValue(ic, 0, 3));
    }
    else if (num_space_dim == 3)
    {
        EXPECT_DOUBLE_EQ(1.1969442882079129, fieldValue(ic, 0, 0));
        EXPECT_DOUBLE_EQ(-0.7015664553792397, fieldValue(ic, 0, 1));
        EXPECT_DOUBLE_EQ(-1.9550602353301940, fieldValue(ic, 0, 2));
        EXPECT_DOUBLE_EQ(-0.7015664553792397, fieldValue(ic, 0, 3));
        EXPECT_DOUBLE_EQ(-0.7015664553792397, fieldValue(ic, 0, 4));
        EXPECT_DOUBLE_EQ(-1.9550602353301940, fieldValue(ic, 0, 5));
        EXPECT_DOUBLE_EQ(-1.4110806511407838, fieldValue(ic, 0, 6));
        EXPECT_DOUBLE_EQ(-1.9550602353301940, fieldValue(ic, 0, 7));
    }
}

//---------------------------------------------------------------------------//
TEST(Sinusoidal2D, Residual)
{
    testSinusoidal<panzer::Traits::Residual, 2>();
}

//---------------------------------------------------------------------------//
TEST(Sinusoidal2D, Jacobian)
{
    testSinusoidal<panzer::Traits::Jacobian, 2>();
}

//---------------------------------------------------------------------------//
TEST(Sinusoidal3D, Residual)
{
    testSinusoidal<panzer::Traits::Residual, 3>();
}

//---------------------------------------------------------------------------//
TEST(Sinusoidal3D, Jacobian)
{
    testSinusoidal<panzer::Traits::Jacobian, 3>();
}

//---------------------------------------------------------------------------//

} // namespace Test
} // namespace VertexCFD
