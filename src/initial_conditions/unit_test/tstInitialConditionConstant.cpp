#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "initial_conditions/VertexCFD_InitialCondition_Constant.hpp"

#include <Teuchos_ParameterList.hpp>

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
template<class EvalType>
void testEval()
{
    const int num_space_dim = 2;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    Teuchos::ParameterList params;
    const double ic_value = 1.3;
    params.set<double>("Value", ic_value);
    params.set<std::string>("Equation Set Name", "dof");
    auto eval = Teuchos::rcp(
        new InitialCondition::Constant<EvalType, panzer::Traits>(
            params, *test_fixture.basis_ir_layout->getBasis()));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_ic);

    test_fixture.evaluate<EvalType>();

    auto ic = test_fixture.getTestFieldData<EvalType>(eval->_ic);
    const int num_point = ic.extent(1);

    // Number of degree of freedom
    const int num_dofs = 4;
    EXPECT_EQ(num_dofs, num_point);

    for (int b = 0; b < num_point; ++b)
    {
        EXPECT_EQ(ic_value, fieldValue(ic, 0, b));
    }
}

//---------------------------------------------------------------------------//
TEST(Constant, residual_test)
{
    testEval<panzer::Traits::Residual>();
}

//---------------------------------------------------------------------------//
TEST(Constant, jacobian_test)
{
    testEval<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//

} // namespace Test
} // namespace VertexCFD
