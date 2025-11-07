#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "boundary_conditions/VertexCFD_BoundaryState_VariableExtrapolate.hpp"

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_config.hpp>

#include <mpi.h>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
// Test data dependencies.
template<class EvalType>
struct Dependencies : public PHX::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _variable;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_variable;

    Dependencies(const panzer::IntegrationRule& ir)
        : _variable("variable", ir.dl_scalar)
        , _grad_variable("GRAD_variable", ir.dl_vector)
    {
        this->addEvaluatedField(_variable);
        this->addEvaluatedField(_grad_variable);
        this->setName("Variable Extrapolate Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData /**d**/) override
    {
        _variable.deep_copy(1.5);
        _grad_variable.deep_copy(2.0);
    }
};

//---------------------------------------------------------------------------//
template<class EvalType>
void testEval(const int num_grad_dim)
{
    // Test fixture
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_grad_dim, integration_order, basis_order);

    // Create dependencies
    const auto dep_eval
        = Teuchos::rcp(new Dependencies<EvalType>(*test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Create extrapolate evaluator.
    const std::string variable_name = "variable";
    const auto extr_eval = Teuchos::rcp(
        new BoundaryCondition::VariableExtrapolate<EvalType, panzer::Traits>(
            *test_fixture.ir, variable_name));
    test_fixture.registerEvaluator<EvalType>(extr_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(extr_eval->_boundary_variable);
    test_fixture.registerTestField<EvalType>(
        extr_eval->_boundary_grad_variable);

    // Evaluate values
    test_fixture.evaluate<EvalType>();

    // Check values
    const auto boundary_var_result = test_fixture.getTestFieldData<EvalType>(
        extr_eval->_boundary_variable);
    const auto boundary_grad_var_result
        = test_fixture.getTestFieldData<EvalType>(
            extr_eval->_boundary_grad_variable);

    const int num_point = boundary_var_result.extent(1);

    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(1.5, fieldValue(boundary_var_result, 0, qp));

        for (int d = 0; d < num_grad_dim; ++d)
        {
            EXPECT_DOUBLE_EQ(2.0,
                             fieldValue(boundary_grad_var_result, 0, qp, d));
        }
    }
}

//---------------------------------------------------------------------------//
// 2-D case
TEST(Test2DVariableExtrapolate, residual)
{
    testEval<panzer::Traits::Residual>(2);
}

TEST(Test2DVariableExtrapolate, jacobian)
{
    testEval<panzer::Traits::Jacobian>(2);
}

//---------------------------------------------------------------------------//
// 3-D case
TEST(Test3DVariableExtrapolate, residual)
{
    testEval<panzer::Traits::Residual>(3);
}

TEST(Test3DVariableExtrapolate, jacobian)
{
    testEval<panzer::Traits::Jacobian>(3);
}

//---------------------------------------------------------------------------//
} // end namespace Test
} // end namespace VertexCFD
