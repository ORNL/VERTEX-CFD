#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "boundary_conditions/VertexCFD_BoundaryState_VariableFixed.hpp"

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_config.hpp>

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
// Test data dependencies.
template<class EvalType>
struct Dependencies : public PHX::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_variable;

    Dependencies(const panzer::IntegrationRule& ir)
        : _grad_variable("GRAD_variable", ir.dl_vector)
    {
        this->addEvaluatedField(_grad_variable);
        this->setName("Variable Fixed Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData /**d**/) override
    {
        _grad_variable.deep_copy(3.0);
    }
};

//---------------------------------------------------------------------------//
template<class EvalType>
void testEval(const int num_grad_dim,
              const Kokkos::Array<double, 3> time_values,
              const Kokkos::Array<double, 1> init_values,
              const Kokkos::Array<double, 1> exp_fields)
{
    // Test fixture
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_grad_dim, integration_order, basis_order);

    // Create dependencies
    const double value = 2.0;
    const double value_init = init_values[0];
    const double time_init = time_values[0];
    const double time_final = time_values[1];
    const double time = time_values[2];

    // Initialize dependecy evaluator
    const auto dep_eval
        = Teuchos::rcp(new Dependencies<EvalType>(*test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Create dirichlet evaluator.
    const std::string variable_name = "variable";
    Teuchos::ParameterList bc_params;
    bc_params.set("variable Value", value);
    if (init_values[0] > 0.0)
        bc_params.set("variable Value Initial", value_init);
    if (time_init > 0.0)
        bc_params.set("Time Initial", time_init);
    if (time_final > 0.0)
        bc_params.set("Time Final", time_final);

    const auto fixed_eval = Teuchos::rcp(
        new BoundaryCondition::VariableFixed<EvalType, panzer::Traits>(
            *test_fixture.ir, bc_params, variable_name));
    test_fixture.registerEvaluator<EvalType>(fixed_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(fixed_eval->_boundary_variable);
    test_fixture.registerTestField<EvalType>(
        fixed_eval->_boundary_grad_variable);

    // Set time
    test_fixture.setTime(time);

    // Evaluate values
    test_fixture.evaluate<EvalType>();

    // Check values
    const auto boundary_var_result = test_fixture.getTestFieldData<EvalType>(
        fixed_eval->_boundary_variable);
    const auto boundary_grad_var_result
        = test_fixture.getTestFieldData<EvalType>(
            fixed_eval->_boundary_grad_variable);

    const int num_point = boundary_var_result.extent(1);
    const double exp_value = exp_fields[0];
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_value, fieldValue(boundary_var_result, 0, qp));

        for (int d = 0; d < num_grad_dim; ++d)
        {
            EXPECT_DOUBLE_EQ(3.0,
                             fieldValue(boundary_grad_var_result, 0, qp, d));
        }
    }
}

//---------------------------------------------------------------------------//
// 2-D case: time transient variable fixed - steady
TEST(SteadyTest2D, Residual)
{
    testEval<panzer::Traits::Residual>(2, {-0.5, -3.0, 3.0}, {-1.0}, {2.0});
}

TEST(SteadyTest2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2, {-0.5, -3.0, 3.0}, {-1.0}, {2.0});
}

//---------------------------------------------------------------------------//
// 2-D case: time transient variable fixed - time > time_final
TEST(TimeFinalTest2D, Residual)
{
    testEval<panzer::Traits::Residual>(2, {0.5, 3.0, 3.5}, {1.0}, {2.0});
}

TEST(TimeFinalTest2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2, {-0.5, -3.0, 3.0}, {1.0}, {2.0});
}

//---------------------------------------------------------------------------//
// 2-D case: time transient variable fixed - time < time_init
TEST(TimeInitTest2D, Residual)
{
    testEval<panzer::Traits::Residual>(2, {0.5, 3.0, 0.2}, {1.0}, {1.0});
}

TEST(TimeInitTest2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2, {0.5, 3.0, 0.2}, {1.0}, {1.0});
}

//---------------------------------------------------------------------------//
// 2-D case: time transient variable fixed - time_init < time < time_final
TEST(TimeIntermediateTest2D, Residual)
{
    testEval<panzer::Traits::Residual>(2, {0.5, 3.0, 1.5}, {1.0}, {1.4});
}

TEST(TimeIntermediateTest2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2, {0.5, 3.0, 1.5}, {1.0}, {1.4});
}

//---------------------------------------------------------------------------//
// 3-D case: time transient variable fixed - time_init < time < time_final
TEST(TimeIntermediateTest3D, Residual)
{
    testEval<panzer::Traits::Residual>(3, {0.5, 3.0, 1.5}, {1.0}, {1.4});
}

TEST(TimeIntermediateTest3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(3, {0.5, 3.0, 1.5}, {1.0}, {1.4});
}

//---------------------------------------------------------------------------//
} // end namespace Test
} // end namespace VertexCFD
