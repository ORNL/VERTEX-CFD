#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "closure_models/VertexCFD_Closure_VariableConvectiveFlux.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
template<class EvalType>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> var;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> vel_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> vel_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> vel_2;

    Dependencies(const panzer::IntegrationRule& ir)
        : var("spalart_allmaras_variable", ir.dl_scalar)
        , vel_0("velocity_0", ir.dl_scalar)
        , vel_1("velocity_1", ir.dl_scalar)
        , vel_2("velocity_2", ir.dl_scalar)

    {
        this->addEvaluatedField(var);
        this->addEvaluatedField(vel_0);
        this->addEvaluatedField(vel_1);
        this->addEvaluatedField(vel_2);

        this->setName("Variable Convective Flux Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData) override
    {
        var.deep_copy(2.0);
        vel_0.deep_copy(4.0);
        vel_1.deep_copy(-5.0);
        vel_2.deep_copy(6.0);
    }
};

template<class EvalType, int NumSpaceDim>
void testEval()
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Closure parameters
    Teuchos::ParameterList closure_params;
    closure_params.set("Field Name", "spalart_allmaras_variable");
    closure_params.set("Equation Name", "spalart_allmaras_equation");

    // Eval dependencies
    auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    auto eval = Teuchos::rcp(
        new ClosureModel::
            VariableConvectiveFlux<EvalType, panzer::Traits, num_space_dim>(
                ir, closure_params));

    // Register and evaluate
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_var_flux);
    test_fixture.evaluate<EvalType>();
    const auto fc_var
        = test_fixture.getTestFieldData<EvalType>(eval->_var_flux);

    // Expected values
    const double exp_var_flux[3] = {
        8.0,
        -10.0,
        num_space_dim == 3 ? 12.0 : std::numeric_limits<double>::quiet_NaN()};

    // Assert values
    const int num_point = ir.num_points;
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int dim = 0; dim < num_space_dim; dim++)
        {
            EXPECT_EQ(exp_var_flux[dim], fieldValue(fc_var, 0, qp, dim));
        }
    }
}

//-----------------------------------------------------------------//
TEST(VariableConvectiveFlux2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(VariableConvectiveFlux2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(VariableConvectiveFlux3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(VariableConvectiveFlux3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
