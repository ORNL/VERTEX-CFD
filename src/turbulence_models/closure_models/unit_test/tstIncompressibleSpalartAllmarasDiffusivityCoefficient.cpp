#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleSpalartAllmarasDiffusivityCoefficient.hpp"

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> sa_var;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> nu;
    const double _sa_var_value;

    Dependencies(const panzer::IntegrationRule& ir, const double sa_var_value)
        : sa_var("spalart_allmaras_variable", ir.dl_scalar)
        , nu("kinematic_viscosity", ir.dl_scalar)
        , _sa_var_value(sa_var_value)
    {
        this->addEvaluatedField(sa_var);
        this->addEvaluatedField(nu);
        this->setName(
            "Spalart-Allmaras Incompressible Diffusivity Coefficient Unit "
            "Test "
            "Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData /**d**/) override
    {
        sa_var.deep_copy(_sa_var_value);
        nu.deep_copy(0.25);
    }
};

template<class EvalType>
void testEval(const double sa_var_value)
{
    const int num_space_dim = 2;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Eval dependencies
    const auto deps
        = Teuchos::rcp(new Dependencies<EvalType>(ir, sa_var_value));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize and register
    auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleSpalartAllmarasDiffusivityCoefficient<
            EvalType,
            panzer::Traits>(ir));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_diffusivity_var);
    test_fixture.evaluate<EvalType>();

    // Evaluate test fields
    const auto fv_var
        = test_fixture.getTestFieldData<EvalType>(eval->_diffusivity_var);

    // Expected values
    const int num_point = ir.num_points;
    const double exp_diffusivity_var = sa_var_value < 0 ? 4.79243119266055
                                                        : 4.875;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_EQ(exp_diffusivity_var, fieldValue(fv_var, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleSADiffusivityCoefficientPositive, residual_test)
{
    testEval<panzer::Traits::Residual>(3.0);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSADiffusivityCoefficientPositive, jacobian_test)
{
    testEval<panzer::Traits::Jacobian>(3.0);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSADiffusivityCoefficientNegative, residual_test)
{
    testEval<panzer::Traits::Residual>(-3.0);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSADiffusivityCoefficientNegative, jacobian_test)
{
    testEval<panzer::Traits::Jacobian>(-3.0);
}

//-----------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD
