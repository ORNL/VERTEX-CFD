#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleSpalartAllmarasEddyViscosity.hpp"

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
            "Spalart-Allmaras Incompressible Eddy Viscosity Unit "
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
        new ClosureModel::IncompressibleSpalartAllmarasEddyViscosity<
            EvalType,
            panzer::Traits>(ir));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_nu_t);
    test_fixture.evaluate<EvalType>();

    // Evaluate test fields
    const auto fv_nu_t = test_fixture.getTestFieldData<EvalType>(eval->_nu_t);

    // Expected values
    const int num_point = ir.num_points;
    const double exp_diffusivity_var = sa_var_value < 0 ? 0.0
                                                        : 2.485245055997116;
    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_diffusivity_var, fieldValue(fv_nu_t, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleSAEddyViscosityPositive, residual_test)
{
    testEval<panzer::Traits::Residual>(3.0);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSAEddyViscosityPositive, jacobian_test)
{
    testEval<panzer::Traits::Jacobian>(3.0);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSAEddyViscosityNegative, residual_test)
{
    testEval<panzer::Traits::Residual>(-3.0);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSAEddyViscosityNegative, jacobian_test)
{
    testEval<panzer::Traits::Jacobian>(-3.0);
}

//-----------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD
