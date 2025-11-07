#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleKEpsilonDiffusivityCoefficient.hpp"

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> nu;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> nu_t;

    Dependencies(const panzer::IntegrationRule& ir)
        : nu("kinematic_viscosity", ir.dl_scalar)
        , nu_t("turbulent_eddy_viscosity", ir.dl_scalar)
    {
        this->addEvaluatedField(nu);
        this->addEvaluatedField(nu_t);
        this->setName(
            "K-Epsilon Incompressible Diffusivity Coefficient Unit "
            "Test "
            "Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData /**d**/) override
    {
        nu.deep_copy(0.25);
        nu_t.deep_copy(1.5);
    }
};

template<class EvalType>
void testEval()
{
    const int num_space_dim = 2;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Set turbulent quantities
    const auto nu_t_value = 1.5;
    const auto sigma_k = 1.00;
    const auto sigma_e = 1.30;

    // Eval dependencies
    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Fluid properties
    const auto nu = 0.25;

    // Initialize and register
    auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleKEpsilonDiffusivityCoefficient<
            EvalType,
            panzer::Traits>(ir));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_diffusivity_var_k);
    test_fixture.registerTestField<EvalType>(eval->_diffusivity_var_e);
    test_fixture.evaluate<EvalType>();

    // Evaluate test fields
    const auto fv_var_k
        = test_fixture.getTestFieldData<EvalType>(eval->_diffusivity_var_k);
    const auto fv_var_e
        = test_fixture.getTestFieldData<EvalType>(eval->_diffusivity_var_e);

    // Expected values
    const int num_point = ir.num_points;
    const double exp_diffusivity_var_k = nu + nu_t_value / sigma_k;
    const double exp_diffusivity_var_e = nu + nu_t_value / sigma_e;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_EQ(exp_diffusivity_var_k, fieldValue(fv_var_k, 0, qp));
        EXPECT_EQ(exp_diffusivity_var_e, fieldValue(fv_var_e, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleKEpsilonDiffusivityCoefficient, Residual)
{
    testEval<panzer::Traits::Residual>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleKEpsilonDiffusivityCoefficient, Jacobian)
{
    testEval<panzer::Traits::Jacobian>();
}

//-----------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD
