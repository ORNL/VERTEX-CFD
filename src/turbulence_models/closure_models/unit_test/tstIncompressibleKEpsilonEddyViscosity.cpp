#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleKEpsilonEddyViscosity.hpp"

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> turb_kinetic_energy;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> turb_dissipation_rate;

    Dependencies(const panzer::IntegrationRule& ir)
        : turb_kinetic_energy("turb_kinetic_energy", ir.dl_scalar)
        , turb_dissipation_rate("turb_dissipation_rate", ir.dl_scalar)
    {
        this->addEvaluatedField(turb_kinetic_energy);
        this->addEvaluatedField(turb_dissipation_rate);
        this->setName(
            "K-Epsilon Incompressible Eddy Viscosity Unit "
            "Test "
            "Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData /**d**/) override
    {
        turb_kinetic_energy.deep_copy(2.5);
        turb_dissipation_rate.deep_copy(1.25);
    }
};

template<class EvalType>
void testEval()
{
    using std::pow;
    const int num_space_dim = 2;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Set turbulent quantities
    const double C_nu = 0.09;
    const double turb_kinetic_energy_value = 2.5;
    const double turb_dissipation_rate_value = 1.25;

    // Eval dependencies
    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize and register
    auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleKEpsilonEddyViscosity<EvalType,
                                                              panzer::Traits>(
            ir));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_nu_t);
    test_fixture.evaluate<EvalType>();

    // Evaluate test fields
    const auto fv_nu_t = test_fixture.getTestFieldData<EvalType>(eval->_nu_t);

    // Expected values
    const int num_point = ir.num_points;
    const double exp_diffusivity_var = C_nu
                                       * pow(turb_kinetic_energy_value, 2.0)
                                       / turb_dissipation_rate_value;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_EQ(exp_diffusivity_var, fieldValue(fv_nu_t, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleKEpsilonEddyViscosity, residual_test)
{
    testEval<panzer::Traits::Residual>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleKEpsilonEddyViscosity, jacobian_test)
{
    testEval<panzer::Traits::Jacobian>();
}
//-----------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD
