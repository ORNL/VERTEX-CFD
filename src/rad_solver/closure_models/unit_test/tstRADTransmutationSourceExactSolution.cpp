#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "rad_solver/closure_models/VertexCFD_Closure_RADTransmutationSourceExactSolution.hpp"
#include "rad_solver/species_properties/VertexCFD_ConstantSpeciesProperties.hpp"

#include <gtest/gtest.h>

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

    Dependencies()
    {
        this->setName(
            "RAD Equation Transmutation Source Exact Solution Unit Test "
            "Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData) override {}
};

template<class EvalType>
void testEval(const bool calc_flux)
{
    // Set up test fixture
    const int num_space_dim = 2;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Get test fixture integrator rule
    const auto& ir = *test_fixture.ir;

    const auto deps = Teuchos::rcp(new Dependencies<EvalType>());
    test_fixture.registerEvaluator<EvalType>(deps);

    // Set list of parameters to pass to the test evaluator
    Teuchos::Array<double> species_amp_vct(2);
    species_amp_vct[0] = 1.0;
    species_amp_vct[1] = 0.0;

    Teuchos::Array<double> mic_cross_section(4);
    mic_cross_section[0] = -2.0;
    mic_cross_section[1] = 1.0;
    mic_cross_section[2] = 3.0;
    mic_cross_section[3] = -5.1;

    Teuchos::ParameterList closure_params;
    closure_params.set("Initial Species Concentration", species_amp_vct);
    closure_params.set("Time Shape Function Ramp Scale", 0.5);
    closure_params.set("Time Shape Function Ramp Offset", 1.0);
    closure_params.set("Time Shape Function Decline Scale", 0.3);
    closure_params.set("Time Shape Function Decline Offset", 0.7);
    closure_params.set("Neutron Flux Amplitude", 1.2);
    closure_params.set("Calculate Analytical Neutron Flux", calc_flux);
    closure_params.set("Taylor Series Expansion Order", 4);

    Teuchos::ParameterList rad_params;
    rad_params.set("Build Reaction Transmutation Source", true);
    rad_params.set("Number of Species", 2);
    Teuchos::ParameterList reaction_params;
    reaction_params.set("Microscopic Cross-Section", mic_cross_section);
    const SpeciesProperties::ConstantSpeciesProperties species_prop(
        rad_params, reaction_params);

    // Create test evaluator
    const auto eval = Teuchos::rcp(
        new ClosureModel::RADTransmutationSourceExactSolution<EvalType,
                                                              panzer::Traits>(
            ir, species_prop, closure_params));

    test_fixture.registerEvaluator<EvalType>(eval);
    if (calc_flux)
        test_fixture.registerTestField<EvalType>(eval->_flux);

    for (int num = 0; num < 2; ++num)
        test_fixture.registerTestField<EvalType>(eval->_exact_species[num]);

    // Set time
    test_fixture.setTime(1.0);

    test_fixture.evaluate<EvalType>();
    const double exp_neutron_flux = {-0.049300917002871};
    const double exp_species[2] = {1.1078726169114728, -0.17658287026643338};

    const int num_point = ir.num_points;
    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        if (calc_flux)
        {
            const auto calc_neutron_flux
                = test_fixture.getTestFieldData<EvalType>(eval->_flux);
            EXPECT_NEAR(
                exp_neutron_flux, fieldValue(calc_neutron_flux, 0, qp), 1e-10);
        }

        for (int num = 0; num < 2; ++num)
        {
            const auto calc_species = test_fixture.getTestFieldData<EvalType>(
                eval->_exact_species[num]);
            EXPECT_NEAR(
                exp_species[num], fieldValue(calc_species, 0, qp), 1e-10);
        }
    }
}

//-----------------------------------------------------------------//
TEST(RADTransmutationSourceExactSolution, Residual)
{
    testEval<panzer::Traits::Residual>(true);
}

//-----------------------------------------------------------------//
TEST(RADTransmutationSourceExactSolution, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(true);
}

//-----------------------------------------------------------------//
TEST(RADTransmutationSourceExactSolutionArborX, Residual)
{
    testEval<panzer::Traits::Residual>(false);
}

//-----------------------------------------------------------------//
TEST(RADTransmutationSourceExactSolutionArborX, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(false);
}

} // namespace Test
} // namespace VertexCFD
