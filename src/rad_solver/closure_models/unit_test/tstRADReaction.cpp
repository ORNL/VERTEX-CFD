#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "rad_solver/closure_models/VertexCFD_Closure_RADReaction.hpp"
#include "rad_solver/species_properties/VertexCFD_ConstantSpeciesProperties.hpp"

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> species_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> species_1;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> flux;

    Dependencies(const panzer::IntegrationRule& ir, const std::string flux_name)
        : species_0("species_0", ir.dl_scalar)
        , species_1("species_1", ir.dl_scalar)
        , flux(flux_name, ir.dl_scalar)
    {
        this->addEvaluatedField(species_0);
        this->addEvaluatedField(species_1);

        this->addEvaluatedField(flux);

        this->setName(
            "RAD Reaction Term Unit Test "
            "Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData) override
    {
        species_0.deep_copy(1.25);
        species_1.deep_copy(2.5);

        flux.deep_copy(0.25);
    }
};

template<class EvalType>
void testEval(const int num_space_dim,
              const bool build_bateman,
              const bool build_transmutation,
              const std::string flux_name)
{
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    auto& ir = *test_fixture.ir;

    // Initialize dependents
    auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir, flux_name));
    test_fixture.registerEvaluator<EvalType>(deps);

    Teuchos::Array<double> species_decay(4);
    species_decay[0] = -1.0;
    species_decay[1] = 0.0;
    species_decay[2] = 1.0;
    species_decay[3] = -0.1;

    Teuchos::Array<double> mic_cross_section(4);
    mic_cross_section[0] = -2.0;
    mic_cross_section[1] = 1.0;
    mic_cross_section[2] = 3.0;
    mic_cross_section[3] = -5.1;

    Teuchos::ParameterList rad_params;
    rad_params.set("Build Reaction Bateman Source", build_bateman);
    rad_params.set("Build Reaction Transmutation Source", build_transmutation);
    rad_params.set("Number of Species", 2);
    Teuchos::ParameterList reaction_params;
    reaction_params.set("Species Decay", species_decay);
    reaction_params.set("Microscopic Cross-Section", mic_cross_section);
    const SpeciesProperties::ConstantSpeciesProperties species_prop(
        rad_params, reaction_params);

    // Initialize class object to test
    auto eval
        = Teuchos::rcp(new ClosureModel::RADReaction<EvalType, panzer::Traits>(
            ir, species_prop, flux_name));
    test_fixture.registerEvaluator<EvalType>(eval);
    for (int num = 0; num < 2; ++num)
        test_fixture.registerTestField<EvalType>(eval->_reaction_term[num]);

    test_fixture.evaluate<EvalType>();

    auto calc_reaction_term_0
        = test_fixture.getTestFieldData<EvalType>(eval->_reaction_term[0]);
    auto calc_reaction_term_1
        = test_fixture.getTestFieldData<EvalType>(eval->_reaction_term[1]);

    const int num_point = ir.num_points;

    // Expected values
    double exp_reaction_term_0;
    double exp_reaction_term_1;

    if (build_bateman && build_transmutation)
    {
        exp_reaction_term_0 = -1.25;
        exp_reaction_term_1 = -1.25;
    }
    else if (build_bateman)
    {
        exp_reaction_term_0 = -1.25;
        exp_reaction_term_1 = 1.0;
    }
    else if (build_transmutation)
    {
        exp_reaction_term_0 = 0.0;
        exp_reaction_term_1 = -2.25;
    }

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_reaction_term_0,
                         fieldValue(calc_reaction_term_0, 0, qp));
        EXPECT_DOUBLE_EQ(exp_reaction_term_1,
                         fieldValue(calc_reaction_term_1, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(RADReactionBateman2D, Residual)
{
    testEval<panzer::Traits::Residual>(2, true, false, "neutron_flux");
}

//-----------------------------------------------------------------//
TEST(RADReactionBateman2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2, true, false, "neutron_flux");
}

//-----------------------------------------------------------------//
TEST(RADReactionTransmutation2D, Residual)
{
    testEval<panzer::Traits::Residual>(2, false, true, "neutron_flux");
}

//-----------------------------------------------------------------//
TEST(RADReactionTransmutation2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2, false, true, "neutron_flux");
}

//-----------------------------------------------------------------//
TEST(RADReactionBatemanTransmutation2D, Residual)
{
    testEval<panzer::Traits::Residual>(2, true, true, "neutron_flux");
}

//-----------------------------------------------------------------//
TEST(RADReactionBatemanTransmutation2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2, true, true, "neutron_flux");
}

//-----------------------------------------------------------------//
TEST(RADReactionBatemanTransmutationFluxName2D, Residual)
{
    testEval<panzer::Traits::Residual>(2, true, true, "flux");
}

//-----------------------------------------------------------------//
TEST(RADReactionBatemanTransmutationFluxName2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2, true, true, "flux");
}

//-----------------------------------------------------------------//
TEST(RADReactionBatemanTransmutation3D, Residual)
{
    testEval<panzer::Traits::Residual>(3, true, true, "neutron_flux");
}

//-----------------------------------------------------------------//
TEST(RADReactionBatemanTransmutation3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(3, true, true, "neutron_flux");
}

} // namespace Test
} // namespace VertexCFD
