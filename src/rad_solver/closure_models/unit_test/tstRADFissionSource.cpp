#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "rad_solver/closure_models/VertexCFD_Closure_RADFissionSource.hpp"

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> flux;

    Dependencies(const panzer::IntegrationRule& ir, const std::string flux_name)
        : flux(flux_name, ir.dl_scalar)
    {
        this->addEvaluatedField(flux);

        this->setName("RAD Fission Source Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData) override
    {
        flux.deep_copy(0.25);
    }
};

template<class EvalType>
void testEval(const int num_species, const std::string flux_name)
{
    constexpr int num_space_dim = 2;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);
    const auto& ir = *test_fixture.ir;

    const double nan_val = std::numeric_limits<double>::quiet_NaN();

    // Initialize dependents
    auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir, flux_name));
    test_fixture.registerEvaluator<EvalType>(deps);

    Teuchos::Array<double> num_atoms(num_species);

    num_atoms[0] = 6.02214076e23;
    num_atoms[1] = 6.02214076e23 * 2.0;
    if (num_species == 3)
        num_atoms[2] = 6.02214076e23 * 4.0;

    Teuchos::ParameterList rad_params;
    rad_params.set("Build Fission Source", true);
    rad_params.set("Number of Species", num_species);
    Teuchos::ParameterList reaction_params;
    reaction_params.set("Fission Cross-Section", 0.8);
    reaction_params.set("Number of atoms per species", num_atoms);
    const SpeciesProperties::ConstantSpeciesProperties species_prop(
        rad_params, reaction_params);

    // Initialize class object to test
    auto eval = Teuchos::rcp(
        new ClosureModel::RADFissionSource<EvalType, panzer::Traits>(
            ir, species_prop, flux_name));

    test_fixture.registerEvaluator<EvalType>(eval);
    for (int num = 0; num < num_species; ++num)
        test_fixture.registerTestField<EvalType>(eval->_fission_source[num]);

    test_fixture.evaluate<EvalType>();

    const int num_point = ir.num_points;

    // Expected values
    const double exp_species[3] = {0.2, 0.4, num_species == 3 ? 0.8 : nan_val};

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int num = 0; num < num_species; num++)
        {
            const auto calc_species = test_fixture.getTestFieldData<EvalType>(
                eval->_fission_source[num]);
            EXPECT_EQ(exp_species[num], fieldValue(calc_species, 0, qp));
        }
    }
}

//-----------------------------------------------------------------//
TEST(RADFissionSource2Species, Residual)
{
    testEval<panzer::Traits::Residual>(2, "neutron_flux");
}

//-----------------------------------------------------------------//
TEST(RADFissionSource2Species, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2, "neutron_flux");
}

//-----------------------------------------------------------------//
TEST(RADFissionSource3Species, Residual)
{
    testEval<panzer::Traits::Residual>(3, "neutron_flux");
}

//-----------------------------------------------------------------//
TEST(RADFissionSource3Species, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(3, "neutron_flux");
}

//-----------------------------------------------------------------//
TEST(RADFissionSource2SpeciesFluxName, Residual)
{
    testEval<panzer::Traits::Residual>(2, "flux");
}

//-----------------------------------------------------------------//
TEST(RADFissionSource2SpeciesFluxName, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2, "flux");
}

//-----------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD
