#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "rad_solver/closure_models/VertexCFD_Closure_RADVaporRemovalSource.hpp"

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> source;

    Dependencies(const panzer::IntegrationRule& ir,
                 const std::string source_name)
        : species_0("species_0", ir.dl_scalar)
        , species_1("species_1", ir.dl_scalar)
        , source(source_name, ir.dl_scalar)
    {
        this->addEvaluatedField(species_0);
        this->addEvaluatedField(species_1);

        this->addEvaluatedField(source);

        this->setName("RAD Vapor Removal Source Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData) override
    {
        species_0.deep_copy(1.25);
        species_1.deep_copy(2.5);

        source.deep_copy(0.25);
    }
};

template<class EvalType>
void testEval(const std::string source_name)
{
    constexpr int num_space_dim = 2;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);
    const auto& ir = *test_fixture.ir;

    // Initialize dependents
    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir, source_name));
    test_fixture.registerEvaluator<EvalType>(deps);

    Teuchos::Array<double> multiplier(2);

    multiplier[0] = 2.5;
    multiplier[1] = 4.0;

    Teuchos::ParameterList rad_params;
    rad_params.set("Build Vapor Removal Source", true);
    rad_params.set("Number of Species", 2);
    Teuchos::ParameterList reaction_params;
    reaction_params.set("Vapor Removal Ratio Multiplier", multiplier);
    const SpeciesProperties::ConstantSpeciesProperties species_prop(
        rad_params, reaction_params);

    // Initialize class object to test
    auto eval = Teuchos::rcp(
        new ClosureModel::RADVaporRemovalSource<EvalType, panzer::Traits>(
            ir, species_prop, source_name));

    test_fixture.registerEvaluator<EvalType>(eval);
    for (int num = 0; num < 2; ++num)
        test_fixture.registerTestField<EvalType>(eval->_removal_source[num]);

    test_fixture.evaluate<EvalType>();

    const int num_point = ir.num_points;

    // Expected values
    const double exp_species[2] = {0.78125, 2.5};

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int num = 0; num < 2; num++)
        {
            const auto calc_species = test_fixture.getTestFieldData<EvalType>(
                eval->_removal_source[num]);
            EXPECT_EQ(exp_species[num], fieldValue(calc_species, 0, qp));
        }
    }
}

//-----------------------------------------------------------------//
TEST(RADVaporRemovalSource, Residual)
{
    testEval<panzer::Traits::Residual>("source");
}

//-----------------------------------------------------------------//
TEST(RADVaporRemovalSource, Jacobian)
{
    testEval<panzer::Traits::Jacobian>("source");
}

//-----------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD
