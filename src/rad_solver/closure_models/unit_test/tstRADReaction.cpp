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

    double _s0;
    double _s1;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> species_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> species_1;

    Dependencies(const panzer::IntegrationRule& ir,
                 const double s0,
                 const double s1)
        : _s0(s0)
        , _s1(s1)
        , species_0("species_0", ir.dl_scalar)
        , species_1("species_1", ir.dl_scalar)
    {
        this->addEvaluatedField(species_0);
        this->addEvaluatedField(species_1);

        this->setName(
            "RAD Reaction Term Unit Test "
            "Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData) override
    {
        species_0.deep_copy(_s0);
        species_1.deep_copy(_s1);
    }
};

template<class EvalType>
void testEval(const int num_space_dim)
{
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    auto& ir = *test_fixture.ir;

    // Initialize dependents
    const double s0 = 1.25;
    const double s1 = 2.5;

    auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir, s0, s1));
    test_fixture.registerEvaluator<EvalType>(deps);

    Teuchos::Array<double> species_decay(4);
    species_decay[0] = -1.0;
    species_decay[1] = 0.0;
    species_decay[2] = 1.0;
    species_decay[3] = -0.1;

    Teuchos::ParameterList rad_params;
    rad_params.set("Build Reaction", true);
    rad_params.set("Number of Species", 2);
    Teuchos::ParameterList reaction_params;
    reaction_params.set("Species Decay", species_decay);
    const SpeciesProperties::ConstantSpeciesProperties species_prop(
        rad_params, reaction_params);

    // Initialize class object to test
    auto eval
        = Teuchos::rcp(new ClosureModel::RADReaction<EvalType, panzer::Traits>(
            ir, species_prop));
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
    const double exp_reaction_term_0 = -1.25;
    const double exp_reaction_term_1 = 1.0;

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
TEST(RADReaction2D, Residual)
{
    testEval<panzer::Traits::Residual>(2);
}

//-----------------------------------------------------------------//
TEST(RADReaction2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2);
}

//-----------------------------------------------------------------//
TEST(RADReaction3D, Residual)
{
    testEval<panzer::Traits::Residual>(3);
}

//-----------------------------------------------------------------//
TEST(RADReaction3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(3);
}

} // namespace Test
} // namespace VertexCFD
