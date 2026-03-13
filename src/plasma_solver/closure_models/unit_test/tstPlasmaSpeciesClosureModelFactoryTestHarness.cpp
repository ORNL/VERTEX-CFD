#include "plasma_solver/closure_models/VertexCFD_PlasmaSpeciesClosureModelFactoryTestHarness.hpp"

#include <Panzer_Traits.hpp>

namespace VertexCFD
{
namespace Test
{

//---------------------------------------------------------------------------//
template<class EvalType>
void testDefaultFixture()
{
    PlasmaSpeciesClosureModelFactoryTestFixture<EvalType> test_fixture;

    EXPECT_EQ("!!! UNDEFINED !!!", test_fixture.type_name);
    EXPECT_EQ("!!! UNDEFINED !!!", test_fixture.eval_name);

    // The default type_name should not match any closure model and throw an
    // error message
    const std::string exp_msg
        = "\n\nPlasma Species closure model/type Model Data/!!! UNDEFINED !!! "
          "failed to build in model id Test Model.\nThe closure models "
          "available in Vertex-CFD are:\nDummy\n\n";

    // Check error.
    EXPECT_THROW(
        try {
            (test_fixture.template buildAndTest<void, 2>());
        } catch (const std::runtime_error& e) {
            EXPECT_EQ(exp_msg, e.what());
            throw;
        },
        std::runtime_error);
}

TEST(PlasmaSpeciesClosureModelFactoryHarness, Residual)
{
    testDefaultFixture<panzer::Traits::Residual>();
}

TEST(PlasmaSpeciesClosureModelFactoryHarness, Jacobian)
{
    testDefaultFixture<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//
template<class EvalType>
void testDefaultClosureModel()
{
    PlasmaSpeciesClosureModelFactoryTestFixture<EvalType> test_fixture;

    test_fixture.type_name = "PlasmaSpeciesTimeDerivative";
    EXPECT_EQ("!!! UNDEFINED !!!", test_fixture.eval_name);

    // The default closure model should be flagged by the logic and throw an
    // error message
    const std::string exp_msg
        = "\n\nThe closure model 'PlasmaSpeciesTimeDerivative' was found in "
          "the "
          "input file.'PlasmaSpeciesTimeDerivative' is a default closure "
          "model and "
          "should be not listed in the input file. The list of default "
          "closure models is provided "
          "below:"
          "\nPlasmaSpeciesTimeDerivative\nPlasmaSpeciesConvectiveFlux\nPlasmaS"
          "pecies"
          "EOS\n"
          "\n";

    // Check error.
    EXPECT_THROW(
        try {
            (test_fixture.template buildAndTest<void, 2>());
        } catch (const std::runtime_error& e) {
            EXPECT_EQ(exp_msg, e.what());
            throw;
        },
        std::runtime_error);
}

TEST(PlasmaSpeciesDefaultClosureModelFactoryHarness, Residual)
{
    testDefaultClosureModel<panzer::Traits::Residual>();
}

TEST(PlasmaSpeciesDefaultClosureModelFactoryHarness, Jacobian)
{
    testDefaultClosureModel<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//
template<class EvalType>
void testMissingSpeciesProperties()
{
    PlasmaSpeciesClosureModelFactoryTestFixture<EvalType> test_fixture;

    EXPECT_EQ("!!! UNDEFINED !!!", test_fixture.type_name);
    EXPECT_EQ("!!! UNDEFINED !!!", test_fixture.eval_name);

    // A missing species properties sublist should return an error message.
    const std::string exp_msg
        = "\n\nThe sublist 'Species Properties' is not found in the closure "
          "model list with model id 'Test Model'.\n\n";

    // Check error.
    EXPECT_THROW(
        try {
            (test_fixture.template buildAndTest<void, 2>(false));
        } catch (const std::runtime_error& e) {
            EXPECT_EQ(exp_msg, e.what());
            throw;
        },
        std::runtime_error);
}

TEST(PlasmaSpeciesMissingSpeciesPropertiesFactoryHarness, Residual)
{
    testMissingSpeciesProperties<panzer::Traits::Residual>();
}

TEST(PlasmaSpeciesMissingSpeciesPropertiesFactoryHarness, Jacobian)
{
    testMissingSpeciesProperties<panzer::Traits::Jacobian>();
}

} // namespace Test
} // namespace VertexCFD
