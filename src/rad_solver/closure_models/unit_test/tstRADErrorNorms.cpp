#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include "rad_solver/closure_models/VertexCFD_Closure_RADErrorNorms.hpp"
#include "rad_solver/species_properties/VertexCFD_ConstantSpeciesProperties.hpp"

#include <gtest/gtest.h>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
template<class EvalType>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    // exact solution
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>, 3>
        exact_species;

    // numerical solution
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>, 3>
        species;

    Dependencies(const panzer::IntegrationRule& ir)
    {
        // exact solution
        Utils::addEvaluatedVectorField(
            *this, ir.dl_scalar, exact_species, "Exact_species_");

        // numerical solution
        Utils::addEvaluatedVectorField(
            *this, ir.dl_scalar, species, "species_");

        this->setName("RAD Error Norms Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData) override
    {
        // assign prescribed values to exact and numerical solution
        for (int i = 0; i < 3; ++i)
        {
            exact_species[i].deep_copy(0.3 + 0.1 * i);
            species[i].deep_copy(0.7 + 0.2 * i);
        }
    }
};

template<class EvalType>
void testEval()
{
    // Setup test fixture.
    const int integration_order = 0;
    const int basis_order = 1;
    // NumSpaceDim=2 is not required for closure model yet it is required for
    // closure model factory and the test harness.
    EvaluatorTestFixture test_fixture(2, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    Teuchos::ParameterList rad_params;
    rad_params.set("Number of Species", 3);
    Teuchos::ParameterList reaction_params;
    const SpeciesProperties::ConstantSpeciesProperties species_prop(
        rad_params, reaction_params);

    // Create evaluator.
    const auto eval = Teuchos::rcp(
        new ClosureModel::RADErrorNorms<EvalType, panzer::Traits>(
            ir, species_prop));
    test_fixture.registerEvaluator<EvalType>(eval);

    // Add required test fields.
    for (int num = 0; num < 3; ++num)
    {
        test_fixture.registerTestField<EvalType>(eval->_L1_error_species[num]);
        test_fixture.registerTestField<EvalType>(eval->_L2_error_species[num]);
    }

    // Evaluate
    test_fixture.evaluate<EvalType>();

    // Reference values
    const double exp_L1_error_species[3]
        = {0.39999999999999997, 0.5, 0.6000000000000001};
    const double exp_L2_error_species[3]
        = {0.15999999999999998, 0.25, 0.3600000000000001};

    // Check the L1/L2 error solutions
    const int num_point = ir.num_points;
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int num = 0; num < 3; ++num)
        {
            const auto calc_L1_error_species
                = test_fixture.getTestFieldData<EvalType>(
                    eval->_L1_error_species[num]);
            const auto calc_L2_error_species
                = test_fixture.getTestFieldData<EvalType>(
                    eval->_L2_error_species[num]);
            EXPECT_DOUBLE_EQ(exp_L1_error_species[num],
                             fieldValue(calc_L1_error_species, 0, qp));
            EXPECT_DOUBLE_EQ(exp_L2_error_species[num],
                             fieldValue(calc_L2_error_species, 0, qp));
        }
    }
}

//---------------------------------------------------------------------------//
TEST(RAD_L1L2_Error, Residual)
{
    testEval<panzer::Traits::Residual>();
}

//---------------------------------------------------------------------------//
TEST(RAD_L1L2_Error, Jacobian)
{
    testEval<panzer::Traits::Jacobian>();
}

} // end namespace Test
} // end namespace VertexCFD
