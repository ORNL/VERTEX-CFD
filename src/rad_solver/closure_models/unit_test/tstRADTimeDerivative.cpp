#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "rad_solver/closure_models/VertexCFD_Closure_RADTimeDerivative.hpp"

#include <Panzer_Dimension.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>

#include <Teuchos_RCP.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dvdt_species_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dvdt_species_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dvdt_species_2;

    Dependencies(const panzer::IntegrationRule& ir)
        : _dvdt_species_0("DXDT_species_0", ir.dl_scalar)
        , _dvdt_species_1("DXDT_species_1", ir.dl_scalar)
        , _dvdt_species_2("DXDT_species_2", ir.dl_scalar)
    {
        this->addEvaluatedField(_dvdt_species_0);
        this->addEvaluatedField(_dvdt_species_1);
        this->addEvaluatedField(_dvdt_species_2);

        this->setName("RAD Time Derivative Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData) override
    {
        _dvdt_species_0.deep_copy(1.25);
        _dvdt_species_1.deep_copy(1.5);
        _dvdt_species_2.deep_copy(1.75);
    }
};

//---------------------------------------------------------------------------//
template<class EvalType>
void testEval(const int num_space_dim, const int num_species)
{
    const double nan_val = std::numeric_limits<double>::quiet_NaN();

    // Setup test fixture.
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Eval dependencies.
    auto dep_eval = Teuchos::rcp(new Dependencies<EvalType>(*test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    Teuchos::ParameterList rad_params;
    rad_params.set("Number of Species", num_species);
    Teuchos::ParameterList reaction_params;
    const SpeciesProperties::ConstantSpeciesProperties species_prop(
        rad_params, reaction_params);

    // Create test evaluator.
    auto dqdt_eval = Teuchos::rcp(
        new ClosureModel::RADTimeDerivative<EvalType, panzer::Traits>(
            *test_fixture.ir, species_prop));
    test_fixture.registerEvaluator<EvalType>(dqdt_eval);

    // Add required test fields.
    for (int num = 0; num < num_species; num++)
    {
        test_fixture.registerTestField<EvalType>(dqdt_eval->_dqdt_rad[num]);
    }

    // Evaluate test fields.
    test_fixture.evaluate<EvalType>();

    // Expected species values
    const double exp_species[3]
        = {1.25, 1.5, num_species == 3 ? 1.75 : nan_val};

    // Check the test fields.
    for (int num = 0; num < num_species; num++)
    {
        const auto calc_species = test_fixture.getTestFieldData<EvalType>(
            dqdt_eval->_dqdt_rad[num]);
        EXPECT_DOUBLE_EQ(exp_species[num], fieldValue(calc_species, 0, 0));
    }
} // namespace Test

//---------------------------------------------------------------------------//
TEST(RADTimeDerivative2Species2D, Residual)
{
    testEval<panzer::Traits::Residual>(2, 2);
}

//---------------------------------------------------------------------------//
TEST(RADTimeDerivative2Species2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2, 2);
}

//---------------------------------------------------------------------------//
TEST(RADTimeDerivative2Species3D, Residual)
{
    testEval<panzer::Traits::Residual>(3, 2);
}

//---------------------------------------------------------------------------//
TEST(RADTimeDerivative2Species3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(3, 2);
}

//---------------------------------------------------------------------------//
TEST(RADTimeDerivative3Species2D, Residual)
{
    testEval<panzer::Traits::Residual>(2, 3);
}

//---------------------------------------------------------------------------//
TEST(RADTimeDerivative3Species2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2, 3);
}

//---------------------------------------------------------------------------//
TEST(RADTimeDerivative3Species3D, Residual)
{
    testEval<panzer::Traits::Residual>(3, 3);
}

//---------------------------------------------------------------------------//
TEST(RADTimeDerivative3Species3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(3, 3);
}

//---------------------------------------------------------------------------//
} // namespace Test
} // end namespace VertexCFD
