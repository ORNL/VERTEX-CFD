#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "rad_solver/closure_models/VertexCFD_Closure_RADAdvectionFlux.hpp"

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> vel_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> vel_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> vel_2;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> species_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> species_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> species_2;

    Dependencies(const panzer::IntegrationRule& ir)
        : vel_0("velocity_0", ir.dl_scalar)
        , vel_1("velocity_1", ir.dl_scalar)
        , vel_2("velocity_2", ir.dl_scalar)
        , species_0("species_0", ir.dl_scalar)
        , species_1("species_1", ir.dl_scalar)
        , species_2("species_2", ir.dl_scalar)
    {
        this->addEvaluatedField(vel_0);
        this->addEvaluatedField(vel_1);
        this->addEvaluatedField(vel_2);

        this->addEvaluatedField(species_0);
        this->addEvaluatedField(species_1);
        this->addEvaluatedField(species_2);

        this->setName("RAD Advection Flux Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData) override
    {
        vel_0.deep_copy(0.25);
        vel_1.deep_copy(0.5);
        vel_2.deep_copy(0.125);

        species_0.deep_copy(1.25);
        species_1.deep_copy(2.5);
        species_2.deep_copy(3.125);
    }
};

template<class EvalType, int NumSpaceDim>
void testEval(const int num_species)
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);
    const auto& ir = *test_fixture.ir;

    const double nan_val = std::numeric_limits<double>::quiet_NaN();

    // Initialize dependents
    auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    Teuchos::ParameterList rad_params;
    rad_params.set("Number of Species", num_species);
    Teuchos::ParameterList reaction_params;
    const SpeciesProperties::ConstantSpeciesProperties species_prop(
        rad_params, reaction_params);

    // Initialize class object to test
    auto eval = Teuchos::rcp(
        new ClosureModel::RADAdvectionFlux<EvalType, panzer::Traits, num_space_dim>(
            ir, species_prop));

    test_fixture.registerEvaluator<EvalType>(eval);
    for (int num = 0; num < num_species; ++num)
        test_fixture.registerTestField<EvalType>(eval->_species_flux[num]);

    test_fixture.evaluate<EvalType>();

    const auto calc_species_flux_0
        = test_fixture.getTestFieldData<EvalType>(eval->_species_flux[0]);
    const auto calc_species_flux_1
        = test_fixture.getTestFieldData<EvalType>(eval->_species_flux[1]);

    const int num_point = ir.num_points;

    // Expected values
    const double exp_species_0_flux[3]
        = {0.3125, 0.625, num_space_dim == 3 ? 0.15625 : nan_val};
    const double exp_species_1_flux[3]
        = {0.625, 1.25, num_space_dim == 3 ? 0.3125 : nan_val};
    const double exp_species_2_flux[3]
        = {0.78125, 1.5625, num_space_dim == 3 ? 0.390625 : nan_val};

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int dim = 0; dim < num_space_dim; dim++)
        {
            EXPECT_EQ(exp_species_0_flux[dim],
                      fieldValue(calc_species_flux_0, 0, qp, dim));
            EXPECT_EQ(exp_species_1_flux[dim],
                      fieldValue(calc_species_flux_1, 0, qp, dim));

            if (num_species == 3)
            {
                const auto calc_species_flux_2
                    = test_fixture.getTestFieldData<EvalType>(
                        eval->_species_flux[2]);
                EXPECT_EQ(exp_species_2_flux[dim],
                          fieldValue(calc_species_flux_2, 0, qp, dim));
            }
        }
    }
}

//-----------------------------------------------------------------//
TEST(RADAdvectionFlux2Species2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(2);
}

//-----------------------------------------------------------------//
TEST(RADAdvectionFlux2Species2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(2);
}

//-----------------------------------------------------------------//
TEST(RADAdvectionFlux2Species3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(2);
}

//-----------------------------------------------------------------//
TEST(RADAdvectionFlux2Species3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(2);
}

//-----------------------------------------------------------------//
TEST(RADAdvectionFlux3Species2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(3);
}

//-----------------------------------------------------------------//
TEST(RADAdvectionFlux3Species2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(3);
}

//-----------------------------------------------------------------//
TEST(RADAdvectionFlux3Species3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(3);
}

//-----------------------------------------------------------------//
TEST(RADAdvectionFlux3Species3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(3);
}

//-----------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD
