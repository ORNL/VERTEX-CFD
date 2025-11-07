#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleChienKEpsilonEddyViscosity.hpp"

#include <Panzer_GlobalData.hpp>
#include <Panzer_ParameterLibrary.hpp>
#include <Panzer_ScalarParameterEntry.hpp>
#include <Teuchos_RCP.hpp>

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
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> distance;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> nu;

    Dependencies(const panzer::IntegrationRule& ir)
        : turb_kinetic_energy("turb_kinetic_energy", ir.dl_scalar)
        , turb_dissipation_rate("turb_dissipation_rate", ir.dl_scalar)
        , distance("distance", ir.dl_scalar)
        , nu("kinematic_viscosity", ir.dl_scalar)
    {
        this->addEvaluatedField(turb_kinetic_energy);
        this->addEvaluatedField(turb_dissipation_rate);
        this->addEvaluatedField(distance);
        this->addEvaluatedField(nu);
        this->setName(
            "Chien K-Epsilon Incompressible Eddy Viscosity Unit "
            "Test "
            "Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData /**d**/) override
    {
        turb_kinetic_energy.deep_copy(40.0);
        turb_dissipation_rate.deep_copy(5.0);
        distance.deep_copy(0.5);
        nu.deep_copy(0.0115);
    }
};

template<class EvalType>
void testEval(const bool non_unit_area)
{
    using std::pow;
    const int num_space_dim = 2;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;
    const double surface_area = non_unit_area ? 0.0001 : 1.0;
    // Eval dependencies
    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Wall Shear Stress
    auto global_data = panzer::createGlobalData();
    auto& pl = *global_data->pl;
    pl.addParameterFamily("Friction Velocity - friction_velocity", true, false);
    Teuchos::RCP<panzer::ScalarParameterEntry<EvalType>> entry;
    entry = Teuchos::rcp(new panzer::ScalarParameterEntry<EvalType>);
    entry->setValue(4.0);
    pl.addEntry<EvalType>("Friction Velocity - friction_velocity", entry);

    // User parameters
    Teuchos::ParameterList user_params;
    user_params.sublist("Turbulence Parameters")
        .set<double>("Boundary Surface Area", surface_area);

    // Initialize and register
    auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleChienKEpsilonEddyViscosity<EvalType,
                                                                   panzer::Traits>(
            ir, global_data, user_params));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_nu_t);
    test_fixture.evaluate<EvalType>();

    // Evaluate test fields
    const auto fv_nu_t = test_fixture.getTestFieldData<EvalType>(eval->_nu_t);

    // Expected values
    const int num_point = ir.num_points;
    const double exp_eddy_visc = non_unit_area ? 28.8 : 24.902343842785555;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_eddy_visc, fieldValue(fv_nu_t, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleChienKEpsilonEddyViscosity, Residual)
{
    testEval<panzer::Traits::Residual>(false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleChienKEpsilonEddyViscosity, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleChienKEpsilonEddyViscosityNonUnitArea, Residual)
{
    testEval<panzer::Traits::Residual>(true);
}

//-----------------------------------------------------------------//
TEST(IncompressibleChienKEpsilonEddyViscosityNonUnitArea, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(true);
}
//-----------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD
