#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleChienKEpsilonSource.hpp"

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
    const double _nanval = std::numeric_limits<double>::quiet_NaN();

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_2;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> nu;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> nu_t;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> turb_kinetic_energy;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> turb_dissipation_rate;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> distance;

    Dependencies(const panzer::IntegrationRule& ir)
        : grad_vel_0("GRAD_velocity_0", ir.dl_vector)
        , grad_vel_1("GRAD_velocity_1", ir.dl_vector)
        , grad_vel_2("GRAD_velocity_2", ir.dl_vector)
        , nu("kinematic_viscosity", ir.dl_scalar)
        , nu_t("turbulent_eddy_viscosity", ir.dl_scalar)
        , turb_kinetic_energy("turb_kinetic_energy", ir.dl_scalar)
        , turb_dissipation_rate("turb_dissipation_rate", ir.dl_scalar)
        , distance("distance", ir.dl_scalar)
    {
        this->addEvaluatedField(grad_vel_0);
        this->addEvaluatedField(grad_vel_1);
        this->addEvaluatedField(grad_vel_2);
        this->addEvaluatedField(nu);
        this->addEvaluatedField(nu_t);
        this->addEvaluatedField(turb_kinetic_energy);
        this->addEvaluatedField(turb_dissipation_rate);
        this->addEvaluatedField(distance);
        this->setName(
            "Chien K-Epsilon Incompressible Source Unit "
            "Test "
            "Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "Chien K-Epsilon source test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = grad_vel_0.extent(1);
        const int num_space_dim = grad_vel_0.extent(2);
        using std::pow;

        for (int qp = 0; qp < num_point; ++qp)
        {
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                const int sign = pow(-1, dim + 1);
                const int dimqp = (dim + 1) * sign;
                grad_vel_0(c, qp, dim) = 0.250 * dimqp;
                grad_vel_1(c, qp, dim) = 0.500 * dimqp;
                grad_vel_2(c, qp, dim) = num_space_dim == 3 ? 0.125 * dimqp
                                                            : _nanval;
            }

            nu(c, qp) = 2.0;
            nu_t(c, qp) = 3.0;
            turb_kinetic_energy(c, qp) = 4.0;
            turb_dissipation_rate(c, qp) = 5.0;
            distance(c, qp) = 0.5;
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval()
{
    const int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

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
        .set<double>("Boundary Surface Area", 10000.0);

    // Initialize and register
    auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleChienKEpsilonSource<EvalType,
                                                            panzer::Traits,
                                                            NumSpaceDim>(
            ir, global_data, user_params));

    // Initialize and register
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_k_source);
    test_fixture.registerTestField<EvalType>(eval->_e_source);
    test_fixture.evaluate<EvalType>();

    // Evaluate test fields
    const auto fv_var_k
        = test_fixture.getTestFieldData<EvalType>(eval->_k_source);
    const auto fv_var_e
        = test_fixture.getTestFieldData<EvalType>(eval->_e_source);

    // Expected values
    const double exp_k_source = num_space_dim == 3 ? -54.796875 : -62.625;
    const double exp_e_source = num_space_dim == 3 ? -64.97311461727274
                                                   : -78.18307555477274;

    // Assert values
    const int num_point = ir.num_points;
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_k_source, fieldValue(fv_var_k, 0, qp));
        EXPECT_DOUBLE_EQ(exp_e_source, fieldValue(fv_var_e, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleChienKEpsilonSource2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleChienKEpsilonSource2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleChienKEpsilonSource3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleChienKEpsilonSource3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//-----------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD
