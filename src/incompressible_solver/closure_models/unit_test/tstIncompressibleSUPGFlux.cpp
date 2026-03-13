#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleSUPGFlux.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
// Test data dependencies.
template<class EvalType>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    // quiet_NaN is a host-side function so we store the value
    const double _nanval = std::numeric_limits<double>::quiet_NaN();

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> dxdt_vel_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> dxdt_vel_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> dxdt_vel_2;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> velocity_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> velocity_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> velocity_2;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> tau_supg_cont;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> tau_supg_mom;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> mom_source_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> mom_source_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> mom_source_2;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> rho;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> cp;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        grad_lag_press;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_2;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> dxdt_temp;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> temp;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> energy_source;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> tau_supg_energy;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_temp;

    Dependencies(const panzer::IntegrationRule& ir)
        : dxdt_vel_0("DXDT_velocity_0", ir.dl_scalar)
        , dxdt_vel_1("DXDT_velocity_1", ir.dl_scalar)
        , dxdt_vel_2("DXDT_velocity_2", ir.dl_scalar)
        , velocity_0("velocity_0", ir.dl_scalar)
        , velocity_1("velocity_1", ir.dl_scalar)
        , velocity_2("velocity_2", ir.dl_scalar)
        , tau_supg_cont("tau_supg_continuity", ir.dl_scalar)
        , tau_supg_mom("tau_supg_momentum", ir.dl_scalar)
        , mom_source_0("CONSTANT_SOURCE_momentum_0", ir.dl_scalar)
        , mom_source_1("CONSTANT_SOURCE_momentum_1", ir.dl_scalar)
        , mom_source_2("CONSTANT_SOURCE_momentum_2", ir.dl_scalar)
        , rho("density", ir.dl_scalar)
        , cp("specific_heat_capacity", ir.dl_scalar)
        , grad_lag_press("GRAD_lagrange_pressure", ir.dl_vector)
        , grad_vel_0("GRAD_velocity_0", ir.dl_vector)
        , grad_vel_1("GRAD_velocity_1", ir.dl_vector)
        , grad_vel_2("GRAD_velocity_2", ir.dl_vector)
        , dxdt_temp("DXDT_temperature", ir.dl_scalar)
        , temp("temperature", ir.dl_scalar)
        , energy_source("CONSTANT_SOURCE_energy", ir.dl_scalar)
        , tau_supg_energy("tau_supg_energy", ir.dl_scalar)
        , grad_temp("GRAD_temperature", ir.dl_vector)
    {
        this->addEvaluatedField(dxdt_vel_0);
        this->addEvaluatedField(dxdt_vel_1);
        this->addEvaluatedField(dxdt_vel_2);
        this->addEvaluatedField(velocity_0);
        this->addEvaluatedField(velocity_1);
        this->addEvaluatedField(velocity_2);
        this->addEvaluatedField(tau_supg_cont);
        this->addEvaluatedField(tau_supg_mom);
        this->addEvaluatedField(mom_source_0);
        this->addEvaluatedField(mom_source_1);
        this->addEvaluatedField(mom_source_2);
        this->addEvaluatedField(rho);
        this->addEvaluatedField(cp);
        this->addEvaluatedField(grad_lag_press);
        this->addEvaluatedField(grad_vel_0);
        this->addEvaluatedField(grad_vel_1);
        this->addEvaluatedField(grad_vel_2);
        this->addEvaluatedField(dxdt_temp);
        this->addEvaluatedField(temp);
        this->addEvaluatedField(energy_source);
        this->addEvaluatedField(tau_supg_energy);
        this->addEvaluatedField(grad_temp);

        this->setName("Incompressible SUPG Flux Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "supg flux test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = grad_temp.extent(1);
        const int num_space_dim = grad_temp.extent(2);
        using std::pow;
        for (int qp = 0; qp < num_point; ++qp)
        {
            dxdt_vel_0(c, qp) = 1.1;
            dxdt_vel_1(c, qp) = 1.2;
            dxdt_vel_2(c, qp) = num_space_dim == 3 ? 1.3 : _nanval;
            velocity_0(c, qp) = 1.5;
            velocity_1(c, qp) = 2.5;
            velocity_2(c, qp) = num_space_dim == 3 ? 3.5 : _nanval;
            tau_supg_cont(c, qp) = 3.5;
            tau_supg_mom(c, qp) = 3.6;
            mom_source_0(c, qp) = 4.1;
            mom_source_1(c, qp) = 4.2;
            mom_source_2(c, qp) = num_space_dim == 3 ? 4.3 : _nanval;
            rho(c, qp) = 1.0;
            cp(c, qp) = 10.0;
            dxdt_temp(c, qp) = 4.0;
            temp(c, qp) = 5.0;
            energy_source(c, qp) = 6.0;
            tau_supg_energy(c, qp) = 7.0;
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                const int sign = pow(-1, dim + 1);
                grad_vel_0(c, qp, dim) = 2.1 * (dim + 1) * sign;
                grad_vel_1(c, qp, dim) = 2.2 * (dim + 1) * sign;
                grad_vel_2(c, qp, dim)
                    = num_space_dim == 3 ? 2.3 * (dim + 1) * sign : _nanval;
                grad_lag_press(c, qp, dim) = 3.1 * (dim + 1) * sign;
                grad_temp(c, qp, dim) = 8.0 * (dim + 1) * sign;
            }
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval()
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    auto& ir = *test_fixture.ir;

    // Initialize dependents
    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize class object to test
    Teuchos::ParameterList closure_params;
    closure_params.set("Prefix for source terms", "CONSTANT");
    Teuchos::ParameterList fluid_params;
    fluid_params.set("Artificial compressibility", 2.0);
    fluid_params.set("Build Temperature Equation", true);

    auto eval = Teuchos::rcp(
        new ClosureModel::
            IncompressibleSUPGFlux<EvalType, panzer::Traits, num_space_dim>(
                ir, fluid_params, closure_params));

    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_continuity_flux);
    for (int dim = 0; dim < num_space_dim; ++dim)
        test_fixture.registerTestField<EvalType>(eval->_momentum_flux[dim]);
    test_fixture.registerTestField<EvalType>(eval->_energy_flux);
    test_fixture.evaluate<EvalType>();

    const auto fc_cont
        = test_fixture.getTestFieldData<EvalType>(eval->_continuity_flux);
    const auto fc_mom_0
        = test_fixture.getTestFieldData<EvalType>(eval->_momentum_flux[0]);
    const auto fc_mom_1
        = test_fixture.getTestFieldData<EvalType>(eval->_momentum_flux[1]);
    const auto fc_ene
        = test_fixture.getTestFieldData<EvalType>(eval->_energy_flux);

    const int num_point = ir.num_points;

    const double exp_cont_flux_2d[2] = {4.375, 38.14999999999999};
    const double exp_cont_flux_3d[3]
        = {-72.80000000000001, -42.699999999999996, -99.4};
    const double* exp_cont_flux = num_space_dim == 3 ? exp_cont_flux_3d
                                                     : exp_cont_flux_2d;
    const double exp_mom_0_flux_2d[2] = {6.75, 11.25};
    const double exp_mom_0_flux_3d[3]
        = {-112.32000000000004, -187.20000000000005, -262.0800000000001};
    const double* exp_mom_0_flux = num_space_dim == 3 ? exp_mom_0_flux_3d
                                                      : exp_mom_0_flux_2d;
    const double exp_mom_1_flux_2d[2] = {58.85999999999999, 98.1};
    const double exp_mom_1_flux_3d[3] = {-65.88, -109.80000000000001, -153.72};
    const double* exp_mom_1_flux = num_space_dim == 3 ? exp_mom_1_flux_3d
                                                      : exp_mom_1_flux_2d;
    const double exp_mom_2_flux[3]
        = {-153.36, -255.60000000000002, -357.84000000000003};

    const double exp_ene_flux_2d[2] = {3297.0, 5495.0};
    const double exp_ene_flux_3d[3] = {-5523.0, -9205.0, -12887.0};
    const double* exp_ene_flux = num_space_dim == 3 ? exp_ene_flux_3d
                                                    : exp_ene_flux_2d;
    // Assert values
    const double tol = 1.0e-10;
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int dim = 0; dim < num_space_dim; dim++)
        {
            EXPECT_NEAR(
                exp_cont_flux[dim], fieldValue(fc_cont, 0, qp, dim), tol);
            EXPECT_NEAR(
                exp_mom_0_flux[dim], fieldValue(fc_mom_0, 0, qp, dim), tol);
            EXPECT_DOUBLE_EQ(exp_mom_1_flux[dim],
                             fieldValue(fc_mom_1, 0, qp, dim));
            if (num_space_dim > 2) // 3D mesh
            {
                const auto fc_mom_2 = test_fixture.getTestFieldData<EvalType>(
                    eval->_momentum_flux[2]);
                EXPECT_DOUBLE_EQ(exp_mom_2_flux[dim],
                                 fieldValue(fc_mom_2, 0, qp, dim));
            }

            EXPECT_DOUBLE_EQ(exp_ene_flux[dim], fieldValue(fc_ene, 0, qp, dim));
        }
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleSUPGFlux2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleSUPGFlux2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleSUPGFlux3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleSUPGFlux3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.type_name = "IncompressibleSUPGFlux";
    test_fixture.eval_name = "Incompressible SUPG Flux "
                             + std::to_string(num_space_dim) + "D";
    test_fixture.template buildAndTest<
        ClosureModel::IncompressibleSUPGFlux<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(IncompressibleSUPGFlux_Factory2D, Residual)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(IncompressibleSUPGFlux_Factory2D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(IncompressibleSUPGFlux_Factory3D, Residual)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(IncompressibleSUPGFlux_Factory3D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
