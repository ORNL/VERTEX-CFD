#include <VertexCFD_EvaluatorTestHarness.hpp>
#include <closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp>

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleViscousFlux.hpp"
#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
// Continuity Equation cases
enum class ContinuityModel
{
    AC,
    EDAC
};
//---------------------------------------------------------------------------//
// Test data dependencies.
template<class EvalType>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    // quiet_NaN is a host-side function so we store the value
    const double _nanval = std::numeric_limits<double>::quiet_NaN();

    double _u;
    double _v;
    double _w;
    bool _build_temp_equ;
    bool _build_turbulence_model;
    bool _unscaled_density;
    ContinuityModel _continuity_model;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> velocity_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> velocity_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> velocity_2;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> rho;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> nu;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> k;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> cp;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> nu_t;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_2;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        grad_temperature;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        grad_lagrange_pressure;

    Dependencies(const panzer::IntegrationRule& ir,
                 const double u,
                 const double v,
                 const double w,
                 const bool build_temp_equ,
                 const bool build_turbulence_model,
                 const bool unscaled_density,
                 const ContinuityModel continuity_model)
        : _u(u)
        , _v(v)
        , _w(w)
        , _build_temp_equ(build_temp_equ)
        , _build_turbulence_model(build_turbulence_model)
        , _unscaled_density(unscaled_density)
        , _continuity_model(continuity_model)
        , velocity_0("velocity_0", ir.dl_scalar)
        , velocity_1("velocity_1", ir.dl_scalar)
        , velocity_2("velocity_2", ir.dl_scalar)
        , rho("density", ir.dl_scalar)
        , nu("kinematic_viscosity", ir.dl_scalar)
        , k("thermal_conductivity", ir.dl_scalar)
        , cp("specific_heat_capacity", ir.dl_scalar)
        , nu_t("turbulent_eddy_viscosity", ir.dl_scalar)
        , grad_vel_0("GRAD_velocity_0", ir.dl_vector)
        , grad_vel_1("GRAD_velocity_1", ir.dl_vector)
        , grad_vel_2("GRAD_velocity_2", ir.dl_vector)
        , grad_temperature("GRAD_temperature", ir.dl_vector)
        , grad_lagrange_pressure("GRAD_lagrange_pressure", ir.dl_vector)
    {
        this->addEvaluatedField(velocity_0);
        this->addEvaluatedField(velocity_1);
        this->addEvaluatedField(velocity_2);
        this->addEvaluatedField(rho);
        this->addEvaluatedField(nu);

        if (_build_turbulence_model)
        {
            this->addEvaluatedField(nu_t);
        }

        this->addEvaluatedField(grad_vel_0);
        this->addEvaluatedField(grad_vel_1);
        this->addEvaluatedField(grad_vel_2);

        if (_build_temp_equ)
        {
            this->addEvaluatedField(k);
            this->addEvaluatedField(cp);
            this->addEvaluatedField(grad_temperature);
        }

        this->addEvaluatedField(grad_lagrange_pressure);

        this->setName("Incompressible Viscous Flux Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "viscous flux test dependencies",
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
                grad_vel_0(c, qp, dim) = _u * dimqp;
                grad_vel_1(c, qp, dim) = _v * dimqp;
                grad_vel_2(c, qp, dim) = _w * dimqp;
                if (_build_temp_equ)
                {
                    k(c, qp) = 0.5;
                    cp(c, qp) = 0.2;
                    grad_temperature(c, qp, dim) = (_u + _v) * dimqp;
                }

                grad_lagrange_pressure(c, qp, dim)
                    = _continuity_model == ContinuityModel::EDAC
                          ? (_u + _v) * dimqp
                          : _nanval;
            }

            velocity_0(c, qp) = _u;
            velocity_1(c, qp) = _v;
            velocity_2(c, qp) = _w;
            rho(c, qp) = _unscaled_density ? 3.0 : 1.0;
            nu(c, qp) = 0.375;

            if (_build_turbulence_model)
                nu_t(c, qp) = 4.0;
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval(const bool unscaled_density,
              const bool build_temp_equ,
              const bool build_turbulence_model,
              const ContinuityModel continuity_model)
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    auto& ir = *test_fixture.ir;
    const double nan_val = std::numeric_limits<double>::quiet_NaN();

    // Initialize velocity components and dependents
    const double u = 0.25;
    const double v = 0.5;
    const double w = num_space_dim == 3 ? 0.125 : nan_val;
    const double Pr_t = 0.8;

    const auto deps
        = Teuchos::rcp(new Dependencies<EvalType>(ir,
                                                  u,
                                                  v,
                                                  w,
                                                  build_temp_equ,
                                                  build_turbulence_model,
                                                  unscaled_density,
                                                  continuity_model));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize class object to test
    const double nu = 0.375;
    const double cp = 0.2;
    const double beta = 2.0;
    const double kappa = build_temp_equ ? 0.5 : nan_val;
    const double gamma = build_temp_equ ? 3.0 : nan_val;

    Teuchos::ParameterList fluid_prop_list;
    fluid_prop_list.set("Density", unscaled_density ? 3.0 : 1.0);
    fluid_prop_list.set("Kinematic viscosity", nu);
    fluid_prop_list.set("Artificial compressibility", beta);
    fluid_prop_list.set("Build Temperature Equation", build_temp_equ);
    if (build_temp_equ)
    {
        fluid_prop_list.set("Thermal conductivity", kappa);
        fluid_prop_list.set("Specific heat capacity", cp);
        fluid_prop_list.set("Heat Capacity Ratio", gamma);
    }

    Teuchos::ParameterList user_params;
    if (build_turbulence_model && build_temp_equ)
        user_params.set("Turbulent Prandtl Number", Pr_t);
    if (continuity_model == ContinuityModel::EDAC)
        user_params.set("Continuity Model", "EDAC");

    const FluidProperties::ConstantFluidProperties fluid_prop(fluid_prop_list);
    auto eval = Teuchos::rcp(
        new ClosureModel::
            IncompressibleViscousFlux<EvalType, panzer::Traits, num_space_dim>(
                ir, fluid_prop, user_params, build_turbulence_model));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_continuity_flux);
    for (int dim = 0; dim < num_space_dim; ++dim)
        test_fixture.registerTestField<EvalType>(eval->_momentum_flux[dim]);

    test_fixture.evaluate<EvalType>();

    const auto fc_cont
        = test_fixture.getTestFieldData<EvalType>(eval->_continuity_flux);
    const auto fc_mom_0
        = test_fixture.getTestFieldData<EvalType>(eval->_momentum_flux[0]);

    const int num_point = ir.num_points;

    const double exp_cont_flux_3d_ac[3] = {0.0, 0.0, 0.0};
    const double exp_cont_flux_3d_edac_isothermal[3]
        = {unscaled_density ? -0.421875 : -0.140625,
           unscaled_density ? 0.84375 : 0.28125,
           unscaled_density ? -1.265625 : -0.421875};

    const double exp_cont_flux_3d_edac_energy[3]
        = {unscaled_density ? -0.9375 : -2.8125,
           unscaled_density ? 1.875 : 5.625,
           unscaled_density ? -2.8125 : -8.4375};

    const double* exp_cont_flux_3d_edac
        = (build_temp_equ ? exp_cont_flux_3d_edac_energy
                          : exp_cont_flux_3d_edac_isothermal);

    const double exp_cont_flux_2d_ac[3]
        = {exp_cont_flux_3d_ac[0], exp_cont_flux_3d_ac[1], nan_val};

    const double exp_cont_flux_2d_edac[3]
        = {exp_cont_flux_3d_edac[0], exp_cont_flux_3d_edac[1], nan_val};

    const double* exp_cont_flux
        = continuity_model == ContinuityModel::EDAC
              ? (num_space_dim == 3 ? exp_cont_flux_3d_edac
                                    : exp_cont_flux_2d_edac)
              : (num_space_dim == 3 ? exp_cont_flux_3d_ac
                                    : exp_cont_flux_2d_ac);

    const double exp_mom_0_flux_3d[3]
        = {build_turbulence_model ? (unscaled_density ? -3.28125 : -1.09375)
                                  : (unscaled_density ? -0.28125 : -0.09375),
           build_turbulence_model ? (unscaled_density ? 6.5625 : 2.1875)
                                  : (unscaled_density ? 0.5625 : 0.1875),
           build_turbulence_model ? (unscaled_density ? -9.84375 : -3.28125)
                                  : (unscaled_density ? -0.84375 : -0.28125)};
    const double exp_mom_0_flux_2d[3]
        = {exp_mom_0_flux_3d[0], exp_mom_0_flux_3d[1], nan_val};
    const double* exp_mom_0_flux = num_space_dim == 3 ? exp_mom_0_flux_3d
                                                      : exp_mom_0_flux_2d;
    const double exp_mom_1_flux_3d[3]
        = {build_turbulence_model ? (unscaled_density ? -6.5625 : -2.1875)
                                  : (unscaled_density ? -0.5625 : -0.1875),
           build_turbulence_model ? (unscaled_density ? 13.125 : 4.375)
                                  : (unscaled_density ? 1.125 : 0.375),
           build_turbulence_model ? (unscaled_density ? -19.6875 : -6.5625)
                                  : (unscaled_density ? -1.6875 : -0.5625)};
    const double exp_mom_1_flux_2d[3]
        = {exp_mom_1_flux_3d[0], exp_mom_1_flux_3d[1], nan_val};
    const double* exp_mom_1_flux = num_space_dim == 3 ? exp_mom_1_flux_3d
                                                      : exp_mom_1_flux_2d;
    const double exp_mom_2_flux_3d[3] = {
        build_turbulence_model ? (unscaled_density ? -1.640625 : -0.546875)
                               : (unscaled_density ? -0.140625 : -0.046875),
        build_turbulence_model ? (unscaled_density ? 3.28125 : 1.09375)
                               : (unscaled_density ? 0.28125 : 0.09375),
        build_turbulence_model ? (unscaled_density ? -4.921875 : -1.640625)
                               : (unscaled_density ? -0.421875 : -0.140625)};
    const double exp_mom_2_flux_2d[3] = {nan_val, nan_val, nan_val};
    const double* exp_mom_2_flux = num_space_dim == 3 ? exp_mom_2_flux_3d
                                                      : exp_mom_2_flux_2d;

    const double exp_ene_flux_3d[3] = {
        build_turbulence_model ? (unscaled_density ? -2.625 : -1.125) : -0.375,
        build_turbulence_model ? (unscaled_density ? 5.25 : 2.25) : 0.75,
        build_turbulence_model ? (unscaled_density ? -7.875 : -3.375) : -1.125};

    const double exp_ene_flux_2d[3]
        = {exp_ene_flux_3d[0], exp_ene_flux_3d[1], nan_val};
    const double* exp_ene_flux = num_space_dim == 3 ? exp_ene_flux_3d
                                                    : exp_ene_flux_2d;
    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int dim = 0; dim < num_space_dim; dim++)
        {
            EXPECT_DOUBLE_EQ(exp_cont_flux[dim],
                             fieldValue(fc_cont, 0, qp, dim));
            EXPECT_EQ(exp_mom_0_flux[dim], fieldValue(fc_mom_0, 0, qp, dim));
            const auto fc_mom_1 = test_fixture.getTestFieldData<EvalType>(
                eval->_momentum_flux[1]);
            EXPECT_EQ(exp_mom_1_flux[dim], fieldValue(fc_mom_1, 0, qp, dim));
            if (num_space_dim > 2) // 3D mesh
            {
                const auto fc_mom_2 = test_fixture.getTestFieldData<EvalType>(
                    eval->_momentum_flux[2]);
                EXPECT_EQ(exp_mom_2_flux[dim],
                          fieldValue(fc_mom_2, 0, qp, dim));
            }
            if (build_temp_equ)
            {
                const auto fc_ene = test_fixture.getTestFieldData<EvalType>(
                    eval->_energy_flux);
                EXPECT_DOUBLE_EQ(exp_ene_flux[dim],
                                 fieldValue(fc_ene, 0, qp, dim));
            }
        }
    }
}

//-----------------------------------------------------------------//
struct IncompressibleViscousFluxTestParams
{
    std::string test_name;
    bool unscaled_density;
    bool build_temp_equ;
    bool build_turbulence_model;
    ContinuityModel continuity_model;
    int num_space_dim;
};

class IncompressibleViscousFluxTest
    : public testing::TestWithParam<IncompressibleViscousFluxTestParams>
{
  public:
    struct PrintToStringParamName
    {
        template<class T>
        std::string operator()(const testing::TestParamInfo<T>& info) const
        {
            auto testParam
                = static_cast<IncompressibleViscousFluxTestParams>(info.param);
            return testParam.test_name;
        }
    };
};

//-----------------------------------------------------------------//
TEST_P(IncompressibleViscousFluxTest, cartesian)
{
    const auto params = GetParam();
    if (std::string::npos != params.test_name.find("residual"))
    {
        if (params.num_space_dim == 2)
        {
            testEval<panzer::Traits::Residual, 2>(params.unscaled_density,
                                                  params.build_temp_equ,
                                                  params.build_turbulence_model,
                                                  params.continuity_model);
        }
        else
        {
            testEval<panzer::Traits::Residual, 3>(params.unscaled_density,
                                                  params.build_temp_equ,
                                                  params.build_turbulence_model,
                                                  params.continuity_model);
        }
    }
    else if (std::string::npos != params.test_name.find("jacobian"))
    {
        if (params.num_space_dim == 2)
        {
            testEval<panzer::Traits::Jacobian, 2>(params.unscaled_density,
                                                  params.build_temp_equ,
                                                  params.build_turbulence_model,
                                                  params.continuity_model);
        }
        else
        {
            testEval<panzer::Traits::Jacobian, 3>(params.unscaled_density,
                                                  params.build_temp_equ,
                                                  params.build_turbulence_model,
                                                  params.continuity_model);
        }
    }
}

//-----------------------------------------------------------------//
INSTANTIATE_TEST_SUITE_P(
    Test,
    IncompressibleViscousFluxTest,
    testing::Values(
        IncompressibleViscousFluxTestParams{"ScaledDensityIsothermal2D_"
                                            "residual",
                                            false,
                                            false,
                                            false,
                                            ContinuityModel::AC,
                                            2},
        IncompressibleViscousFluxTestParams{"ScaledDensityIsothermal2D_"
                                            "jacobian",
                                            false,
                                            false,
                                            false,
                                            ContinuityModel::AC,
                                            2},
        IncompressibleViscousFluxTestParams{"ScaledDensity2D_residual",
                                            false,
                                            true,
                                            false,
                                            ContinuityModel::AC,
                                            2},
        IncompressibleViscousFluxTestParams{"ScaledDensity2D_jacobian",
                                            false,
                                            true,
                                            false,
                                            ContinuityModel::AC,
                                            2},
        IncompressibleViscousFluxTestParams{"ScaledDensityIsothermal3D_"
                                            "residual",
                                            false,
                                            false,
                                            false,
                                            ContinuityModel::AC,
                                            3},
        IncompressibleViscousFluxTestParams{"ScaledDensityIsothermal3D_"
                                            "jacobian",
                                            false,
                                            false,
                                            false,
                                            ContinuityModel::AC,
                                            3},
        IncompressibleViscousFluxTestParams{"UnScaledDensityIsothermal3D_"
                                            "residual",
                                            true,
                                            false,
                                            false,
                                            ContinuityModel::AC,
                                            3},
        IncompressibleViscousFluxTestParams{"UnScaledDensityIsothermal3D_"
                                            "jacobian",
                                            true,
                                            false,
                                            false,
                                            ContinuityModel::AC,
                                            3},
        IncompressibleViscousFluxTestParams{"ScaledDensity3D_residual",
                                            false,
                                            true,
                                            false,
                                            ContinuityModel::AC,
                                            3},
        IncompressibleViscousFluxTestParams{"ScaledDensity3D_jacobian",
                                            false,
                                            true,
                                            false,
                                            ContinuityModel::AC,
                                            3},
        IncompressibleViscousFluxTestParams{"TurbulentScaledDensity3D_"
                                            "residual",
                                            false,
                                            true,
                                            true,
                                            ContinuityModel::AC,
                                            3},
        IncompressibleViscousFluxTestParams{"TurbulentScaledDensity3D_"
                                            "jacobian",
                                            false,
                                            true,
                                            true,
                                            ContinuityModel::AC,
                                            3},
        IncompressibleViscousFluxTestParams{"TurbulentUnScaledDensity3D_"
                                            "residual",
                                            true,
                                            true,
                                            true,
                                            ContinuityModel::AC,
                                            3},
        IncompressibleViscousFluxTestParams{"TurbulentUnScaledDensity3D_"
                                            "jacobian",
                                            true,
                                            true,
                                            true,
                                            ContinuityModel::AC,
                                            3},
        IncompressibleViscousFluxTestParams{"ScaledDensityIsothermalEDAC2D_"
                                            "residual",
                                            false,
                                            false,
                                            false,
                                            ContinuityModel::EDAC,
                                            2},
        IncompressibleViscousFluxTestParams{"ScaledDensityIsothermalEDAC2D_"
                                            "jacobian",
                                            false,
                                            false,
                                            false,
                                            ContinuityModel::EDAC,
                                            2},
        IncompressibleViscousFluxTestParams{"ScaledDensityEDAC2D_residual",
                                            false,
                                            true,
                                            false,
                                            ContinuityModel::EDAC,
                                            2},
        IncompressibleViscousFluxTestParams{"ScaledDensityEDAC2D_jacobian",
                                            false,
                                            true,
                                            false,
                                            ContinuityModel::EDAC,
                                            2},
        IncompressibleViscousFluxTestParams{"ScaledDensityIsothermalEDAC3D_"
                                            "residual",
                                            false,
                                            false,
                                            false,
                                            ContinuityModel::EDAC,
                                            3},
        IncompressibleViscousFluxTestParams{"ScaledDensityIsothermalEDAC3D_"
                                            "jacobian",
                                            false,
                                            false,
                                            false,
                                            ContinuityModel::EDAC,
                                            3},
        IncompressibleViscousFluxTestParams{"UnScaledDensityIsothermalEDAC3D_"
                                            "residual",
                                            true,
                                            false,
                                            false,
                                            ContinuityModel::EDAC,
                                            3},
        IncompressibleViscousFluxTestParams{"UnScaledDensityIsothermalEDAC3D_"
                                            "jacobian",
                                            true,
                                            false,
                                            false,
                                            ContinuityModel::EDAC,
                                            3},
        IncompressibleViscousFluxTestParams{"ScaledDensityEDAC3D_residual",
                                            false,
                                            true,
                                            false,
                                            ContinuityModel::EDAC,
                                            3},
        IncompressibleViscousFluxTestParams{"ScaledDensityEDAC3D_jacobian",
                                            false,
                                            true,
                                            false,
                                            ContinuityModel::EDAC,
                                            3},
        IncompressibleViscousFluxTestParams{"TurbulentScaledDensityEDAC3D_"
                                            "residual",
                                            false,
                                            true,
                                            true,
                                            ContinuityModel::EDAC,
                                            3},
        IncompressibleViscousFluxTestParams{"TurbulentScaledDensityEDAC3D_"
                                            "jacobian",
                                            false,
                                            true,
                                            true,
                                            ContinuityModel::EDAC,
                                            3},
        IncompressibleViscousFluxTestParams{"TurbulentUnScaledDensityEDAC3D_"
                                            "residual",
                                            true,
                                            true,
                                            true,
                                            ContinuityModel::EDAC,
                                            3},
        IncompressibleViscousFluxTestParams{"TurbulentUnScaledDensityEDAC3D_"
                                            "jacobian",
                                            true,
                                            true,
                                            true,
                                            ContinuityModel::EDAC,
                                            3}),
    IncompressibleViscousFluxTest::PrintToStringParamName());

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.user_params.set("Build Temperature Equation", false);
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.type_name = "IncompressibleViscousFlux";
    test_fixture.eval_name = "Incompressible Viscous Flux "
                             + std::to_string(num_space_dim) + "D";
    test_fixture.template buildAndTest<
        ClosureModel::IncompressibleViscousFlux<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(IncompressibleViscousFlux_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(IncompressibleViscousFlux_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(IncompressibleViscousFlux_Factory3D, residual_test)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(IncompressibleViscousFlux_Factory3D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
