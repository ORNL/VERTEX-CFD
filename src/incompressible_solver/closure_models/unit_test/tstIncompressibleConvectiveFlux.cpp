#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleConvectiveFlux.hpp"
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
    EDAC,
    EDACTempNC
};
//---------------------------------------------------------------------------//
// Test data dependencies.
template<class EvalType>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    double _u;
    double _v;
    double _w;
    bool _build_temp;
    ContinuityModel _continuity_model;
    bool _unscaled_density;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> lagrange_pressure;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> rho;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> cp;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> vel_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> vel_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> vel_2;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> temperature;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        grad_temperature;

    Dependencies(const panzer::IntegrationRule& ir,
                 const double u,
                 const double v,
                 const double w,
                 const bool build_temp,
                 const ContinuityModel continuity_model,
                 const bool unscaled_density)
        : _u(u)
        , _v(v)
        , _w(w)
        , _build_temp(build_temp)
        , _continuity_model(continuity_model)
        , _unscaled_density(unscaled_density)
        , lagrange_pressure("lagrange_pressure", ir.dl_scalar)
        , rho("density", ir.dl_scalar)
        , cp("specific_heat_capacity", ir.dl_scalar)
        , vel_0("velocity_0", ir.dl_scalar)
        , vel_1("velocity_1", ir.dl_scalar)
        , vel_2("velocity_2", ir.dl_scalar)
        , temperature("temperature", ir.dl_scalar)
        , grad_temperature("GRAD_temperature", ir.dl_vector)

    {
        this->addEvaluatedField(lagrange_pressure);
        this->addEvaluatedField(rho);
        this->addEvaluatedField(vel_0);
        this->addEvaluatedField(vel_1);
        this->addEvaluatedField(vel_2);
        if (_build_temp)
        {
            this->addEvaluatedField(cp);
            this->addEvaluatedField(temperature);
        }

        if (_build_temp && _continuity_model == ContinuityModel::EDACTempNC)
        {
            this->addEvaluatedField(grad_temperature);
        }

        this->setName("Incompressible Convective Flux Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        lagrange_pressure.deep_copy(0.75);
        rho.deep_copy(_unscaled_density ? 3.0 : 1.0);
        vel_0.deep_copy(_u);
        vel_1.deep_copy(_v);
        vel_2.deep_copy(_w);
        if (_build_temp)
        {
            cp.deep_copy(5.0);
            temperature.deep_copy(_u + _v);
        }

        Kokkos::parallel_for(
            "convective flux test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = grad_temperature.extent(1);
        const int num_space_dim = grad_temperature.extent(2);
        using Kokkos::pow;
        for (int qp = 0; qp < num_point; ++qp)
        {
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                const int sign = pow(-1, dim + 1);
                const int dimqp = (dim + 1) * sign;
                if (_build_temp
                    && _continuity_model == ContinuityModel::EDACTempNC)
                {
                    grad_temperature(c, qp, dim) = (_u + _v) * dimqp;
                }
            }
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval(const bool unscaled_density,
              const bool build_temp,
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
    const double w = num_space_dim > 2 ? 0.125 : nan_val;

    auto deps = Teuchos::rcp(new Dependencies<EvalType>(
        ir, u, v, w, build_temp, continuity_model, unscaled_density));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize class object to test
    Teuchos::ParameterList fluid_prop_list;
    fluid_prop_list.set("Density", unscaled_density ? 3.0 : 1.0);
    fluid_prop_list.set("Kinematic viscosity", 0.375);
    fluid_prop_list.set("Artificial compressibility", 2.0);
    fluid_prop_list.set("Build Temperature Equation", build_temp);
    if (build_temp)
    {
        fluid_prop_list.set("Thermal conductivity", 0.5);
        fluid_prop_list.set("Specific heat capacity", 5.0);
    }

    Teuchos::ParameterList user_params;
    if (continuity_model == ContinuityModel::EDAC)
    {
        user_params.set("Continuity Model", "EDAC");
    }
    else if (continuity_model == ContinuityModel::EDACTempNC)
    {
        user_params.set("Continuity Model", "EDACTempNC");
    }

    const FluidProperties::ConstantFluidProperties fluid_prop(fluid_prop_list);
    auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleConvectiveFlux<EvalType,
                                                       panzer::Traits,
                                                       num_space_dim>(
            ir, fluid_prop, user_params));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_continuity_flux);
    for (int dim = 0; dim < num_space_dim; ++dim)
        test_fixture.registerTestField<EvalType>(eval->_momentum_flux[dim]);

    test_fixture.evaluate<EvalType>();

    auto fc_cont
        = test_fixture.getTestFieldData<EvalType>(eval->_continuity_flux);
    auto fc_mom_0
        = test_fixture.getTestFieldData<EvalType>(eval->_momentum_flux[0]);

    const int num_point = ir.num_points;

    // Expected values
    const double exp_cont_flux_3d_ac[3] = {unscaled_density ? 0.75 : 0.25,
                                           unscaled_density ? 1.5 : 0.5,
                                           unscaled_density ? 0.375 : 0.125};

    const double exp_cont_flux_3d_edac[3]
        = {unscaled_density ? 0.84375 : 0.34375,
           unscaled_density ? 1.6875 : 0.6875,
           unscaled_density ? 0.421875 : 0.171875};

    const double exp_cont_flux_2d_ac[3]
        = {exp_cont_flux_3d_ac[0], exp_cont_flux_3d_ac[1], nan_val};

    const double exp_cont_flux_2d_edac[3]
        = {exp_cont_flux_3d_edac[0], exp_cont_flux_3d_edac[1], nan_val};

    const double* exp_cont_flux
        = continuity_model != ContinuityModel::AC
              ? (num_space_dim == 3 ? exp_cont_flux_3d_edac
                                    : exp_cont_flux_2d_edac)
              : (num_space_dim == 3 ? exp_cont_flux_3d_ac
                                    : exp_cont_flux_2d_ac);

    const double exp_mom_0_flux[3] = {
        unscaled_density ? 0.9375 : 0.8125,
        unscaled_density ? 0.375 : 0.125,
        num_space_dim == 3 ? unscaled_density ? 0.09375 : 0.03125 : nan_val};
    const double exp_mom_1_flux[3]
        = {unscaled_density ? 0.375 : 0.125,
           unscaled_density ? 1.5 : 1.,
           num_space_dim == 3 ? unscaled_density ? 0.1875 : 0.0625 : nan_val};
    const double exp_mom_2_flux[3] = {
        unscaled_density ? 0.09375 : 0.03125,
        unscaled_density ? 0.1875 : 0.0625,
        num_space_dim == 3 ? unscaled_density ? 0.796875 : 0.765625 : nan_val};
    const double exp_ener_flux[3] = {
        unscaled_density ? 2.8125 : 0.9375,
        unscaled_density ? 5.625 : 1.875,
        num_space_dim == 3 ? unscaled_density ? 1.40625 : 0.46875 : nan_val};

    const double exp_ener_source_2d
        = continuity_model == ContinuityModel::EDACTempNC
              ? (unscaled_density ? 8.4375 : 2.8125)
              : nan_val;
    const double exp_ener_source_3d
        = continuity_model == ContinuityModel::EDACTempNC
              ? (unscaled_density ? 4.21875 : 1.40625)
              : nan_val;

    const double exp_ener_source
        = continuity_model == ContinuityModel::EDACTempNC
              ? (num_space_dim == 3 ? exp_ener_source_3d : exp_ener_source_2d)
              : nan_val;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int dim = 0; dim < num_space_dim; dim++)
        {
            EXPECT_EQ(exp_cont_flux[dim], fieldValue(fc_cont, 0, qp, dim));
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
            if (build_temp)
            {
                if (continuity_model != ContinuityModel::EDACTempNC)
                {
                    const auto fc_energy
                        = test_fixture.getTestFieldData<EvalType>(
                            eval->_energy_flux);
                    EXPECT_EQ(exp_ener_flux[dim],
                              fieldValue(fc_energy, 0, qp, dim));
                }
            }
        }
        if (continuity_model == ContinuityModel::EDACTempNC)
        {
            const auto fc_energy_source
                = test_fixture.getTestFieldData<EvalType>(eval->_energy_source);
            EXPECT_EQ(exp_ener_source, fieldValue(fc_energy_source, 0, qp));
        }
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxScaledDensityIsothermalAC2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(false, false, ContinuityModel::AC);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxScaledDensityIsothermalAC2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(false, false, ContinuityModel::AC);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxScaledDensityAC2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(false, true, ContinuityModel::AC);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxScaledDensityAC2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(false, true, ContinuityModel::AC);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxScaledDensityIsothermalAC3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(false, false, ContinuityModel::AC);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxScaledDensityIsothermalAC3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(false, false, ContinuityModel::AC);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxUnscaledDensityIsothermalAC3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(true, false, ContinuityModel::AC);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxUnscaledDensityIsothermalAC3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(true, false, ContinuityModel::AC);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxUnscaledDensityAC3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(true, true, ContinuityModel::AC);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxUnscaledDensityAC3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(true, true, ContinuityModel::AC);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxScaledDensityIsothermalEDAC2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(false, false, ContinuityModel::EDAC);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxScaledDensityIsothermalEDAC2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(false, false, ContinuityModel::EDAC);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxScaledDensityEDAC2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(false, true, ContinuityModel::EDAC);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxScaledDensityEDAC2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(false, true, ContinuityModel::EDAC);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxScaledDensityIsothermalEDAC3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(false, false, ContinuityModel::EDAC);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxScaledDensityIsothermalEDAC3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(false, false, ContinuityModel::EDAC);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxUnscaledDensityIsothermalEDAC3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(true, false, ContinuityModel::EDAC);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxUnscaledDensityIsothermalEDAC3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(true, false, ContinuityModel::EDAC);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxUnscaledDensityEDAC3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(true, true, ContinuityModel::EDAC);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxUnscaledDensityEDAC3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(true, true, ContinuityModel::EDAC);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxScaledDensityEDACTempNC2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(
        false, true, ContinuityModel::EDACTempNC);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxScaledDensityEDACTempNC2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(
        false, true, ContinuityModel::EDACTempNC);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxUnscaledDensityEDACTempNC3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(
        true, true, ContinuityModel::EDACTempNC);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxUnscaledDensityEDACTempNC3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(
        true, true, ContinuityModel::EDACTempNC);
}

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
    test_fixture.type_name = "IncompressibleConvectiveFlux";
    test_fixture.eval_name = "Incompressible Convective Flux "
                             + std::to_string(num_space_dim) + "D";
    test_fixture.template buildAndTest<
        ClosureModel::IncompressibleConvectiveFlux<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(IncompressibleConvectiveFlux_Factory2D, Residual)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(IncompressibleConvectiveFlux_Factory2D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(IncompressibleConvectiveFlux_Factory3D, Residual)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(IncompressibleConvectiveFlux_Factory3D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
