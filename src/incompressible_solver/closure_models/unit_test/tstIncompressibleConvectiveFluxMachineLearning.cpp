#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleConvectiveFluxMachineLearning.hpp"

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

    const double _u;
    const double _v;
    const double _w;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> lagrange_pressure;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> vel_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> vel_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> vel_2;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> temperature;

    Dependencies(const panzer::IntegrationRule& ir,
                 const double u,
                 const double v,
                 const double w)
        : _u(u)
        , _v(v)
        , _w(w)
        , lagrange_pressure("lagrange_pressure", ir.dl_scalar)
        , vel_0("velocity_0", ir.dl_scalar)
        , vel_1("velocity_1", ir.dl_scalar)
        , vel_2("velocity_2", ir.dl_scalar)
        , temperature("temperature", ir.dl_scalar)

    {
        this->addEvaluatedField(lagrange_pressure);
        this->addEvaluatedField(vel_0);
        this->addEvaluatedField(vel_1);
        this->addEvaluatedField(vel_2);
        this->addEvaluatedField(temperature);

        this->setName("Incompressible Convective Flux Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData) override
    {
        lagrange_pressure.deep_copy(0.75);
        vel_0.deep_copy(_u);
        vel_1.deep_copy(_v);
        vel_2.deep_copy(_w);
        temperature.deep_copy(_u + _v);
    }
};

template<class EvalType, int NumSpaceDim>
void testEval(const bool unscaled_density)
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    auto& ir = *test_fixture.ir;

    // Initialize velocity components and dependents
    const double u = 0.25;
    const double v = 0.5;
    const double w
        = num_space_dim > 2 ? 0.125 : std::numeric_limits<double>::quiet_NaN();

    auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir, u, v, w));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize class object to test
    const double rho = unscaled_density ? 3.0 : 1.0;
    const double nu = 200.0;
    Teuchos::ParameterList fluid_params;
    fluid_params.set("Build Temperature Equation", true);
    fluid_params.set("Artificial compressibility", 2.0);
    Teuchos::ParameterList closure_params;
    closure_params.set("TensorFlow File",
                       "incompressible_2d_planar_poiseuille_tensorflow."
                       "tflite");
    closure_params.set("TensorFlow Metadata File",
                       "incompressible_2d_planar_poiseuille_tensorflow_"
                       "metadata."
                       "yml");

    const double H_min = -1.1;
    const double H_max = 1.1;
    const double Cp = 100.;
    closure_params.set("H min", H_min);
    closure_params.set("H max", H_max);
    const double h = .5 * (H_max - H_min);
    const double S_u = 24000;
    closure_params.set("Momentum source", S_u);
    fluid_params.set("Density", rho);
    fluid_params.set("Kinematic viscosity", nu);
    fluid_params.set("Specific heat capacity", Cp);

    const double U_max = 1.5 * S_u * h * h / 3.0 / nu;

    auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleConvectiveFluxMachineLearning<
            EvalType,
            panzer::Traits,
            num_space_dim>(ir, fluid_params, closure_params));
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
    const double exp_cont_flux[3] = {rho * u, rho * v, rho * w};
    const double exp_mom_0_flux[3]
        = {unscaled_density ? 0.9375 : 0.8125,
           unscaled_density ? 0.375 : 0.125,
           num_space_dim == 3 ? unscaled_density ? 0.09375 : 0.03125
                              : std::numeric_limits<double>::quiet_NaN()};
    const double exp_mom_1_flux[3]
        = {unscaled_density ? 0.375 : 0.125,
           unscaled_density ? 1.5 : 1.,
           num_space_dim == 3 ? unscaled_density ? 0.1875 : 0.0625
                              : std::numeric_limits<double>::quiet_NaN()};
    const double exp_mom_2_flux[3]
        = {unscaled_density ? 0.09375 : 0.03125,
           unscaled_density ? 0.1875 : 0.0625,
           num_space_dim == 3 ? unscaled_density ? 0.796875 : 0.765625
                              : std::numeric_limits<double>::quiet_NaN()};

    auto ip_coord_view
        = test_fixture.int_values->ip_coordinates.get_static_view();
    auto coordinates = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                           ip_coord_view);
    auto y_values = Kokkos::subview(coordinates, 0, Kokkos::ALL, 1);

    auto host_temperature = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, deps->temperature.get_static_view());

    const Kokkos::View<double**, Kokkos::HostSpace> energy_references(
        "energy_references", num_point, num_space_dim);
    energy_references(0, 0) = unscaled_density ? 8140.2740099662151
                                               : 2713.424669988739;
    energy_references(1, 0) = unscaled_density ? 8140.2740099662151
                                               : 2713.424669988739;
    energy_references(2, 0) = unscaled_density ? 15638.689376278577
                                               : 5212.8964587595256;
    energy_references(3, 0) = unscaled_density ? 15638.689376278577
                                               : 5212.8964587595256;
    if constexpr (num_space_dim == 3)
    {
        energy_references(4, 0) = unscaled_density ? 8140.2740099662151
                                                   : 2713.424669988739;
        energy_references(5, 0) = unscaled_density ? 8140.2740099662151
                                                   : 2713.424669988739;
        energy_references(6, 0) = unscaled_density ? 15638.689376278577
                                                   : 5212.8964587595256;
        energy_references(7, 0) = unscaled_density ? 15638.689376278577
                                                   : 5212.8964587595256;
    }

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
            const auto fc_energy
                = test_fixture.getTestFieldData<EvalType>(eval->_energy_flux);
            const auto exp_vel
                = dim == 0
                      ? 4 * (y_values[qp] - H_min) * (H_max - y_values[qp])
                            / ((H_max - H_min) * (H_max - H_min)) * U_max
                      : 0;
            EXPECT_NEAR(exp_vel * Cp * rho * host_temperature(0, qp),
                        fieldValue(fc_energy, 0, qp, dim),
                        0.06 * fieldValue(fc_energy, 0, qp, dim));
            EXPECT_DOUBLE_EQ(energy_references(qp, dim),
                             fieldValue(fc_energy, 0, qp, dim));
        }
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxMachineLearningUnscaledDensity2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(true);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxMachineLearningScaledDensity2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxMachineLearningUnscaledDensity3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(true);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConvectiveFluxMachineLearningScaledDensity3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(false);
}

} // namespace Test
} // namespace VertexCFD
