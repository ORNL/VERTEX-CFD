#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleTimeDerivative.hpp"

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
    const double nan_val = std::numeric_limits<double>::quiet_NaN();

    double _dphi_dt;
    double _du_dt;
    double _dv_dt;
    double _dw_dt;
    bool _build_temp_equ;
    bool _unscaled_density;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dvdt_lagrange_pressure;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _rho;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _cp;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dvdt_velocity_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dvdt_velocity_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dvdt_velocity_2;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dvdt_temperature;

    Dependencies(const panzer::IntegrationRule& ir,
                 const double dphi_dt,
                 const double du_dt,
                 const double dv_dt,
                 const double dw_dt,
                 const bool build_temp_equ,
                 const bool unscaled_density)
        : _dphi_dt(dphi_dt)
        , _du_dt(du_dt)
        , _dv_dt(dv_dt)
        , _dw_dt(dw_dt)
        , _build_temp_equ(build_temp_equ)
        , _unscaled_density(unscaled_density)
        , _dvdt_lagrange_pressure("DXDT_lagrange_pressure", ir.dl_scalar)
        , _rho("density", ir.dl_scalar)
        , _cp("specific_heat_capacity", ir.dl_scalar)
        , _dvdt_velocity_0("DXDT_velocity_0", ir.dl_scalar)
        , _dvdt_velocity_1("DXDT_velocity_1", ir.dl_scalar)
        , _dvdt_velocity_2("DXDT_velocity_2", ir.dl_scalar)
        , _dvdt_temperature("DXDT_temperature", ir.dl_scalar)
    {
        this->addEvaluatedField(_dvdt_lagrange_pressure);
        this->addEvaluatedField(_rho);
        this->addEvaluatedField(_cp);
        this->addEvaluatedField(_dvdt_velocity_0);
        this->addEvaluatedField(_dvdt_velocity_1);
        this->addEvaluatedField(_dvdt_velocity_2);
        if (_build_temp_equ)
            this->addEvaluatedField(_dvdt_temperature);
        this->setName("Incompressible Time Derivative Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData) override
    {
        _dvdt_lagrange_pressure.deep_copy(_dphi_dt);
        _rho.deep_copy(_unscaled_density ? 2.0 : 1.0);
        _cp.deep_copy(_build_temp_equ ? 5.0 : nan_val);
        _dvdt_velocity_0.deep_copy(_du_dt);
        _dvdt_velocity_1.deep_copy(_dv_dt);
        _dvdt_velocity_2.deep_copy(_dw_dt);
        if (_build_temp_equ)
            _dvdt_temperature.deep_copy(_du_dt + _dv_dt);
    }
};

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval(const bool unscaled_density, const bool build_temp_equ)
{
    // Setup test fixture.
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Eval dependencies.
    const double dphi_dt = 0.125;
    const double du_dt = 1.25;
    const double dv_dt = 1.5;
    const double dw_dt
        = num_space_dim == 3 ? 1.75 : std::numeric_limits<double>::quiet_NaN();
    auto dep_eval = Teuchos::rcp(new Dependencies<EvalType>(*test_fixture.ir,
                                                            dphi_dt,
                                                            du_dt,
                                                            dv_dt,
                                                            dw_dt,
                                                            build_temp_equ,
                                                            unscaled_density));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Fluid properties
    const double beta = 0.1;
    const double rho = unscaled_density ? 2.0 : 1.0;
    const double Cp
        = build_temp_equ ? 5.0 : std::numeric_limits<double>::quiet_NaN();

    Teuchos::ParameterList fluid_params_list;
    fluid_params_list.set("Artificial compressibility", beta);
    fluid_params_list.set("Build Temperature Equation", build_temp_equ);

    // Create test evaluator.
    auto dqdt_eval = Teuchos::rcp(
        new ClosureModel::IncompressibleTimeDerivative<EvalType,
                                                       panzer::Traits,
                                                       num_space_dim>(
            *test_fixture.ir, fluid_params_list));
    test_fixture.registerEvaluator<EvalType>(dqdt_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(dqdt_eval->_dqdt_continuity);
    for (int dim = 0; dim < num_space_dim; dim++)
    {
        test_fixture.registerTestField<EvalType>(
            dqdt_eval->_dqdt_momentum[dim]);
    }

    // Evaluate test fields.
    test_fixture.evaluate<EvalType>();

    // Expected momentum values
    const double exp_mom[3] = {rho * du_dt, rho * dv_dt, rho * dw_dt};

    // Expected energy values
    const double exp_ener = rho * Cp * (du_dt + dv_dt);

    // Check the test fields.
    auto continuity_result
        = test_fixture.getTestFieldData<EvalType>(dqdt_eval->_dqdt_continuity);

    EXPECT_DOUBLE_EQ(0.125 / beta, fieldValue(continuity_result, 0, 0));
    for (int dim = 0; dim < num_space_dim; dim++)
    {
        auto momentum_dim_result = test_fixture.getTestFieldData<EvalType>(
            dqdt_eval->_dqdt_momentum[dim]);
        EXPECT_DOUBLE_EQ(exp_mom[dim], fieldValue(momentum_dim_result, 0, 0));
    }
    if (build_temp_equ)
    {
        const auto energy_result
            = test_fixture.getTestFieldData<EvalType>(dqdt_eval->_dqdt_energy);
        EXPECT_DOUBLE_EQ(exp_ener, fieldValue(energy_result, 0, 0));
    }
} // namespace Test

//---------------------------------------------------------------------------//
TEST(IncompressibleTimeDerivativeScaledDensityIsothermal2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>(false, false);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleTimeDerivativeScaledDensityIsothermal2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>(false, false);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleTimeDerivativeScaledDensity2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>(false, true);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleTimeDerivativeScaledDensity2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>(false, true);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleTimeDerivativeScaledDensityIsothermal3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>(false, false);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleTimeDerivativeScaledDensityIsothermal3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>(false, false);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleTimeDerivativeUnscaledDensityIsothermal3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>(true, false);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleTimeDerivativeUnscaledDensityIsothermal3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>(true, false);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleTimeDerivativeUnscaledDensity3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>(true, true);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleTimeDerivativeUnscaledDensity3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>(true, true);
}

//---------------------------------------------------------------------------//

template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.type_name = "IncompressibleTimeDerivative";
    test_fixture.eval_name = "Incompressible Time Derivative "
                             + std::to_string(num_space_dim) + "D";
    test_fixture.template buildAndTest<
        ClosureModel::IncompressibleTimeDerivative<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

//---------------------------------------------------------------------------//
TEST(IncompressibleTimeDerivative_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

//---------------------------------------------------------------------------//
TEST(IncompressibleTimeDerivative_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

//---------------------------------------------------------------------------//
TEST(IncompressibleTimeDerivative_Factory3D, residual_test)
{
    testFactory<panzer::Traits::Residual, 3>();
}

//---------------------------------------------------------------------------//
TEST(IncompressibleTimeDerivative_Factory3D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // end namespace VertexCFD
