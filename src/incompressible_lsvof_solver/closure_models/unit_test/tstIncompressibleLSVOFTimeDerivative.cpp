#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFTimeDerivative.hpp"

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

    double _rho;
    double _u;
    double _v;
    double _w;

    double _dp_dt;
    double _du_dt;
    double _dv_dt;
    double _dw_dt;
    double _drho_dt;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _vel_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _vel_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _vel_2;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _density;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dvdt_pressure;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dvdt_density;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dvdt_vel_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dvdt_vel_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dvdt_vel_2;

    Dependencies(const panzer::IntegrationRule& ir,
                 const double rho,
                 const double u,
                 const double v,
                 const double w,
                 const double dp_dt,
                 const double du_dt,
                 const double dv_dt,
                 const double dw_dt,
                 const double drho_dt)
        : _rho(rho)
        , _u(u)
        , _v(v)
        , _w(w)
        , _dp_dt(dp_dt)
        , _du_dt(du_dt)
        , _dv_dt(dv_dt)
        , _dw_dt(dw_dt)
        , _drho_dt(drho_dt)
        , _vel_0("velocity_0", ir.dl_scalar)
        , _vel_1("velocity_1", ir.dl_scalar)
        , _vel_2("velocity_2", ir.dl_scalar)
        , _density("density", ir.dl_scalar)
        , _dvdt_pressure("DXDT_lagrange_pressure", ir.dl_scalar)
        , _dvdt_density("DXDT_density", ir.dl_scalar)
        , _dvdt_vel_0("DXDT_velocity_0", ir.dl_scalar)
        , _dvdt_vel_1("DXDT_velocity_1", ir.dl_scalar)
        , _dvdt_vel_2("DXDT_velocity_2", ir.dl_scalar)
    {
        this->addEvaluatedField(_vel_0);
        this->addEvaluatedField(_vel_1);
        this->addEvaluatedField(_vel_2);
        this->addEvaluatedField(_density);
        this->addEvaluatedField(_dvdt_pressure);
        this->addEvaluatedField(_dvdt_density);
        this->addEvaluatedField(_dvdt_vel_0);
        this->addEvaluatedField(_dvdt_vel_1);
        this->addEvaluatedField(_dvdt_vel_2);

        this->setName(
            "Incompressible LSVOF Time Derivative Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData) override
    {
        _vel_0.deep_copy(_u);
        _vel_1.deep_copy(_v);
        _vel_2.deep_copy(_w);
        _density.deep_copy(_rho);
        _dvdt_pressure.deep_copy(_dp_dt);
        _dvdt_density.deep_copy(_drho_dt);
        _dvdt_vel_0.deep_copy(_du_dt);
        _dvdt_vel_1.deep_copy(_dv_dt);
        _dvdt_vel_2.deep_copy(_dw_dt);
    }
};

template<class EvalType, int NumSpaceDim>
void testEval()
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Initialize velocity components and dependents
    const double nanval = std::numeric_limits<double>::quiet_NaN();
    const double u = 0.25;
    const double v = 0.5;
    const double w = num_space_dim > 2 ? 0.125 : nanval;
    const double rho = 1.375;
    const double dp_dt = 1.2;
    const double drho_dt = 3.5;
    const double du_dt = 0.7;
    const double dv_dt = 0.1;
    const double dw_dt = 2.4;

    auto deps = Teuchos::rcp(new Dependencies<EvalType>(
        ir, rho, u, v, w, dp_dt, du_dt, dv_dt, dw_dt, drho_dt));
    test_fixture.registerEvaluator<EvalType>(deps);

    // LSVOF properties
    const double beta = 10.0;
    Teuchos::ParameterList lsvof_params;
    lsvof_params.set("Mixture Artificial Compressibility", beta);

    // Initialize class object to test
    const auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleLSVOFTimeDerivative<EvalType,
                                                            panzer::Traits,
                                                            num_space_dim>(
            ir, lsvof_params));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_dqdt_continuity);
    for (int dim = 0; dim < num_space_dim; ++dim)
        test_fixture.registerTestField<EvalType>(eval->_dqdt_momentum[dim]);

    test_fixture.evaluate<EvalType>();

    const auto fc_cont
        = test_fixture.getTestFieldData<EvalType>(eval->_dqdt_continuity);
    const auto fc_mom_0
        = test_fixture.getTestFieldData<EvalType>(eval->_dqdt_momentum[0]);
    const auto fc_mom_1
        = test_fixture.getTestFieldData<EvalType>(eval->_dqdt_momentum[1]);

    const int num_point = ir.num_points;

    // Expected values
    const double exp_dqdt_continuity = dp_dt / beta;
    const double exp_dqdt_momentum[3] = {rho * du_dt + drho_dt * u,
                                         rho * dv_dt + drho_dt * v,
                                         rho * dw_dt + drho_dt * w};

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_EQ(exp_dqdt_continuity, fieldValue(fc_cont, 0, qp));
        EXPECT_EQ(exp_dqdt_momentum[0], fieldValue(fc_mom_0, 0, qp));
        EXPECT_EQ(exp_dqdt_momentum[1], fieldValue(fc_mom_1, 0, qp));

        if (num_space_dim > 2) // 3D mesh
        {
            const auto fc_mom_2 = test_fixture.getTestFieldData<EvalType>(
                eval->_dqdt_momentum[2]);
            EXPECT_EQ(exp_dqdt_momentum[2], fieldValue(fc_mom_2, 0, qp));
        }
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleLSVOFTimeDerivative2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleLSVOFTimeDerivative2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleLSVOFTimeDerivative3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleLSVOFTimeDerivative3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
