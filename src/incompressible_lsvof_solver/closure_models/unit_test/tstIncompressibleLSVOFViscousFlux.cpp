#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFViscousFlux.hpp"

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
    ContinuityModel _continuity_model;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> velocity_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> velocity_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> velocity_2;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> rho;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> mu;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_2;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_press;

    Dependencies(const panzer::IntegrationRule& ir,
                 const double u,
                 const double v,
                 const double w,
                 const ContinuityModel continuity_model)
        : _u(u)
        , _v(v)
        , _w(w)
        , _continuity_model(continuity_model)
        , velocity_0("velocity_0", ir.dl_scalar)
        , velocity_1("velocity_1", ir.dl_scalar)
        , velocity_2("velocity_2", ir.dl_scalar)
        , rho("density", ir.dl_scalar)
        , mu("dynamic_viscosity", ir.dl_scalar)
        , grad_vel_0("GRAD_velocity_0", ir.dl_vector)
        , grad_vel_1("GRAD_velocity_1", ir.dl_vector)
        , grad_vel_2("GRAD_velocity_2", ir.dl_vector)
        , grad_press("GRAD_lagrange_pressure", ir.dl_vector)
    {
        this->addEvaluatedField(velocity_0);
        this->addEvaluatedField(velocity_1);
        this->addEvaluatedField(velocity_2);

        this->addEvaluatedField(rho);
        this->addEvaluatedField(mu);
        this->addEvaluatedField(grad_vel_0);
        this->addEvaluatedField(grad_vel_1);
        this->addEvaluatedField(grad_vel_2);

        this->addEvaluatedField(grad_press);

        this->setName(
            "Incompressible LSVOF Viscous Flux Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "LSVOF viscous flux test dependencies",
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

                grad_press(c, qp, dim) = _continuity_model
                                                 == ContinuityModel::EDAC
                                             ? (_u + _v) * dimqp
                                             : _nanval;
            }

            velocity_0(c, qp) = _u;
            velocity_1(c, qp) = _v;
            velocity_2(c, qp) = _w;
            rho(c, qp) = 13.75;
            mu(c, qp) = 10.0;
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval(const ContinuityModel continuity_model)
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

    const auto deps = Teuchos::rcp(
        new Dependencies<EvalType>(ir, u, v, w, continuity_model));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize class object to test
    const double betam = 2.0;

    Teuchos::ParameterList closure_params;
    closure_params.set("Mixture Artificial Compressibility", betam);

    Teuchos::ParameterList user_params;
    if (continuity_model == ContinuityModel::EDAC)
        user_params.set("Continuity Model", "EDAC");

    const auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleLSVOFViscousFlux<EvalType,
                                                         panzer::Traits,
                                                         num_space_dim>(
            ir, closure_params, user_params));
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

    // Expected values
    const double exp_cont_flux_3d_ac[3] = {0.0, 0.0, 0.0};
    const double exp_cont_flux_3d_edac[3] = {-3.75, 7.5, -11.25};

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

    const double exp_mom_0_flux_3d[3] = {-5.0, 0.0, -8.75};
    const double exp_mom_0_flux_2d[3]
        = {exp_mom_0_flux_3d[0], exp_mom_0_flux_3d[1], nan_val};
    const double* exp_mom_0_flux = num_space_dim == 3 ? exp_mom_0_flux_3d
                                                      : exp_mom_0_flux_2d;

    const double exp_mom_1_flux_3d[3] = {0.0, 20.0, -12.5};
    const double exp_mom_1_flux_2d[3]
        = {exp_mom_1_flux_3d[0], exp_mom_1_flux_3d[1], nan_val};
    const double* exp_mom_1_flux = num_space_dim == 3 ? exp_mom_1_flux_3d
                                                      : exp_mom_1_flux_2d;

    const double exp_mom_2_flux_3d[3] = {-8.75, -12.5, -7.5};
    const double exp_mom_2_flux_2d[3] = {nan_val, nan_val, nan_val};
    const double* exp_mom_2_flux = num_space_dim == 3 ? exp_mom_2_flux_3d
                                                      : exp_mom_2_flux_2d;

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
        }
    }
}

//---------------------------------------------------------------------------//
// Value parameterized test fixture
struct TestLSVOFViscous : public testing::TestWithParam<ContinuityModel>
{
    // Case generator for parameterized test
    struct ParamNameGenerator
    {
        std::string
        operator()(const testing::TestParamInfo<ContinuityModel>& info) const
        {
            const auto continuity_model = info.param;
            switch (continuity_model)
            {
                case (ContinuityModel::AC):
                    return "AC";
                case (ContinuityModel::EDAC):
                    return "EDAC";
                default:
                    return "INVALID_NAME";
            }
        }
    };
};

//-----------------------------------------------------------------//
TEST_P(TestLSVOFViscous, Residual2D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 2>(continuity_model);
}

//-----------------------------------------------------------------//
TEST_P(TestLSVOFViscous, Jacobian2D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 2>(continuity_model);
}

//-----------------------------------------------------------------//
TEST_P(TestLSVOFViscous, Residual3D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 3>(continuity_model);
}

//-----------------------------------------------------------------//
TEST_P(TestLSVOFViscous, Jacobian3D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 3>(continuity_model);
}

//---------------------------------------------------------------------------//
// Generate test suite with continuity models
INSTANTIATE_TEST_SUITE_P(ContinuityModelType,
                         TestLSVOFViscous,
                         testing::Values(ContinuityModel::AC,
                                         ContinuityModel::EDAC),
                         TestLSVOFViscous::ParamNameGenerator{});

} // namespace Test
} // namespace VertexCFD
