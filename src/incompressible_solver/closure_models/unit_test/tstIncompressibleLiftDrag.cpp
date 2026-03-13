#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleLiftDrag.hpp"

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

    double _u;
    double _v;
    double _w;
    bool _use_compressible_formula;
    bool _build_turbulence_model;
    bool _unscaled_density;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_2;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> normals;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> lagrange_pressure;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> rho;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> nu;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> nu_t;

    Dependencies(const panzer::IntegrationRule& ir,
                 const double u,
                 const double v,
                 const double w,
                 const bool use_compressible_formula,
                 const bool build_turbulence_model,
                 const bool unscaled_density)
        : _u(u)
        , _v(v)
        , _w(w)
        , _use_compressible_formula(use_compressible_formula)
        , _build_turbulence_model(build_turbulence_model)
        , _unscaled_density(unscaled_density)
        , grad_vel_0("GRAD_velocity_0", ir.dl_vector)
        , grad_vel_1("GRAD_velocity_1", ir.dl_vector)
        , grad_vel_2("GRAD_velocity_2", ir.dl_vector)
        , normals("Side Normal", ir.dl_vector)
        , lagrange_pressure("lagrange_pressure", ir.dl_scalar)
        , rho("density", ir.dl_scalar)
        , nu("kinematic_viscosity", ir.dl_scalar)
        , nu_t("turbulent_eddy_viscosity", ir.dl_scalar)
    {
        this->addEvaluatedField(grad_vel_0);
        this->addEvaluatedField(grad_vel_1);
        this->addEvaluatedField(grad_vel_2);

        this->addEvaluatedField(normals);

        this->addEvaluatedField(lagrange_pressure);
        this->addEvaluatedField(rho);
        this->addEvaluatedField(nu);

        if (_build_turbulence_model)
        {
            this->addEvaluatedField(nu_t);
        }

        this->setName("Incompressible Lift Drag Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "lift drag test dependencies",
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

                normals(c, qp, dim) = (_u + _v) * dimqp;
            }

            lagrange_pressure(c, qp) = (_u + _v);
            rho(c, qp) = _unscaled_density ? 3.0 : 1.0;
            nu(c, qp) = 0.375;

            if (_build_turbulence_model)
            {
                nu_t(c, qp) = 4.0;
            }
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval(const bool unscaled_density,
              const bool use_compressible_formula,
              const bool build_turbulence_model)
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;
    const double nan_val = std::numeric_limits<double>::quiet_NaN();

    // Initialize velocity components and dependents
    const double u = 0.25;
    const double v = 0.5;
    const double w = num_space_dim == 3
                         ? 0.125
                         : std::numeric_limits<double>::quiet_NaN();

    auto deps
        = Teuchos::rcp(new Dependencies<EvalType>(ir,
                                                  u,
                                                  v,
                                                  w,
                                                  use_compressible_formula,
                                                  build_turbulence_model,
                                                  unscaled_density));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize class object to test
    Teuchos::ParameterList closure_params;
    closure_params.set("Compressible Formula", use_compressible_formula);
    auto eval = Teuchos::rcp(
        new ClosureModel::
            IncompressibleLiftDrag<EvalType, panzer::Traits, num_space_dim>(
                ir, closure_params, build_turbulence_model));
    test_fixture.registerEvaluator<EvalType>(eval);

    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        test_fixture.registerTestField<EvalType>(eval->_total_force[dim]);
        test_fixture.registerTestField<EvalType>(eval->_viscous_force[dim]);
        test_fixture.registerTestField<EvalType>(eval->_pressure_force[dim]);
    }

    test_fixture.evaluate<EvalType>();

    const auto calc_total_force_0
        = test_fixture.getTestFieldData<EvalType>(eval->_total_force[0]);
    const auto calc_viscous_force_0
        = test_fixture.getTestFieldData<EvalType>(eval->_viscous_force[0]);
    const auto calc_pressure_force_0
        = test_fixture.getTestFieldData<EvalType>(eval->_pressure_force[0]);
    const auto calc_total_force_1
        = test_fixture.getTestFieldData<EvalType>(eval->_total_force[1]);
    const auto calc_viscous_force_1
        = test_fixture.getTestFieldData<EvalType>(eval->_viscous_force[1]);
    const auto calc_pressure_force_1
        = test_fixture.getTestFieldData<EvalType>(eval->_pressure_force[1]);

    const int num_point = ir.num_points;

    // Expected values
    const double exp_pressure_force_3d[3] = {-0.5625, 1.125, -1.6875};
    const double exp_pressure_force_2d[3]
        = {exp_pressure_force_3d[0], exp_pressure_force_3d[1], nan_val};
    const double* exp_pressure_force
        = num_space_dim == 3 ? exp_pressure_force_3d : exp_pressure_force_2d;

    const double exp_viscous_force_3d[3] = {
        build_turbulence_model
            ? (unscaled_density ? -30.76171875 : -10.25390625)
            : -0.87890625,
        build_turbulence_model ? (unscaled_density ? -76.2890625 : -25.4296875)
                               : -2.1796875,
        build_turbulence_model ? (unscaled_density ? -6.15234375 : -2.05078125)
                               : -0.17578125};
    const double exp_viscous_force_2d[3]
        = {build_turbulence_model ? (unscaled_density ? -4.921875 : -1.640625)
                                  : -0.140625,
           build_turbulence_model ? (unscaled_density ? -39.375 : -13.125)
                                  : -1.125,
           nan_val};
    const double exp_compressible_viscous_force_3d[3] = {
        build_turbulence_model
            ? (unscaled_density ? -33.22265625 : -11.07421875)
            : -0.94921875,
        build_turbulence_model ? (unscaled_density ? -71.3671875 : -23.7890625)
                               : -2.0390625,
        build_turbulence_model ? (unscaled_density ? -13.53515625 : -4.51171875)
                               : -0.38671875};
    const double exp_compressible_viscous_force_2d[3] = {
        build_turbulence_model ? (unscaled_density ? -12.3046875 : -4.1015625)
                               : -0.3515625,
        build_turbulence_model ? (unscaled_density ? -24.609375 : -8.203125)
                               : -0.703125,
        nan_val};
    const double* exp_viscous_force
        = use_compressible_formula
              ? (num_space_dim == 3 ? exp_compressible_viscous_force_3d
                                    : exp_compressible_viscous_force_2d)
              : (num_space_dim == 3 ? exp_viscous_force_3d
                                    : exp_viscous_force_2d);

    const double exp_total_force_3d[3] = {
        build_turbulence_model
            ? (unscaled_density ? -31.32421875 : -10.81640625)
            : -1.44140625,
        build_turbulence_model ? (unscaled_density ? -75.1640625 : -24.3046875)
                               : -1.0546875,
        build_turbulence_model ? (unscaled_density ? -7.83984375 : -3.73828125)
                               : -1.86328125};
    const double exp_total_force_2d[3]
        = {build_turbulence_model ? (unscaled_density ? -5.484375 : -2.203125)
                                  : -0.703125,
           build_turbulence_model ? (unscaled_density ? -38.25 : -12.0) : 0.0,
           nan_val};
    const double exp_compressible_total_force_3d[3] = {
        build_turbulence_model
            ? (unscaled_density ? -33.78515625 : -11.63671875)
            : -1.51171875,
        build_turbulence_model ? (unscaled_density ? -70.2421875 : -22.6640625)
                               : -0.9140625,
        build_turbulence_model ? (unscaled_density ? -15.22265625 : -6.19921875)
                               : -2.07421875};
    const double exp_compressible_total_force_2d[3] = {
        build_turbulence_model ? (unscaled_density ? -12.8671875 : -4.6640625)
                               : -0.9140625,
        build_turbulence_model ? (unscaled_density ? -23.484375 : -7.078125)
                               : 0.421875,
        nan_val};
    const double* exp_total_force
        = use_compressible_formula
              ? (num_space_dim == 3 ? exp_compressible_total_force_3d
                                    : exp_compressible_total_force_2d)
              : (num_space_dim == 3 ? exp_total_force_3d : exp_total_force_2d);

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int dim = 0; dim < num_space_dim; dim++)
        {
            const auto calc_total_force
                = test_fixture.getTestFieldData<EvalType>(
                    eval->_total_force[dim]);
            const auto calc_viscous_force
                = test_fixture.getTestFieldData<EvalType>(
                    eval->_viscous_force[dim]);
            const auto calc_pressure_force
                = test_fixture.getTestFieldData<EvalType>(
                    eval->_pressure_force[dim]);

            EXPECT_EQ(exp_total_force[dim],
                      fieldValue(calc_total_force, 0, qp));
            EXPECT_EQ(exp_viscous_force[dim],
                      fieldValue(calc_viscous_force, 0, qp));
            EXPECT_EQ(exp_pressure_force[dim],
                      fieldValue(calc_pressure_force, 0, qp));
        }
    }
}

//-----------------------------------------------------------------//
struct IncompressibleLiftDragTestParams
{
    std::string test_name;
    bool unscaled_density;
    bool use_compressible_formula;
    bool build_turbulence_model;
    int num_space_dim;
};

class IncompressibleLiftDragTest
    : public testing::TestWithParam<IncompressibleLiftDragTestParams>
{
  public:
    struct PrintToStringParamName
    {
        template<class T>
        std::string operator()(const testing::TestParamInfo<T>& info) const
        {
            auto testParam
                = static_cast<IncompressibleLiftDragTestParams>(info.param);
            return testParam.test_name;
        }
    };
};

//-----------------------------------------------------------------//
TEST_P(IncompressibleLiftDragTest, cartesian)
{
    const auto params = GetParam();
    if (std::string::npos != params.test_name.find("residual"))
    {
        if (params.num_space_dim == 2)
        {
            testEval<panzer::Traits::Residual, 2>(
                params.unscaled_density,
                params.use_compressible_formula,
                params.build_turbulence_model);
        }
        else
        {
            testEval<panzer::Traits::Residual, 3>(
                params.unscaled_density,
                params.use_compressible_formula,
                params.build_turbulence_model);
        }
    }
    else if (std::string::npos != params.test_name.find("jacobian"))
    {
        if (params.num_space_dim == 2)
        {
            testEval<panzer::Traits::Jacobian, 2>(
                params.unscaled_density,
                params.use_compressible_formula,
                params.build_turbulence_model);
        }
        else
        {
            testEval<panzer::Traits::Jacobian, 3>(
                params.unscaled_density,
                params.use_compressible_formula,
                params.build_turbulence_model);
        }
    }
}

//-----------------------------------------------------------------//
INSTANTIATE_TEST_SUITE_P(
    Test,
    IncompressibleLiftDragTest,
    testing::Values(
        IncompressibleLiftDragTestParams{
            "ScaledDensity2D_residual", false, false, false, 2},
        IncompressibleLiftDragTestParams{
            "ScaledDensity2D_jacobian", false, false, false, 2},
        IncompressibleLiftDragTestParams{
            "ScaledDensity3D_residual", false, false, false, 3},
        IncompressibleLiftDragTestParams{
            "ScaledDensity3D_jacobian", false, false, false, 3},
        IncompressibleLiftDragTestParams{
            "ScaledDensityCompressible2D_residual", false, true, false, 2},
        IncompressibleLiftDragTestParams{
            "ScaledDensityCompressible2D_jacobian", false, true, false, 2},
        IncompressibleLiftDragTestParams{
            "ScaledDensityCompressible3D_residual", false, true, false, 3},
        IncompressibleLiftDragTestParams{
            "ScaledDensityCompressible3D_jacobian", false, true, false, 3},
        IncompressibleLiftDragTestParams{
            "ScaledDensityTurbulent2D_residual", false, false, true, 2},
        IncompressibleLiftDragTestParams{
            "ScaledDensityTurbulent2D_jacobian", false, false, true, 2},
        IncompressibleLiftDragTestParams{
            "UnScaledDensityTurbulent2D_residual", true, false, true, 2},
        IncompressibleLiftDragTestParams{
            "UnScaledDensityTurbulent2D_jacobian", true, false, true, 2},
        IncompressibleLiftDragTestParams{
            "ScaledDensityTurbulent3D_residual", false, false, true, 3},
        IncompressibleLiftDragTestParams{
            "ScaledDensityTurbulent3D_jacobian", false, false, true, 3},
        IncompressibleLiftDragTestParams{
            "UnScaledDensityTurbulent3D_residual", true, false, true, 3},
        IncompressibleLiftDragTestParams{
            "UnScaledDensityTurbulent3D_jacobian", true, false, true, 3},
        IncompressibleLiftDragTestParams{
            "ScaledDensityCompressibleTurbulent2D_residual", false, true, true, 2},
        IncompressibleLiftDragTestParams{
            "ScaledDensityCompressibleTurbulent2D_jacobian", false, true, true, 2},
        IncompressibleLiftDragTestParams{"UnScaledDensityCompressibleTurbulent"
                                         "2D_residual",
                                         true,
                                         true,
                                         true,
                                         2},
        IncompressibleLiftDragTestParams{"UnScaledDensityCompressibleTurbulent"
                                         "2D_jacobian",
                                         true,
                                         true,
                                         true,
                                         2},
        IncompressibleLiftDragTestParams{
            "ScaledDensityCompressibleTurbulent3D_residual", false, true, true, 3},
        IncompressibleLiftDragTestParams{
            "ScaledDensityCompressibleTurbulent3D_jacobian", false, true, true, 3},
        IncompressibleLiftDragTestParams{"UnScaledDensityCompressibleTurbulent"
                                         "3D_residual",
                                         true,
                                         true,
                                         true,
                                         3},
        IncompressibleLiftDragTestParams{"UnScaledDensityCompressibleTurbulent"
                                         "3D_jacobian",
                                         true,
                                         true,
                                         true,
                                         3}),

    IncompressibleLiftDragTest::PrintToStringParamName());

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.model_params.set("Compressible Formula", true);
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.375)
        .set("Artificial compressibility", 2.0);
    test_fixture.type_name = "IncompressibleLiftDrag";
    test_fixture.eval_name = "Incompressible Lift/Drag "
                             + std::to_string(num_space_dim) + "D";
    test_fixture.template buildAndTest<
        ClosureModel::IncompressibleLiftDrag<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(IncompressibleLiftDrag_Factory2D, Residual)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(IncompressibleLiftDrag_Factory2D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(IncompressibleLiftDrag_Factory3D, Residual)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(IncompressibleLiftDrag_Factory3D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
