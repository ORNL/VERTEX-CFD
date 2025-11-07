#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleNoSlip.hpp"
#include "incompressible_solver/fluid_properties/VertexCFD_Closure_IncompressibleFluidProperties.hpp"
#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
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
// Temperature Profile cases
enum TempProfile
{
    isothermal,
    adiabatic,
    flux
};
//---------------------------------------------------------------------------//
// Test data dependencies.
template<class EvalType>
struct Dependencies : public PHX::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    double _phi, _u_0, _u_1, _u_2;
    bool _build_tmp_equ;
    ContinuityModel _continuity_model;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _lagrange_pressure;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _temperature;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _velocity_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _velocity_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _velocity_2;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> _normals;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_velocity_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_velocity_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_velocity_2;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_temperature;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_lagrange_pressure;

    Dependencies(const panzer::IntegrationRule& ir,
                 double phi,
                 double u_0,
                 double u_1,
                 double u_2,
                 const bool build_tmp_equ,
                 const ContinuityModel continuity_model)
        : _phi(phi)
        , _u_0(u_0)
        , _u_1(u_1)
        , _u_2(u_2)
        , _build_tmp_equ(build_tmp_equ)
        , _continuity_model(continuity_model)
        , _lagrange_pressure("lagrange_pressure", ir.dl_scalar)
        , _temperature("temperature", ir.dl_scalar)
        , _velocity_0("velocity_0", ir.dl_scalar)
        , _velocity_1("velocity_1", ir.dl_scalar)
        , _velocity_2("velocity_2", ir.dl_scalar)
        , _normals("Side Normal", ir.dl_vector)
        , _grad_velocity_0("GRAD_velocity_0", ir.dl_vector)
        , _grad_velocity_1("GRAD_velocity_1", ir.dl_vector)
        , _grad_velocity_2("GRAD_velocity_2", ir.dl_vector)
        , _grad_temperature("GRAD_temperature", ir.dl_vector)
        , _grad_lagrange_pressure("GRAD_lagrange_pressure", ir.dl_vector)
    {
        this->addEvaluatedField(_lagrange_pressure);
        if (_build_tmp_equ)
            this->addEvaluatedField(_temperature);
        this->addEvaluatedField(_velocity_0);
        this->addEvaluatedField(_velocity_1);
        this->addEvaluatedField(_velocity_2);

        this->addEvaluatedField(_normals);

        this->addEvaluatedField(_grad_velocity_0);
        this->addEvaluatedField(_grad_velocity_1);
        this->addEvaluatedField(_grad_velocity_2);
        if (_build_tmp_equ)
            this->addEvaluatedField(_grad_temperature);
        if (_continuity_model == ContinuityModel::EDAC)
            this->addEvaluatedField(_grad_lagrange_pressure);

        this->setName("Incompressible NoSlip Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        // Set scalar variables
        _lagrange_pressure.deep_copy(_phi);
        _velocity_0.deep_copy(_u_0);
        _velocity_1.deep_copy(_u_1);
        _velocity_2.deep_copy(_u_2);
        if (_build_tmp_equ)
            _temperature.deep_copy(_u_0 + _u_1);

        Kokkos::parallel_for(
            "incompressible no slip test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int c) const
    {
        const int num_point = _lagrange_pressure.extent(1);
        const int num_space_dim = _grad_velocity_0.extent(2);
        using Kokkos::pow;
        for (int qp = 0; qp < num_point; ++qp)
        {
            // Set gradient and normal vectors
            for (int d = 0; d < num_space_dim; ++d)
            {
                const int sign = pow(-1, d + 1);
                const int dimqp = (d + 1) * sign;

                _normals(c, qp, d) = (_u_0 + _u_1) * dimqp;

                _grad_velocity_0(c, qp, d) = _u_0 * dimqp;
                _grad_velocity_1(c, qp, d) = _u_1 * dimqp;
                _grad_velocity_2(c, qp, d) = _u_2 * dimqp;

                if (_build_tmp_equ)
                    _grad_temperature(c, qp, d) = (_u_0 + _u_1) * dimqp;
                if (_continuity_model == ContinuityModel::EDAC)
                    _grad_lagrange_pressure(c, qp, d) = (_u_0 + _u_1) * dimqp;
            }
        }
    }
};

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval(const bool set_lagrange_pressure,
              const bool build_temp_equ,
              const ContinuityModel continuity_model,
              const TempProfile temperature_profile)
{
    // Test fixture
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const double nan_val = std::numeric_limits<double>::quiet_NaN();

    bool is_edac = false;
    switch (continuity_model)
    {
        case (ContinuityModel::AC):
            is_edac = false;
            break;
        case (ContinuityModel::EDAC):
            is_edac = true;
            break;
    }

    // Initialize values and create dependencies
    const double phi = 1.5;
    const double u = 0.25;
    const double v = 0.5;
    const double w = num_space_dim == 3 ? 0.125 : nan_val;
    const double T_wall = u * v;
    const double wall_flux = 2.0;
    const auto dep_eval = Teuchos::rcp(new Dependencies<EvalType>(
        *test_fixture.ir, phi, u, v, w, build_temp_equ, continuity_model));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Equation of state
    Teuchos::ParameterList fluid_prop_list;
    fluid_prop_list.set("Kinematic viscosity", 0.375);
    fluid_prop_list.set("Artificial compressibility", 2.0);
    fluid_prop_list.set("Build Temperature Equation", build_temp_equ);
    if (build_temp_equ)
    {
        fluid_prop_list.set("Thermal conductivity", 0.5);
        fluid_prop_list.set("Specific heat capacity", 0.6);
    }
    const FluidProperties::ConstantFluidProperties fluid_prop(fluid_prop_list);

    const auto eval_prop = Teuchos::rcp(
        new FluidProperties::IncompressibleFluidProperties<EvalType,
                                                           panzer::Traits>(
            *test_fixture.ir, fluid_prop_list));

    test_fixture.registerEvaluator<EvalType>(eval_prop);

    // Boundary condition
    Teuchos::ParameterList bc_params;
    if (build_temp_equ)
    {
        if (temperature_profile == TempProfile::isothermal)
        {
            bc_params.set("Wall Temperature", T_wall);
        }
        else if (temperature_profile == TempProfile::adiabatic)
        {
            bc_params.set("Temperature Profile", "Adiabatic");
        }
        else if (temperature_profile == TempProfile::flux)
        {
            bc_params.set("Temperature Profile", "Flux");
            bc_params.set("Wall Flux", wall_flux);
        }
    }

    if (set_lagrange_pressure)
        bc_params.set("Lagrange Pressure", 0.0);

    // Create no slip evaluator.
    auto no_slip_eval = Teuchos::rcp(
        new BoundaryCondition::
            IncompressibleNoSlip<EvalType, panzer::Traits, num_space_dim>(
                *test_fixture.ir, fluid_prop, bc_params, is_edac));
    test_fixture.registerEvaluator<EvalType>(no_slip_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(
        no_slip_eval->_boundary_lagrange_pressure);
    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        test_fixture.registerTestField<EvalType>(
            no_slip_eval->_boundary_velocity[dim]);
        test_fixture.registerTestField<EvalType>(
            no_slip_eval->_boundary_grad_velocity[dim]);
    }

    if (build_temp_equ)
    {
        test_fixture.registerTestField<EvalType>(
            no_slip_eval->_boundary_temperature);
        test_fixture.registerTestField<EvalType>(
            no_slip_eval->_boundary_grad_temperature);
    }

    if (continuity_model == ContinuityModel::EDAC)
    {
        test_fixture.registerTestField<EvalType>(
            no_slip_eval->_boundary_grad_lagrange_pressure);
    }

    // Evaluate incompressible free slip
    test_fixture.evaluate<EvalType>();

    // Expected values
    const double exp_lagrange_pres = set_lagrange_pressure ? 0 : 1.5;
    const double exp_grad_lagrange_pres[3]
        = {num_space_dim == 3 ? 5.15625 : 1.359375,
           num_space_dim == 3 ? -10.3125 : -2.71875,
           num_space_dim == 3 ? 15.46875 : nan_val};
    const double exp_temp
        = temperature_profile == TempProfile::isothermal ? 0.125 : 0.75;

    const double exp_grad_vel_0_3d[3] = {-0.25, 0.5, -0.75};
    const double exp_grad_vel_1_3d[3] = {-0.5, 1.0, -1.5};
    const double exp_grad_vel_2_3d[3] = {-0.125, 0.25, -0.375};

    const double exp_grad_vel_0_2d[2]
        = {exp_grad_vel_0_3d[0], exp_grad_vel_0_3d[1]};
    const double exp_grad_vel_1_2d[2]
        = {exp_grad_vel_1_3d[0], exp_grad_vel_1_3d[1]};

    const double* exp_grad_vel_0 = num_space_dim == 3 ? exp_grad_vel_0_3d
                                                      : exp_grad_vel_0_2d;
    const double* exp_grad_vel_1 = num_space_dim == 3 ? exp_grad_vel_1_3d
                                                      : exp_grad_vel_1_2d;
    const double* exp_grad_vel_2 = exp_grad_vel_2_3d;

    const double* exp_grad_temp = nullptr;
    double exp_grad_temp_3d[3];
    double exp_grad_temp_2d[2];

    if (temperature_profile == TempProfile::isothermal)
    {
        exp_grad_temp_3d[0] = -0.75;
        exp_grad_temp_3d[1] = 1.5;
        exp_grad_temp_3d[2] = -2.25;
        exp_grad_temp_2d[0] = -0.75;
        exp_grad_temp_2d[1] = 1.5;
    }
    else if (temperature_profile == TempProfile::adiabatic)
    {
        exp_grad_temp_3d[0] = 5.15625;
        exp_grad_temp_3d[1] = -10.3125;
        exp_grad_temp_3d[2] = 15.46875;
        exp_grad_temp_2d[0] = 1.359375;
        exp_grad_temp_2d[1] = -2.71875;
    }
    else
    {
        exp_grad_temp_3d[0] = 2.15625;
        exp_grad_temp_3d[1] = -4.3125;
        exp_grad_temp_3d[2] = 6.46875;
        exp_grad_temp_2d[0] = -1.640625;
        exp_grad_temp_2d[1] = 3.28125;
    }

    exp_grad_temp = (num_space_dim == 3) ? exp_grad_temp_3d : exp_grad_temp_2d;

    // Get no slip field
    auto boundary_lagrange_pressure_result
        = test_fixture.getTestFieldData<EvalType>(
            no_slip_eval->_boundary_lagrange_pressure);

    // Loop over quadrature points and mesh dimension
    const int num_point = boundary_lagrange_pressure_result.extent(1);
    for (int qp = 0; qp < num_point; ++qp)
    {
        // Lagrange pressure
        EXPECT_DOUBLE_EQ(exp_lagrange_pres,
                         fieldValue(boundary_lagrange_pressure_result, 0, qp));

        // Temperature
        if (build_temp_equ)
        {
            const auto boundary_temperature_result
                = test_fixture.getTestFieldData<EvalType>(
                    no_slip_eval->_boundary_temperature);
            EXPECT_DOUBLE_EQ(exp_temp,
                             fieldValue(boundary_temperature_result, 0, qp));
        }

        // Loop over mesh dimension to assert velocity and gradient vectors
        for (int d = 0; d < num_space_dim; d++)
        {
            const auto boundary_velocity_d_result
                = test_fixture.getTestFieldData<EvalType>(
                    no_slip_eval->_boundary_velocity[d]);
            EXPECT_DOUBLE_EQ(0.0,
                             fieldValue(boundary_velocity_d_result, 0, qp));

            const auto boundary_grad_velocity_0_result
                = test_fixture.getTestFieldData<EvalType>(
                    no_slip_eval->_boundary_grad_velocity[0]);
            EXPECT_DOUBLE_EQ(
                exp_grad_vel_0[d],
                fieldValue(boundary_grad_velocity_0_result, 0, qp, d));

            const auto boundary_grad_velocity_1_result
                = test_fixture.getTestFieldData<EvalType>(
                    no_slip_eval->_boundary_grad_velocity[1]);
            EXPECT_DOUBLE_EQ(
                exp_grad_vel_1[d],
                fieldValue(boundary_grad_velocity_1_result, 0, qp, d));

            if (num_space_dim > 2)
            {
                const auto boundary_grad_velocity_2_result
                    = test_fixture.getTestFieldData<EvalType>(
                        no_slip_eval->_boundary_grad_velocity[2]);
                EXPECT_DOUBLE_EQ(
                    exp_grad_vel_2[d],
                    fieldValue(boundary_grad_velocity_2_result, 0, qp, d));
            }

            if (build_temp_equ)
            {
                const auto boundary_grad_temperature_result
                    = test_fixture.getTestFieldData<EvalType>(
                        no_slip_eval->_boundary_grad_temperature);
                EXPECT_DOUBLE_EQ(
                    exp_grad_temp[d],
                    fieldValue(boundary_grad_temperature_result, 0, qp, d));
            }

            if (continuity_model == ContinuityModel::EDAC)
            {
                const auto boundary_grad_lagrange_pressure_result
                    = test_fixture.getTestFieldData<EvalType>(
                        no_slip_eval->_boundary_grad_lagrange_pressure);
                EXPECT_DOUBLE_EQ(
                    exp_grad_lagrange_pres[d],
                    fieldValue(
                        boundary_grad_lagrange_pressure_result, 0, qp, d));
            }
        }
    }
}

//---------------------------------------------------------------------------//
// Value parameterized test fixture
struct EvaluationTest : public testing::TestWithParam<ContinuityModel>
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

// 2-D incompressible isothermal no slip
TEST_P(EvaluationTest, isothermalresidual2D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 2>(
        false, false, continuity_model, TempProfile::isothermal);
}

TEST_P(EvaluationTest, isothermaljacobian2D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 2>(
        false, false, continuity_model, TempProfile::isothermal);
}

//---------------------------------------------------------------------------//
// 3-D incompressible isothermal no slip
TEST_P(EvaluationTest, isothermalresidual3D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 3>(
        false, false, continuity_model, TempProfile::isothermal);
}

TEST_P(EvaluationTest, isothermaljacobian3D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 3>(
        false, false, continuity_model, TempProfile::isothermal);
}

// 2-D incompressible adiabatic no slip
TEST_P(EvaluationTest, adiabaticresidual2D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 2>(
        false, true, continuity_model, TempProfile::adiabatic);
}

TEST_P(EvaluationTest, adiabaticjacobian2D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 2>(
        false, true, continuity_model, TempProfile::adiabatic);
}

//---------------------------------------------------------------------------//
// 3-D incompressible adiabatic no slip
TEST_P(EvaluationTest, adiabaticresidual3D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 3>(
        false, true, continuity_model, TempProfile::adiabatic);
}

TEST_P(EvaluationTest, adiabaticjacobian3D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 3>(
        false, true, continuity_model, TempProfile::adiabatic);
}

// 2-D incompressible heat flux no slip
TEST_P(EvaluationTest, fluxresidual2D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 2>(
        false, true, continuity_model, TempProfile::flux);
}

TEST_P(EvaluationTest, fluxjacobian2D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 2>(
        false, true, continuity_model, TempProfile::flux);
}

// 3-D incompressible heat flux no slip
TEST_P(EvaluationTest, fluxresidual3D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 3>(
        false, true, continuity_model, TempProfile::flux);
}

TEST_P(EvaluationTest, fluxjacobian3D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 3>(
        false, true, continuity_model, TempProfile::flux);
}

//---------------------------------------------------------------------------//
// 2-D incompressible no slip
TEST_P(EvaluationTest, residual2D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 2>(
        false, true, continuity_model, TempProfile::isothermal);
}

TEST_P(EvaluationTest, jacobian2D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 2>(
        false, true, continuity_model, TempProfile::isothermal);
}

//---------------------------------------------------------------------------//
// 3-D incompressible no slip
TEST_P(EvaluationTest, residual3D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 3>(
        false, true, continuity_model, TempProfile::isothermal);
}

TEST_P(EvaluationTest, jacobian3D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 3>(
        false, true, continuity_model, TempProfile::isothermal);
}

//---------------------------------------------------------------------------//
// 3-D incompressible no slip - set boundary lagrange pressure
TEST_P(EvaluationTest, residual3DBoundaryPressure)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 3>(
        true, true, continuity_model, TempProfile::isothermal);
}

TEST_P(EvaluationTest, jacobian3DBoundaryPressure)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 3>(
        true, true, continuity_model, TempProfile::isothermal);
}

//---------------------------------------------------------------------------//
// Generate test suite with continuity models
INSTANTIATE_TEST_SUITE_P(ContinuityModelType,
                         EvaluationTest,
                         testing::Values(ContinuityModel::AC,
                                         ContinuityModel::EDAC),
                         EvaluationTest::ParamNameGenerator{});

} // end namespace Test
} // end namespace VertexCFD
