#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleCavityLid.hpp"
#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"
#include <VertexCFD_EvaluatorTestHarness.hpp>

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_config.hpp>

#include <mpi.h>

#include <iostream>
#include <stdexcept>
#include <string>

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
// Test data dependencies.
template<class EvalType>
struct Dependencies : public PHX::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    double _u0, _u1, _u2;
    bool _build_tmp_equ;
    ContinuityModel _continuity_model;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _lagrange_pressure;
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
                 const double u0,
                 const double u1,
                 const double u2,
                 const bool build_tmp_equ,
                 const ContinuityModel continuity_model)
        : _u0(u0)
        , _u1(u1)
        , _u2(u2)
        , _build_tmp_equ(build_tmp_equ)
        , _continuity_model(continuity_model)
        , _lagrange_pressure("lagrange_pressure", ir.dl_scalar)
        , _grad_velocity_0("GRAD_velocity_0", ir.dl_vector)
        , _grad_velocity_1("GRAD_velocity_1", ir.dl_vector)
        , _grad_velocity_2("GRAD_velocity_2", ir.dl_vector)
        , _grad_temperature("GRAD_temperature", ir.dl_vector)
        , _grad_lagrange_pressure("GRAD_lagrange_pressure", ir.dl_vector)
    {
        this->addEvaluatedField(_lagrange_pressure);
        this->addEvaluatedField(_grad_velocity_0);
        this->addEvaluatedField(_grad_velocity_1);
        this->addEvaluatedField(_grad_velocity_2);
        if (_build_tmp_equ)
            this->addEvaluatedField(_grad_temperature);
        if (_continuity_model == ContinuityModel::EDAC)
            this->addEvaluatedField(_grad_lagrange_pressure);
        this->setName("Incompressible Cavity Lid Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData /**d**/) override
    {
        _lagrange_pressure.deep_copy(_u0 + _u1);
        _grad_velocity_0.deep_copy(_u0 * _u0);
        _grad_velocity_1.deep_copy(_u1 * _u1);
        _grad_velocity_2.deep_copy(_u2 * _u2);
        if (_build_tmp_equ)
            _grad_temperature.deep_copy(_u0 - _u1);
        if (_continuity_model == ContinuityModel::EDAC)
            _grad_lagrange_pressure.deep_copy(_u0 - _u1);
    }
};

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval(const bool build_temp_equ, const ContinuityModel continuity_model)
{
    // Test fixture
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    std::string continuity_model_name = "";
    switch (continuity_model)
    {
        case (ContinuityModel::AC):
            continuity_model_name = "AC";
            break;
        case (ContinuityModel::EDAC):
            continuity_model_name = "EDAC";
            break;
    }

    // Set non-trivial values for quadrature points
    auto ip_coord_view
        = test_fixture.int_values->ip_coordinates.get_static_view();
    auto ip_coord_mirror = Kokkos::create_mirror(ip_coord_view);
    ip_coord_mirror(0, 0, 1) = 0.7375;
    if (num_space_dim == 3)
        ip_coord_mirror(0, 0, 2) = 0.9775;
    Kokkos::deep_copy(ip_coord_view, ip_coord_mirror);

    // Create dependencies
    const double nanval = std::numeric_limits<double>::quiet_NaN();
    const double u0 = 0.2;
    const double u1 = 0.3;
    const double u2 = num_space_dim == 3 ? 0.4 : nanval;
    const double vel[3] = {u0, u1, u2};

    const auto dep_eval = Teuchos::rcp(new Dependencies<EvalType>(
        *test_fixture.ir, u0, u1, u2, build_temp_equ, continuity_model));
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

    // Create the param list to initialize the evaluator
    Teuchos::ParameterList bc_params;
    bc_params.set("Wall Normal Direction", 0);
    bc_params.set("Velocity Direction", 1);
    bc_params.set("Wall Velocity", 2.0);
    bc_params.set("Half Width", 1.0);
    const double T_bc = build_temp_equ ? 4.0 : nanval;
    if (build_temp_equ)
        bc_params.set("Temperature", T_bc);

    // Create evaluator.
    auto bc_eval = Teuchos::rcp(
        new BoundaryCondition::IncompressibleCavityLid<EvalType,
                                                       panzer::Traits,
                                                       num_space_dim>(
            *test_fixture.ir, fluid_prop, bc_params, continuity_model_name));
    test_fixture.registerEvaluator<EvalType>(bc_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(
        bc_eval->_boundary_lagrange_pressure);
    for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
    {
        test_fixture.registerTestField<EvalType>(
            bc_eval->_boundary_velocity[vel_dim]);
        test_fixture.registerTestField<EvalType>(
            bc_eval->_boundary_grad_velocity[vel_dim]);
    }
    if (build_temp_equ)
    {
        test_fixture.registerTestField<EvalType>(
            bc_eval->_boundary_temperature);
        test_fixture.registerTestField<EvalType>(
            bc_eval->_boundary_grad_temperature);
    }
    if (continuity_model == ContinuityModel::EDAC)
    {
        test_fixture.registerTestField<EvalType>(
            bc_eval->_boundary_grad_lagrange_pressure);
    }

    // Evaluate boundary values.
    test_fixture.evaluate<EvalType>();

    const double u_exp = num_space_dim == 3 ? 0.22404972666527662
                                            : 1.9833708189379662;

    // Check boundary values.
    const auto boundary_phi_result = test_fixture.getTestFieldData<EvalType>(
        bc_eval->_boundary_lagrange_pressure);

    const int num_point = boundary_phi_result.extent(1);

    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(u0 + u1, fieldValue(boundary_phi_result, 0, qp));

        for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
        {
            const auto boundary_velocity_d_result
                = test_fixture.getTestFieldData<EvalType>(
                    bc_eval->_boundary_velocity[vel_dim]);
            if (vel_dim == 1)
            {
                EXPECT_DOUBLE_EQ(
                    u_exp, fieldValue(boundary_velocity_d_result, 0, qp));
            }
            else
            {
                EXPECT_DOUBLE_EQ(
                    0.0, fieldValue(boundary_velocity_d_result, 0, qp));
            }
        }

        if (build_temp_equ)
        {
            const auto boundary_temperature_result
                = test_fixture.getTestFieldData<EvalType>(
                    bc_eval->_boundary_temperature);
            EXPECT_DOUBLE_EQ(T_bc,
                             fieldValue(boundary_temperature_result, 0, qp));
        }

        for (int d = 0; d < num_space_dim; ++d)
        {
            for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
            {
                const double exp_val = vel[vel_dim] * vel[vel_dim];
                const auto boundary_grad_velocity_d_result
                    = test_fixture.getTestFieldData<EvalType>(
                        bc_eval->_boundary_grad_velocity[vel_dim]);
                EXPECT_DOUBLE_EQ(
                    exp_val,
                    fieldValue(boundary_grad_velocity_d_result, 0, qp, d));
            }

            if (build_temp_equ)
            {
                const double exp_val = vel[0] - vel[1];
                const auto boundary_grad_temperature_result
                    = test_fixture.getTestFieldData<EvalType>(
                        bc_eval->_boundary_grad_temperature);
                EXPECT_DOUBLE_EQ(
                    exp_val,
                    fieldValue(boundary_grad_temperature_result, 0, qp, d));
            }

            if (continuity_model == ContinuityModel::EDAC)
            {
                const double exp_val = vel[0] - vel[1];
                const auto boundary_grad_lagrange_pressure_result
                    = test_fixture.getTestFieldData<EvalType>(
                        bc_eval->_boundary_grad_lagrange_pressure);
                EXPECT_DOUBLE_EQ(
                    exp_val,
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

//---------------------------------------------------------------------------//
// 2-D incompressible isothermal cavity lid
TEST_P(EvaluationTest, isothermalresidual2D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 2>(false, continuity_model);
}

TEST_P(EvaluationTest, isothermaljacobian2D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 2>(false, continuity_model);
}

//---------------------------------------------------------------------------//
// 3-D incompressible isothermal cavity lid
TEST_P(EvaluationTest, isothermalresidual3D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 3>(false, continuity_model);
}

TEST_P(EvaluationTest, isothermaljacobian3D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 3>(false, continuity_model);
}

//---------------------------------------------------------------------------//
// 2-D incompressible cavity lid
TEST_P(EvaluationTest, residual2D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 2>(true, continuity_model);
}

TEST_P(EvaluationTest, jacobian2D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 2>(true, continuity_model);
}

//---------------------------------------------------------------------------//
// 3-D incompressible cavity lid
TEST_P(EvaluationTest, residual3D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 3>(true, continuity_model);
}

TEST_P(EvaluationTest, jacobian3D)
{
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 3>(true, continuity_model);
}

//---------------------------------------------------------------------------//
// Generate test suite with continuity models
INSTANTIATE_TEST_SUITE_P(ContinuityModelType,
                         EvaluationTest,
                         testing::Values(ContinuityModel::AC,
                                         ContinuityModel::EDAC),
                         EvaluationTest::ParamNameGenerator{});

//---------------------------------------------------------------------------//
// Error message cases
enum class ErrorMessages
{
    wall,
    velocity,
    equal
};

//---------------------------------------------------------------------------//
// Test for error messages
template<class EvalType, int NumSpaceDim>
void testErrors(const ErrorMessages error_message)
{
    // Test fixture
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Create dependencies
    const double nanval = std::numeric_limits<double>::quiet_NaN();
    const double u2 = num_space_dim == 3 ? 0.4 : nanval;

    auto dep_eval = Teuchos::rcp(new Dependencies<EvalType>(
        *test_fixture.ir, 1.0, 2.0, u2, false, ContinuityModel::EDAC));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Equation of state
    Teuchos::ParameterList fluid_prop_list;
    fluid_prop_list.set("Kinematic viscosity", 0.375);
    fluid_prop_list.set("Artificial compressibility", 2.0);
    fluid_prop_list.set("Build Temperature Equation", false);
    const FluidProperties::ConstantFluidProperties fluid_prop(fluid_prop_list);

    std::string msg = "";
    int wall_dir = 0;
    int vel_dir = 1;

    // Set wall and velocity direction based on error message
    switch (error_message)
    {
        case (ErrorMessages::wall):
            wall_dir = num_space_dim;
            msg
                = "Wall normal direction greater than "
                  "number of solution dimensions in Cavity Lid boundary "
                  "condition.";
            break;
        case (ErrorMessages::velocity):
            vel_dir = num_space_dim;
            msg
                = "Velocity direction greater than "
                  "number of solution dimensions in Cavity Lid boundary "
                  "condition.";
            break;
        case (ErrorMessages::equal):
            vel_dir = 0;
            msg
                = "Velocity direction is same as wall normal in "
                  "Cavity Lid boundary condition.";
            break;
    }

    // Create the param list to initialize the evaluator
    Teuchos::ParameterList bc_params;
    bc_params.set("Wall Normal Direction", wall_dir);
    bc_params.set("Velocity Direction", vel_dir);
    bc_params.set("Wall Velocity", 2.0);
    bc_params.set("Half Width", 1.0);

    using eval = BoundaryCondition::
        IncompressibleCavityLid<EvalType, panzer::Traits, num_space_dim>;

    ASSERT_THROW(
        try {
            auto cavity_lid_eval
                = eval(*test_fixture.ir, fluid_prop, bc_params, "EDAC");
        } catch (const std::runtime_error& e) {
            EXPECT_EQ(msg, e.what());
            throw;
        },
        std::runtime_error);
}

//---------------------------------------------------------------------------//
// Value parameterized test fixture
struct ErrorTest : public testing::TestWithParam<ErrorMessages>
{
    // Case generator for parameterized test
    struct ParamNameGenerator
    {
        std::string
        operator()(const testing::TestParamInfo<ErrorMessages>& info) const
        {
            const auto error_message = info.param;
            switch (error_message)
            {
                case (ErrorMessages::wall):
                    return "wall";
                case (ErrorMessages::velocity):
                    return "velocity";
                case (ErrorMessages::equal):
                    return "equal";
                default:
                    return "INVALID_NAME";
            }
        }
    };
};

//---------------------------------------------------------------------------//
// Residual evaluation, 2D
TEST_P(ErrorTest, residual2D)
{
    ErrorMessages error_message;
    error_message = GetParam();
    testErrors<panzer::Traits::Residual, 2>(error_message);
}

//---------------------------------------------------------------------------//
// Residual evaluation, 3D
TEST_P(ErrorTest, residual3D)
{
    ErrorMessages error_message;
    error_message = GetParam();
    testErrors<panzer::Traits::Residual, 3>(error_message);
}

//---------------------------------------------------------------------------//
// Jacobian evaluation, 2D
TEST_P(ErrorTest, jacobian2D)
{
    ErrorMessages error_message;
    error_message = GetParam();
    testErrors<panzer::Traits::Jacobian, 2>(error_message);
}

//---------------------------------------------------------------------------//
// Jacobian evaluation, 3D
TEST_P(ErrorTest, jacobian3D)
{
    ErrorMessages error_message;
    error_message = GetParam();
    testErrors<panzer::Traits::Jacobian, 3>(error_message);
}

//---------------------------------------------------------------------------//
// Generate test suite with errors
INSTANTIATE_TEST_SUITE_P(CavityLidBC,
                         ErrorTest,
                         testing::Values(ErrorMessages::wall,
                                         ErrorMessages::velocity,
                                         ErrorMessages::equal),
                         ErrorTest::ParamNameGenerator{});

//---------------------------------------------------------------------------//
} // end namespace Test
} // end namespace VertexCFD
