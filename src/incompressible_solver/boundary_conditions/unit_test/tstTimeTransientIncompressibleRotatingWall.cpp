#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleRotatingWall.hpp"
#include <VertexCFD_EvaluatorTestHarness.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_MDField.hpp>

#include <Teuchos_ParameterList.hpp>

#include <iostream>

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _lagrange_pressure;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_velocity_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_velocity_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_velocity_2;

    bool _build_tmp_equ;
    ContinuityModel _continuity_model;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_temperature;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_lagrange_pressure;

    Dependencies(const panzer::IntegrationRule& ir,
                 const bool build_tmp_equ,
                 const ContinuityModel continuity_model)
        : _lagrange_pressure("lagrange_pressure", ir.dl_scalar)
        , _grad_velocity_0("GRAD_velocity_0", ir.dl_vector)
        , _grad_velocity_1("GRAD_velocity_1", ir.dl_vector)
        , _grad_velocity_2("GRAD_velocity_2", ir.dl_vector)
        , _build_tmp_equ(build_tmp_equ)
        , _continuity_model(continuity_model)
        , _grad_temperature("GRAD_temperature", ir.dl_vector)
        , _grad_lagrange_pressure("GRAD_lagrange_pressure", ir.dl_vector)
    {
        this->addEvaluatedField(_lagrange_pressure);
        this->addEvaluatedField(_grad_velocity_0);
        this->addEvaluatedField(_grad_velocity_1);
        this->addEvaluatedField(_grad_velocity_2);

        if (build_tmp_equ)
            this->addEvaluatedField(_grad_temperature);
        if (_continuity_model == ContinuityModel::EDAC)
            this->addEvaluatedField(_grad_lagrange_pressure);

        this->setName(
            "Time Transient Incompressible Rotating Wall Unit Test "
            "Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        // Initialize pressure
        _lagrange_pressure.deep_copy(0.4);

        // Initialize gradients
        Kokkos::parallel_for(
            "incompressible rotating wall test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = _lagrange_pressure.extent(1);
        const int num_space_dim = _grad_velocity_0.extent(2);

        // Loop over quadrature points and mesh dimension
        for (int qp = 0; qp < num_point; ++qp)
        {
            for (int d = 0; d < num_space_dim; d++)
            {
                const int dqp = (d + 1 + num_space_dim) * (qp + 1);
                _grad_velocity_0(c, qp, d) = 0.1 * dqp;
                _grad_velocity_1(c, qp, d) = 0.2 * dqp;
                if (num_space_dim == 3)
                    _grad_velocity_2(c, qp, d) = 0.3 * dqp;

                if (_build_tmp_equ)
                    _grad_temperature(c, qp, d) = 325 * dqp;

                if (_continuity_model == ContinuityModel::EDAC)
                    _grad_lagrange_pressure(c, qp, d) = 0.4;
            }
        }
    }
};

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval(const Kokkos::Array<double, 3> time_values,
              const Kokkos::Array<double, 1> init_values,
              const Kokkos::Array<double, NumSpaceDim> exp_fields,
              const bool build_temp_equ,
              const ContinuityModel continuity_model)
{
    // Test fixture
    constexpr int num_space_dim = NumSpaceDim;
    const int num_grad_dim = num_space_dim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_grad_dim, integration_order, basis_order);

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

    // Create dependencies
    const double angular_velocity = 2.0;
    const double angular_velocity_init = init_values[0];
    const double time_init = time_values[0];
    const double time_final = time_values[1];
    const double time = time_values[2];

    // Initialize values
    const double T_wall
        = build_temp_equ ? 325 : std::numeric_limits<double>::quiet_NaN();

    // Set non-trivial quadrature points to avoid x = y
    auto ip_coord_view
        = test_fixture.int_values->ip_coordinates.get_static_view();
    auto ip_coord_mirror = Kokkos::create_mirror(ip_coord_view);
    ip_coord_mirror(0, 0, 0) = 0.7375;
    ip_coord_mirror(0, 0, 1) = 0.9775;
    Kokkos::deep_copy(ip_coord_view, ip_coord_mirror);

    // Initialize dependecy evaluator
    const auto dep_eval = Teuchos::rcp(new Dependencies<EvalType>(
        *test_fixture.ir, build_temp_equ, continuity_model));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Thermophysical properties
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
    bc_params.set("Angular Velocity", angular_velocity);
    if (init_values[0] > 0.0)
        bc_params.set("Angular Velocity Initial", angular_velocity_init);
    if (time_init > 0.0)
        bc_params.set("Time Initial", time_init);
    if (time_final > 0.0)
        bc_params.set("Time Final", time_final);
    if (build_temp_equ)
        bc_params.set("Wall Temperature", T_wall);

    // Create evaluator.
    auto isotherm_eval = Teuchos::rcp(
        new BoundaryCondition::IncompressibleRotatingWall<EvalType,
                                                          panzer::Traits,
                                                          num_space_dim>(
            *test_fixture.ir, fluid_prop, bc_params, continuity_model_name));
    test_fixture.registerEvaluator<EvalType>(isotherm_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(
        isotherm_eval->_boundary_lagrange_pressure);
    for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
    {
        test_fixture.registerTestField<EvalType>(
            isotherm_eval->_boundary_velocity[vel_dim]);
        test_fixture.registerTestField<EvalType>(
            isotherm_eval->_boundary_grad_velocity[vel_dim]);
    }

    if (build_temp_equ)
    {
        test_fixture.registerTestField<EvalType>(
            isotherm_eval->_boundary_temperature);
        test_fixture.registerTestField<EvalType>(
            isotherm_eval->_boundary_grad_temperature);
    }

    if (continuity_model == ContinuityModel::EDAC)
    {
        test_fixture.registerTestField<EvalType>(
            isotherm_eval->_boundary_grad_lagrange_pressure);
    }

    // Set time
    test_fixture.setTime(time);

    // Evaluate
    test_fixture.evaluate<EvalType>();

    // Get field
    const auto boundary_phi_result = test_fixture.getTestFieldData<EvalType>(
        isotherm_eval->_boundary_lagrange_pressure);

    // Assert variables and gradients at each quadrature points
    const int num_point = boundary_phi_result.extent(1);
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(0.4, fieldValue(boundary_phi_result, 0, qp));

        for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
        {
            const auto boundary_velocity_d_result
                = test_fixture.getTestFieldData<EvalType>(
                    isotherm_eval->_boundary_velocity[vel_dim]);
            EXPECT_DOUBLE_EQ(exp_fields[vel_dim],
                             fieldValue(boundary_velocity_d_result, 0, qp));
        }

        if (build_temp_equ)
        {
            const auto boundary_temperature_result
                = test_fixture.getTestFieldData<EvalType>(
                    isotherm_eval->_boundary_temperature);
            EXPECT_DOUBLE_EQ(T_wall,
                             fieldValue(boundary_temperature_result, 0, qp));
        }

        for (int d = 0; d < num_grad_dim; ++d)
        {
            const int dqp = (d + 1 + num_space_dim) * (qp + 1);
            for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
            {
                const auto boundary_grad_velocity_d_result
                    = test_fixture.getTestFieldData<EvalType>(
                        isotherm_eval->_boundary_grad_velocity[vel_dim]);
                EXPECT_DOUBLE_EQ(
                    (vel_dim + 1) * 0.1 * dqp,
                    fieldValue(boundary_grad_velocity_d_result, 0, qp, d));

                if (build_temp_equ)
                {
                    const auto boundary_grad_temperature_result
                        = test_fixture.getTestFieldData<EvalType>(
                            isotherm_eval->_boundary_grad_temperature);
                    EXPECT_DOUBLE_EQ(
                        T_wall * dqp,
                        fieldValue(boundary_grad_temperature_result, 0, qp, d));
                }

                if (continuity_model == ContinuityModel::EDAC)
                {
                    const double exp_val = 0.4;
                    const auto boundary_grad_lagrange_pressure_result
                        = test_fixture.getTestFieldData<EvalType>(
                            isotherm_eval->_boundary_grad_lagrange_pressure);
                    EXPECT_DOUBLE_EQ(
                        exp_val,
                        fieldValue(
                            boundary_grad_lagrange_pressure_result, 0, qp, d));
                }
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
// 2-D: time transient incompressible rotating wall - steady
TEST_P(EvaluationTest, steadyresidual2D)
{
    const Kokkos::Array<double, 3> time_values = {-0.5, -3.0, 3.0};
    const Kokkos::Array<double, 1> init_values = {-1.0};
    const Kokkos::Array<double, 2> exp_fields = {-1.955, 1.475};
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 2>(
        time_values, init_values, exp_fields, false, continuity_model);
}

TEST_P(EvaluationTest, steadyjacobian2D)
{
    const Kokkos::Array<double, 3> time_values = {-0.5, -3.0, 3.0};
    const Kokkos::Array<double, 1> init_values = {-1.0};
    const Kokkos::Array<double, 2> exp_fields = {-1.955, 1.475};
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 2>(
        time_values, init_values, exp_fields, false, continuity_model);
}

//---------------------------------------------------------------------------//
// 2-D: time transient incompressible rotating wall - time > time_final
TEST_P(EvaluationTest, timefinalresidual2D)
{
    const Kokkos::Array<double, 3> time_values = {0.5, 3.0, 3.5};
    const Kokkos::Array<double, 1> init_values = {1.0};
    const Kokkos::Array<double, 2> exp_fields = {-1.955, 1.475};
    ContinuityModel continuity_model;
    continuity_model = GetParam();

    testEval<panzer::Traits::Residual, 2>(
        time_values, init_values, exp_fields, false, continuity_model);
}

TEST_P(EvaluationTest, timefinaljacobian2D)
{
    const Kokkos::Array<double, 3> time_values = {0.5, 3.0, 3.5};
    const Kokkos::Array<double, 1> init_values = {1.0};
    const Kokkos::Array<double, 2> exp_fields = {-1.955, 1.475};
    ContinuityModel continuity_model;
    continuity_model = GetParam();

    testEval<panzer::Traits::Jacobian, 2>(
        time_values, init_values, exp_fields, false, continuity_model);
}

//---------------------------------------------------------------------------//
// 2-D: time transient incompressible rotating wall - time < time_init
TEST_P(EvaluationTest, timeinitresidual2D)
{
    const Kokkos::Array<double, 3> time_values = {0.5, 3.0, 0.2};
    const Kokkos::Array<double, 1> init_values = {1.0};
    const Kokkos::Array<double, 2> exp_fields = {-0.9775, 0.7375};
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 2>(
        time_values, init_values, exp_fields, false, continuity_model);
}

TEST_P(EvaluationTest, timeinitjacobian2D)
{
    const Kokkos::Array<double, 3> time_values = {0.5, 3.0, 0.2};
    const Kokkos::Array<double, 1> init_values = {1.0};
    const Kokkos::Array<double, 2> exp_fields = {-0.9775, 0.7375};
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 2>(
        time_values, init_values, exp_fields, false, continuity_model);
}

//---------------------------------------------------------------------------//
// 2-D: time transient incompressible rotating wall - time_init < time <
// time_final
TEST_P(EvaluationTest, timeintermediateresidual2D)
{
    const Kokkos::Array<double, 3> time_values = {0.5, 3.0, 1.5};
    const Kokkos::Array<double, 1> init_values = {1.0};
    const Kokkos::Array<double, 2> exp_fields = {-1.3685, 1.0325};
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 2>(
        time_values, init_values, exp_fields, false, continuity_model);
}

TEST_P(EvaluationTest, timeintermediatejacobian2D)
{
    const Kokkos::Array<double, 3> time_values = {0.5, 3.0, 1.5};
    const Kokkos::Array<double, 1> init_values = {1.0};
    const Kokkos::Array<double, 2> exp_fields = {-1.3685, 1.0325};
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 2>(
        time_values, init_values, exp_fields, false, continuity_model);
}

//---------------------------------------------------------------------------//
// 3-D: time transient incompressible rotating wall - time_init < time <
// time_final
TEST_P(EvaluationTest, timeintermediateresidual3D)
{
    const Kokkos::Array<double, 3> time_values = {0.5, 3.0, 1.5};
    const Kokkos::Array<double, 1> init_values = {1.0};
    const Kokkos::Array<double, 3> exp_fields = {-1.3685, 1.0325, 0.0};
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 3>(
        time_values, init_values, exp_fields, false, continuity_model);
}

TEST_P(EvaluationTest, timeintermediatejacobian3D)
{
    const Kokkos::Array<double, 3> time_values = {0.5, 3.0, 1.5};
    const Kokkos::Array<double, 1> init_values = {1.0};
    const Kokkos::Array<double, 3> exp_fields = {-1.3685, 1.0325, 0.0};
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 3>(
        time_values, init_values, exp_fields, false, continuity_model);
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
