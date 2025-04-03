#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleLaminarFlow.hpp"
#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"
#include <VertexCFD_EvaluatorTestHarness.hpp>

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_config.hpp>

#include <mpi.h>

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> _normals;

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
        , _normals("Side Normal", ir.dl_vector)
    {
        this->addEvaluatedField(_lagrange_pressure);
        this->addEvaluatedField(_grad_velocity_0);
        this->addEvaluatedField(_grad_velocity_1);
        this->addEvaluatedField(_grad_velocity_2);

        this->addEvaluatedField(_normals);

        if (_build_tmp_equ)
            this->addEvaluatedField(_grad_temperature);
        if (_continuity_model == ContinuityModel::EDAC)
            this->addEvaluatedField(_grad_lagrange_pressure);
        this->setName("Incompressible Laminar Flow Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        _lagrange_pressure.deep_copy(_u0 + _u1);
        _grad_velocity_0.deep_copy(_u0 * _u0);
        _grad_velocity_1.deep_copy(_u1 * _u1);
        _grad_velocity_2.deep_copy(_u2 * _u2);
        if (_build_tmp_equ)
            _grad_temperature.deep_copy(_u0 - _u1);
        if (_continuity_model == ContinuityModel::EDAC)
            _grad_lagrange_pressure.deep_copy(_u0 - _u1);

        Kokkos::parallel_for(
            "incompressible boundary laminar flow test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int c) const
    {
        const int num_point = _lagrange_pressure.extent(1);
        const int num_space_dim = _grad_velocity_0.extent(2);

        for (int qp = 0; qp < num_point; ++qp)
        {
            // Set gradient and normal vectors
            for (int d = 0; d < num_space_dim; ++d)
            {
                _normals(c, qp, d) = 0.0;
                if (d == 0)
                    _normals(c, qp, d) = -1.0;
            }
        }
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
    ip_coord_mirror(0, 0, 0) = 0.0;
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
    Teuchos::Array<double> origin_coord(num_space_dim);
    origin_coord[0] = 0.0;
    origin_coord[1] = 0.1;
    if (num_space_dim == 3)
    {
        origin_coord[2] = 0.0;
    }

    bc_params.set("Origin Coordinates", origin_coord);
    bc_params.set("Characteristic Radius", 2.1);
    bc_params.set("Average velocity", 3.0);
    const double T_bc = build_temp_equ ? 4.0 : nanval;
    if (build_temp_equ)
        bc_params.set("Temperature", T_bc);

    // Create evaluator.
    auto laminar_eval = Teuchos::rcp(
        new BoundaryCondition::IncompressibleLaminarFlow<EvalType,
                                                         panzer::Traits,
                                                         num_space_dim>(
            *test_fixture.ir, fluid_prop, bc_params, continuity_model_name));
    test_fixture.registerEvaluator<EvalType>(laminar_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(
        laminar_eval->_boundary_lagrange_pressure);
    for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
    {
        test_fixture.registerTestField<EvalType>(
            laminar_eval->_boundary_velocity[vel_dim]);
        test_fixture.registerTestField<EvalType>(
            laminar_eval->_boundary_grad_velocity[vel_dim]);
    }

    if (build_temp_equ)
    {
        test_fixture.registerTestField<EvalType>(
            laminar_eval->_boundary_temperature);
        test_fixture.registerTestField<EvalType>(
            laminar_eval->_boundary_grad_temperature);
    }

    if (continuity_model == ContinuityModel::EDAC)
    {
        test_fixture.registerTestField<EvalType>(
            laminar_eval->_boundary_grad_lagrange_pressure);
    }

    // Evaluate boundary values.
    test_fixture.evaluate<EvalType>();

    const double u_ref = num_space_dim == 2 ? 4.085299744897959
                                            : 4.147057823129252;

    // Check boundary values.
    const auto boundary_phi_result = test_fixture.getTestFieldData<EvalType>(
        laminar_eval->_boundary_lagrange_pressure);

    const int num_point = boundary_phi_result.extent(1);

    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(u0 + u1, fieldValue(boundary_phi_result, 0, qp));

        const auto boundary_velocity_0_result
            = test_fixture.getTestFieldData<EvalType>(
                laminar_eval->_boundary_velocity[0]);
        EXPECT_DOUBLE_EQ(u_ref, fieldValue(boundary_velocity_0_result, 0, qp));
        for (int vel_dim = 1; vel_dim < num_space_dim; ++vel_dim)
        {
            const auto boundary_velocity_d_result
                = test_fixture.getTestFieldData<EvalType>(
                    laminar_eval->_boundary_velocity[vel_dim]);
            EXPECT_DOUBLE_EQ(0.0,
                             fieldValue(boundary_velocity_d_result, 0, qp));
        }

        if (build_temp_equ)
        {
            const auto boundary_temperature_result
                = test_fixture.getTestFieldData<EvalType>(
                    laminar_eval->_boundary_temperature);
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
                        laminar_eval->_boundary_grad_velocity[vel_dim]);
                EXPECT_DOUBLE_EQ(
                    exp_val,
                    fieldValue(boundary_grad_velocity_d_result, 0, qp, d));
            }

            if (build_temp_equ)
            {
                const double exp_val = vel[0] - vel[1];
                const auto boundary_grad_temperature_result
                    = test_fixture.getTestFieldData<EvalType>(
                        laminar_eval->_boundary_grad_temperature);
                EXPECT_DOUBLE_EQ(
                    exp_val,
                    fieldValue(boundary_grad_temperature_result, 0, qp, d));
            }

            if (continuity_model == ContinuityModel::EDAC)
            {
                const double exp_val = vel[0] - vel[1];
                const auto boundary_grad_lagrange_pressure_result
                    = test_fixture.getTestFieldData<EvalType>(
                        laminar_eval->_boundary_grad_lagrange_pressure);
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
// 2-D incompressible isothermal LaminarFlow
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
// 3-D incompressible isothermal LaminarFlow
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
// 2-D incompressible LaminarFlow
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
// 3-D incompressible LaminarFlow
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

} // end namespace Test
} // end namespace VertexCFD
