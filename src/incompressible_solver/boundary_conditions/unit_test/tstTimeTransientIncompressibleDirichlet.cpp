#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "incompressible_solver/boundary_conditions/VertexCFD_BoundaryState_IncompressibleDirichlet.hpp"

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

    double _phi, _u0, _u1, _u2;
    ContinuityModel _continuity_model;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _lagrange_pressure;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_velocity_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_velocity_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_velocity_2;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_lagrange_pressure;

    Dependencies(const panzer::IntegrationRule& ir,
                 const double phi,
                 const double u0,
                 const double u1,
                 const double u2,
                 const ContinuityModel continuity_model)
        : _phi(phi)
        , _u0(u0)
        , _u1(u1)
        , _u2(u2)
        , _continuity_model(continuity_model)
        , _lagrange_pressure("lagrange_pressure", ir.dl_scalar)
        , _grad_velocity_0("GRAD_velocity_0", ir.dl_vector)
        , _grad_velocity_1("GRAD_velocity_1", ir.dl_vector)
        , _grad_velocity_2("GRAD_velocity_2", ir.dl_vector)
        , _grad_lagrange_pressure("GRAD_lagrange_pressure", ir.dl_vector)
    {
        this->addEvaluatedField(_lagrange_pressure);
        this->addEvaluatedField(_grad_velocity_0);
        this->addEvaluatedField(_grad_velocity_1);
        this->addEvaluatedField(_grad_velocity_2);
        if (_continuity_model == ContinuityModel::EDAC)
            this->addEvaluatedField(_grad_lagrange_pressure);
        this->setName(
            "Time Transient Incompressible Dirichlet Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData /**d**/) override
    {
        _lagrange_pressure.deep_copy(_phi);
        _grad_velocity_0.deep_copy(_u0);
        _grad_velocity_1.deep_copy(_u1);
        _grad_velocity_2.deep_copy(_u2);
        if (_continuity_model == ContinuityModel::EDAC)
            _grad_lagrange_pressure.deep_copy(_u0 - _u1);
    }
};

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval(const Kokkos::Array<double, 3> time_values,
              const Kokkos::Array<double, NumSpaceDim> final_values,
              const Kokkos::Array<double, NumSpaceDim> init_values,
              const Kokkos::Array<double, NumSpaceDim> exp_fields,
              const ContinuityModel continuity_model)
{
    // Test fixture
    constexpr int num_space_dim = NumSpaceDim;
    const int num_grad_dim = 2;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_grad_dim, integration_order, basis_order);

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

    // Create dependencies
    const double phi = 3.0;
    const double u0 = final_values[0];
    const double u1 = final_values[1];
    const double u2 = num_space_dim == 3 ? final_values[2] : 0.0;
    const double u0_init = init_values[0];
    const double u1_init = init_values[1];
    const double u2_init = num_space_dim == 3 ? init_values[2] : -1.0;
    const double time_init = time_values[0];
    const double time_final = time_values[1];
    const double time = time_values[2];

    const auto dep_eval = Teuchos::rcp(new Dependencies<EvalType>(
        *test_fixture.ir, phi, u0, u1, u2, continuity_model));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Flow parameters
    const bool build_temp_equ = false;
    Teuchos::ParameterList fluid_param_list;
    fluid_param_list.set("Build Temperature Equation", build_temp_equ);

    // Create the param list to initialize the evaluator
    Teuchos::ParameterList bc_params;
    bc_params.set("velocity_0", u0);
    bc_params.set("velocity_1", u1);
    if (num_space_dim == 3)
        bc_params.set("velocity_2", u2);
    if (u0_init > 0.0)
        bc_params.set("velocity_0_init", u0_init);
    if (u1_init > 0.0)
        bc_params.set("velocity_1_init", u1_init);
    if (u2_init > 0.0)
        bc_params.set("velocity_2_init", u2_init);
    if (time_init > 0.0)
        bc_params.set("Time Initial", time_init);
    if (time_final > 0.0)
        bc_params.set("Time Final", time_final);

    // Create dirichlet evaluator.
    auto dirichlet_eval = Teuchos::rcp(
        new BoundaryCondition::
            IncompressibleDirichlet<EvalType, panzer::Traits, num_space_dim>(
                *test_fixture.ir, fluid_param_list, bc_params, is_edac));
    test_fixture.registerEvaluator<EvalType>(dirichlet_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(
        dirichlet_eval->_boundary_lagrange_pressure);
    for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
    {
        test_fixture.registerTestField<EvalType>(
            dirichlet_eval->_boundary_velocity[vel_dim]);
        test_fixture.registerTestField<EvalType>(
            dirichlet_eval->_boundary_grad_velocity[vel_dim]);
    }

    if (continuity_model == ContinuityModel::EDAC)
    {
        test_fixture.registerTestField<EvalType>(
            dirichlet_eval->_boundary_grad_lagrange_pressure);
    }

    // Set time
    test_fixture.setTime(time);

    // Evaluate time transient dirichlet BC.
    test_fixture.evaluate<EvalType>();

    // Check the time transient dirichlet BC.
    const auto boundary_phi_result = test_fixture.getTestFieldData<EvalType>(
        dirichlet_eval->_boundary_lagrange_pressure);

    const int num_point = boundary_phi_result.extent(1);

    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(phi, fieldValue(boundary_phi_result, 0, qp));
        for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
        {
            const auto boundary_velocity_d_result
                = test_fixture.getTestFieldData<EvalType>(
                    dirichlet_eval->_boundary_velocity[vel_dim]);
            EXPECT_DOUBLE_EQ(exp_fields[vel_dim],
                             fieldValue(boundary_velocity_d_result, 0, qp));
        }

        for (int d = 0; d < num_grad_dim; ++d)
        {
            for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
            {
                // The gradients are initialized with entries from
                // `final_values`.
                const double exp_val = final_values[vel_dim];
                const auto boundary_grad_velocity_d_result
                    = test_fixture.getTestFieldData<EvalType>(
                        dirichlet_eval->_boundary_grad_velocity[vel_dim]);
                EXPECT_DOUBLE_EQ(
                    exp_val,
                    fieldValue(boundary_grad_velocity_d_result, 0, qp, d));
            }

            if (continuity_model == ContinuityModel::EDAC)
            {
                const double exp_val = final_values[0] - final_values[1];
                const auto boundary_grad_lagrange_pressure_result
                    = test_fixture.getTestFieldData<EvalType>(
                        dirichlet_eval->_boundary_grad_lagrange_pressure);
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
// 2-D: time transient dirichlet - steady
TEST_P(EvaluationTest, steadyresidual2D)
{
    const Kokkos::Array<double, 3> time_values = {-0.5, -2.0, 3.0};
    const Kokkos::Array<double, 2> final_values = {2.0, 3.0};
    const Kokkos::Array<double, 2> init_values = {-1.0, -1.2};
    const Kokkos::Array<double, 2> exp_fields = {2.0, 3.0};
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 2>(
        time_values, final_values, init_values, exp_fields, continuity_model);
}

TEST_P(EvaluationTest, steadyjacobian2D)
{
    const Kokkos::Array<double, 3> time_values = {-0.5, -2.0, 3.0};
    const Kokkos::Array<double, 2> final_values = {2.0, 3.0};
    const Kokkos::Array<double, 2> init_values = {-1.0, -1.2};
    const Kokkos::Array<double, 2> exp_fields = {2.0, 3.0};
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 2>(
        time_values, final_values, init_values, exp_fields, continuity_model);
}

//---------------------------------------------------------------------------//
// 2-D: time transient dirichlet - time > time_final
TEST_P(EvaluationTest, timefinalresidual2D)
{
    const Kokkos::Array<double, 3> time_values = {0.5, 2.0, 3.0};
    const Kokkos::Array<double, 2> final_values = {2.0, 3.0};
    const Kokkos::Array<double, 2> init_values = {1.0, 1.2};
    const Kokkos::Array<double, 2> exp_fields = {2.0, 3.0};
    ContinuityModel continuity_model;
    continuity_model = GetParam();

    testEval<panzer::Traits::Residual, 2>(
        time_values, final_values, init_values, exp_fields, continuity_model);
}

TEST_P(EvaluationTest, timefinaljacobian2D)
{
    const Kokkos::Array<double, 3> time_values = {0.5, 2.0, 3.0};
    const Kokkos::Array<double, 2> final_values = {2.0, 3.0};
    const Kokkos::Array<double, 2> init_values = {1.0, 1.2};
    const Kokkos::Array<double, 2> exp_fields = {2.0, 3.0};
    ContinuityModel continuity_model;
    continuity_model = GetParam();

    testEval<panzer::Traits::Jacobian, 2>(
        time_values, final_values, init_values, exp_fields, continuity_model);
}

//---------------------------------------------------------------------------//
// 2-D: time transient dirichlet - time < time_init
TEST_P(EvaluationTest, timeinitresidual2D)
{
    const Kokkos::Array<double, 3> time_values = {0.5, 1.0, 0.2};
    const Kokkos::Array<double, 2> final_values = {2.0, 3.0};
    const Kokkos::Array<double, 2> init_values = {1.0, 1.2};
    const Kokkos::Array<double, 2> exp_fields = {1.0, 1.2};
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 2>(
        time_values, final_values, init_values, exp_fields, continuity_model);
}

TEST_P(EvaluationTest, timeinitjacobian2D)
{
    const Kokkos::Array<double, 3> time_values = {0.5, 1.0, 0.2};
    const Kokkos::Array<double, 2> final_values = {2.0, 3.0};
    const Kokkos::Array<double, 2> init_values = {1.0, 1.2};
    const Kokkos::Array<double, 2> exp_fields = {1.0, 1.2};
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 2>(
        time_values, final_values, init_values, exp_fields, continuity_model);
}

//---------------------------------------------------------------------------//
// 2-D: time transient dirichlet - time_init < time <
// time_final
TEST_P(EvaluationTest, timeintermediateresidual2D)
{
    const Kokkos::Array<double, 3> time_values = {1.0, 2.0, 1.5};
    const Kokkos::Array<double, 2> final_values = {2.0, 3.0};
    const Kokkos::Array<double, 2> init_values = {1.0, 2.0};
    const Kokkos::Array<double, 2> exp_fields = {1.5, 2.5};
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 2>(
        time_values, final_values, init_values, exp_fields, continuity_model);
}

TEST_P(EvaluationTest, timeintermediatejacobian2D)
{
    const Kokkos::Array<double, 3> time_values = {1.0, 2.0, 1.5};
    const Kokkos::Array<double, 2> final_values = {2.0, 3.0};
    const Kokkos::Array<double, 2> init_values = {1.0, 2.0};
    const Kokkos::Array<double, 2> exp_fields = {1.5, 2.5};
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 2>(
        time_values, final_values, init_values, exp_fields, continuity_model);
}

//---------------------------------------------------------------------------//
// 3-D: time transient dirichlet - time_init < time <
// time_final
TEST_P(EvaluationTest, timeintermediateresidual3D)
{
    const Kokkos::Array<double, 3> time_values = {1.0, 2.0, 1.5};
    const Kokkos::Array<double, 3> final_values = {2.0, 3.0, 4.0};
    const Kokkos::Array<double, 3> init_values = {1.0, 2.0, 3.0};
    const Kokkos::Array<double, 3> exp_fields = {1.5, 2.5, 3.5};
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Residual, 3>(
        time_values, final_values, init_values, exp_fields, continuity_model);
}

TEST_P(EvaluationTest, timeintermediatejacobian3D)
{
    const Kokkos::Array<double, 3> time_values = {1.0, 2.0, 1.5};
    const Kokkos::Array<double, 3> final_values = {2.0, 3.0, 4.0};
    const Kokkos::Array<double, 3> init_values = {1.0, 2.0, 3.0};
    const Kokkos::Array<double, 3> exp_fields = {1.5, 2.5, 3.5};
    ContinuityModel continuity_model;
    continuity_model = GetParam();
    testEval<panzer::Traits::Jacobian, 3>(
        time_values, final_values, init_values, exp_fields, continuity_model);
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
