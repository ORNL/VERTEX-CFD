#include <VertexCFD_EvaluatorTestHarness.hpp>

#include "boundary_conditions/VertexCFD_BoundaryState_MethodManufacturedSolution.hpp"
#include "utils/VertexCFD_Utils_Constants.hpp"

#include <Panzer_Dimension.hpp>

#include <Phalanx_Evaluator_Derived.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <stdexcept>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//¬
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval()
{
    // Test fixture
    constexpr int num_space_dim = NumSpaceDim;
    constexpr int num_coeff = num_space_dim * 2 + 2;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Set non-trivial coordinates for the quadrature point
    Kokkos::Array<double, num_space_dim> x;
    x[0] = 0.1;
    x[1] = 0.2;
    auto ip_coord_view
        = test_fixture.int_values->ip_coordinates.get_static_view();
    auto ip_coord_mirror = Kokkos::create_mirror(ip_coord_view);
    ip_coord_mirror(0, 0, 0) = x[0];
    ip_coord_mirror(0, 0, 1) = x[1];
    if (num_space_dim == 3)
    {
        x[2] = 0.3;
        ip_coord_mirror(0, 0, 2) = x[2];
    }
    Kokkos::deep_copy(ip_coord_view, ip_coord_mirror);

    // Create mms evaluator.
    auto mms_eval = Teuchos::rcp(
        new BoundaryCondition::
            MethodManufacturedSolution<EvalType, panzer::Traits, num_space_dim>(
                *test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(mms_eval);

    // Function used for the MMS
    using std::cos;
    using std::sin;
    using VertexCFD::Constants::pi;

    // function = B + A * sin(2*pi*f_x*(x-phi_x)) * sin(2*pi*f_y*(x-phi_y))
    //                  * sin(2*pi*f_z*(z-phi_z)) for 3D
    auto set_function = [=](const Kokkos::Array<double, num_coeff> coeff,
                            const Kokkos::Array<double, num_space_dim> x) {
        const double val = coeff[0]
                           * sin(2.0 * pi * coeff[2] * (x[0] - coeff[3]))
                           * sin(2.0 * pi * coeff[4] * (x[1] - coeff[5]));
        const double return_val
            = num_space_dim == 2
                  ? val + coeff[1]
                  : val * sin(2.0 * pi * coeff[6] * (x[2] - coeff[7]))
                        + coeff[1];
        return return_val;
    };

    // function = A * 2*pi*f_x* cos(2*pi*f_x*(x-phi_x)) *
    // sin(2*pi*f_y*(y-phi_y))
    //              * sin(2*pi*f_z*(z-phi_z)) for 3D
    auto set_gradX_function = [=](const Kokkos::Array<double, num_coeff> coeff,
                                  const Kokkos::Array<double, num_space_dim> x) {
        const double val = coeff[0] * 2.0 * pi * coeff[2]
                           * cos(2.0 * pi * coeff[2] * (x[0] - coeff[3]))
                           * sin(2.0 * pi * coeff[4] * (x[1] - coeff[5]));
        const double return_val
            = num_space_dim == 2
                  ? val
                  : val * sin(2.0 * pi * coeff[6] * (x[2] - coeff[7]));

        return return_val;
    };

    // function = A * sin(2*pi*f_x*(x-phi_x)) * 2*pi*f_y*
    // cos(2*pi*f_y*(y-phi_y))
    //              * sin(2*pi*f_z*(z-phi_z)) for 3D
    auto set_gradY_function = [=](const Kokkos::Array<double, num_coeff> coeff,
                                  const Kokkos::Array<double, num_space_dim> x) {
        const double val = coeff[0] * 2.0 * pi * coeff[4]
                           * sin(2.0 * pi * coeff[2] * (x[0] - coeff[3]))
                           * cos(2.0 * pi * coeff[4] * (x[1] - coeff[5]));
        const double return_val
            = num_space_dim == 2
                  ? val
                  : val * sin(2.0 * pi * coeff[6] * (x[2] - coeff[7]));

        return return_val;
    };

    // function = A * sin(2*pi*f_x*(x-phi_x)) * sin(2*pi*f_y*(y-phi_y))
    //              * 2*pi*f_Z * cos(2*pi*f_z*(z-phi_z)) for 3D
    auto set_gradZ_function = [=](const Kokkos::Array<double, num_coeff> coeff,
                                  const Kokkos::Array<double, num_space_dim> x) {
        return coeff[0] * 2.0 * pi * coeff[6]
               * sin(2.0 * pi * coeff[2] * (x[0] - coeff[3]))
               * sin(2.0 * pi * coeff[4] * (x[1] - coeff[5]))
               * cos(2.0 * pi * coeff[6] * (x[2] - coeff[7]));
    };

    // Coefficients to be used with the above function to compute reference
    // values for each primitive variable and gradients. They are hard coded in
    // the source code
    Kokkos::Array<double, num_coeff> phi_coeff;
    Kokkos::Array<Kokkos::Array<double, num_coeff>, num_space_dim> vel_coeff;
    Kokkos::Array<double, num_coeff> T_coeff;

    phi_coeff[0] = 0.0125;
    phi_coeff[1] = 1.0;
    phi_coeff[2] = 0.25;
    phi_coeff[3] = 0.5;
    phi_coeff[4] = 0.125;
    phi_coeff[5] = 0.0;

    vel_coeff[0][0] = 0.0125;
    vel_coeff[0][1] = 0.08;
    vel_coeff[0][2] = 0.125;
    vel_coeff[0][3] = 0.0;
    vel_coeff[0][4] = 0.125;
    vel_coeff[0][5] = 0.25;

    vel_coeff[1][0] = 0.0375;
    vel_coeff[1][1] = 1.125;
    vel_coeff[1][2] = 0.25;
    vel_coeff[1][3] = 0.0;
    vel_coeff[1][4] = 0.375;
    vel_coeff[1][5] = 0.5;

    T_coeff[0] = 0.0625;
    T_coeff[1] = 1.0;
    T_coeff[2] = 0.375;
    T_coeff[3] = 0.25;
    T_coeff[4] = 0.25;
    T_coeff[5] = 0.5;

    if (num_space_dim == 3)
    {
        phi_coeff[6] = 0.375;
        phi_coeff[7] = 1.0;

        vel_coeff[0][6] = 0.25;
        vel_coeff[0][7] = 0.0;

        vel_coeff[1][6] = 0.25;
        vel_coeff[1][7] = 0.5;

        vel_coeff[2][0] = 0.025;
        vel_coeff[2][1] = 0.0;
        vel_coeff[2][2] = 0.125;
        vel_coeff[2][3] = 1.0;
        vel_coeff[2][4] = 0.25;
        vel_coeff[2][5] = 0.25;
        vel_coeff[2][6] = 0.25;
        vel_coeff[2][7] = 0.25;

        T_coeff[6] = 0.125;
        T_coeff[7] = 0.5;
    }

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(
        mms_eval->_boundary_lagrange_pressure);
    test_fixture.registerTestField<EvalType>(mms_eval->_boundary_temperature);

    test_fixture.registerTestField<EvalType>(
        mms_eval->_boundary_grad_lagrange_pressure);
    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        test_fixture.registerTestField<EvalType>(
            mms_eval->_boundary_velocity[dim]);
        test_fixture.registerTestField<EvalType>(
            mms_eval->_boundary_grad_velocity[dim]);
    }
    test_fixture.registerTestField<EvalType>(
        mms_eval->_boundary_grad_temperature);

    // Evaluate MMS BC.
    test_fixture.evaluate<EvalType>();

    // Check the BC value for all scalars.
    auto boundary_lagrange_pressure_result
        = test_fixture.getTestFieldData<EvalType>(
            mms_eval->_boundary_lagrange_pressure);
    auto boundary_velocity_0_result = test_fixture.getTestFieldData<EvalType>(
        mms_eval->_boundary_velocity[0]);
    auto boundary_velocity_1_result = test_fixture.getTestFieldData<EvalType>(
        mms_eval->_boundary_velocity[1]);
    auto boundary_temperature_result = test_fixture.getTestFieldData<EvalType>(
        mms_eval->_boundary_temperature);

    auto boundary_grad_lagrange_pressure_result
        = test_fixture.getTestFieldData<EvalType>(
            mms_eval->_boundary_grad_lagrange_pressure);
    auto boundary_grad_velocity_0_result
        = test_fixture.getTestFieldData<EvalType>(
            mms_eval->_boundary_grad_velocity[0]);
    auto boundary_grad_velocity_1_result
        = test_fixture.getTestFieldData<EvalType>(
            mms_eval->_boundary_grad_velocity[1]);
    auto boundary_grad_temperature_result
        = test_fixture.getTestFieldData<EvalType>(
            mms_eval->_boundary_grad_temperature);

    // Compute reference values using ideal gas equation of state for pressure
    // and specific total energy
    const double phi_ref = set_function(phi_coeff, x);
    const double u_ref = set_function(vel_coeff[0], x);
    const double v_ref = set_function(vel_coeff[1], x);
    const double T_ref = set_function(T_coeff, x);
    Kokkos::Array<double, num_space_dim> vel_ref;
    for (int dim = 0; dim < num_space_dim; ++dim)
        vel_ref[dim] = set_function(vel_coeff[dim], x);

    const double phi_gradX_ref = set_gradX_function(phi_coeff, x);
    const double phi_gradY_ref = set_gradY_function(phi_coeff, x);
    const double u_gradX_ref = set_gradX_function(vel_coeff[0], x);
    const double u_gradY_ref = set_gradY_function(vel_coeff[0], x);
    const double v_gradX_ref = set_gradX_function(vel_coeff[1], x);
    const double v_gradY_ref = set_gradY_function(vel_coeff[1], x);
    const double T_gradX_ref = set_gradX_function(T_coeff, x);
    const double T_gradY_ref = set_gradY_function(T_coeff, x);

    const int num_point = boundary_lagrange_pressure_result.extent(1);

    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(phi_ref,
                         fieldValue(boundary_lagrange_pressure_result, 0, qp));
        EXPECT_DOUBLE_EQ(u_ref, fieldValue(boundary_velocity_0_result, 0, qp));
        EXPECT_DOUBLE_EQ(v_ref, fieldValue(boundary_velocity_1_result, 0, qp));
        EXPECT_DOUBLE_EQ(T_ref, fieldValue(boundary_temperature_result, 0, qp));

        EXPECT_DOUBLE_EQ(
            phi_gradX_ref,
            fieldValue(boundary_grad_lagrange_pressure_result, 0, qp, 0));
        EXPECT_DOUBLE_EQ(
            phi_gradY_ref,
            fieldValue(boundary_grad_lagrange_pressure_result, 0, qp, 1));
        EXPECT_DOUBLE_EQ(u_gradX_ref,
                         fieldValue(boundary_grad_velocity_0_result, 0, qp, 0));
        EXPECT_DOUBLE_EQ(u_gradY_ref,
                         fieldValue(boundary_grad_velocity_0_result, 0, qp, 1));
        EXPECT_DOUBLE_EQ(v_gradX_ref,
                         fieldValue(boundary_grad_velocity_1_result, 0, qp, 0));
        EXPECT_DOUBLE_EQ(v_gradY_ref,
                         fieldValue(boundary_grad_velocity_1_result, 0, qp, 1));
        EXPECT_DOUBLE_EQ(
            T_gradX_ref,
            fieldValue(boundary_grad_temperature_result, 0, qp, 0));
        EXPECT_DOUBLE_EQ(
            T_gradY_ref,
            fieldValue(boundary_grad_temperature_result, 0, qp, 1));
    }

    if (num_space_dim == 3)
    {
        auto boundary_velocity_2_result
            = test_fixture.getTestFieldData<EvalType>(
                mms_eval->_boundary_velocity[2]);
        auto boundary_grad_velocity_2_result
            = test_fixture.getTestFieldData<EvalType>(
                mms_eval->_boundary_grad_velocity[2]);

        const double w_ref = set_function(vel_coeff[2], x);
        const double phi_gradZ_ref = set_gradZ_function(phi_coeff, x);
        const double u_gradZ_ref = set_gradZ_function(vel_coeff[0], x);
        const double v_gradZ_ref = set_gradZ_function(vel_coeff[1], x);
        const double w_gradX_ref = set_gradX_function(vel_coeff[2], x);
        const double w_gradY_ref = set_gradY_function(vel_coeff[2], x);
        const double w_gradZ_ref = set_gradZ_function(vel_coeff[2], x);
        const double T_gradZ_ref = set_gradZ_function(T_coeff, x);

        for (int qp = 0; qp < num_point; ++qp)
        {
            EXPECT_DOUBLE_EQ(w_ref,
                             fieldValue(boundary_velocity_2_result, 0, qp));

            EXPECT_DOUBLE_EQ(
                phi_gradZ_ref,
                fieldValue(boundary_grad_lagrange_pressure_result, 0, qp, 2));
            EXPECT_DOUBLE_EQ(
                u_gradZ_ref,
                fieldValue(boundary_grad_velocity_0_result, 0, qp, 2));
            EXPECT_DOUBLE_EQ(
                v_gradZ_ref,
                fieldValue(boundary_grad_velocity_1_result, 0, qp, 2));
            EXPECT_DOUBLE_EQ(
                w_gradX_ref,
                fieldValue(boundary_grad_velocity_2_result, 0, qp, 0));
            EXPECT_DOUBLE_EQ(
                w_gradY_ref,
                fieldValue(boundary_grad_velocity_2_result, 0, qp, 1));
            EXPECT_DOUBLE_EQ(
                w_gradZ_ref,
                fieldValue(boundary_grad_velocity_2_result, 0, qp, 2));
            EXPECT_DOUBLE_EQ(
                T_gradZ_ref,
                fieldValue(boundary_grad_temperature_result, 0, qp, 2));
        }
    }
}

//---------------------------------------------------------------------------//
// Method of Manufactured Solution
// Residual
TEST(MethodManufacturedSolutionBC2D, residual_mms_test)
{
    testEval<panzer::Traits::Residual, 2>();
}

// Jacobian
TEST(MethodManufacturedSolutionBC2D, jacobian_mms_test)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

TEST(MethodManufacturedSolutionBC3D, residual_mms_test)
{
    testEval<panzer::Traits::Residual, 3>();
}

// Jacobian
TEST(MethodManufacturedSolutionBC3D, jacobian_mms_test)
{
    testEval<panzer::Traits::Jacobian, 3>();
}
//---------------------------------------------------------------------------//
} // end namespace Test
} // end namespace VertexCFD
