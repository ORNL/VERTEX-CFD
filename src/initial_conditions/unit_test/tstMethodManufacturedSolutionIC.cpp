#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "initial_conditions/VertexCFD_InitialCondition_MethodManufacturedSolution.hpp"
#include "utils/VertexCFD_Utils_Constants.hpp"

#include <gtest/gtest.h>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//¬
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
// function = B + A * sin(2*pi*f_x*(x-phi_x)) *
// sin(2*pi*f_y*(x-phi_y))
//                  * sin(2*pi*f_z*(z-phi_z)) for 3D
// FIXME: warning: lambda templates are only available with ‘-std=c++20’ or
// ‘-std=gnu++20’
//
template<int num_space_dim, int num_coeff, class SubviewType>
double
set_function(const Kokkos::Array<double, num_coeff> coeff, const SubviewType& x)
{
    using std::sin;
    using VertexCFD::Constants::pi;

    double return_val = coeff[0];
    for (int i = 0; i < num_space_dim; ++i)
        return_val *= sin(2.0 * pi * coeff[2 * (i + 1)]
                          * (x[i] - coeff[2 * (i + 1) + 1]));
    return_val += coeff[1];
    return return_val;
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval()
{
    // Test fixture
    const int num_space_dim = NumSpaceDim;
    const int num_coeff = 2 * (num_space_dim + 1);
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const int num_coord = test_fixture.cell_topo->getNodeCount();

    // Set non-trivial coordinates for the degree of freedom
    const Kokkos::View<double**, Kokkos::HostSpace> x(
        "coordinate", num_space_dim, num_coord);
    auto basis_coord_view
        = test_fixture.workset->bases[0]->basis_coordinates.get_static_view();
    auto basis_coord_mirror = Kokkos::create_mirror(basis_coord_view);
    for (int dim = 0; dim < num_space_dim; dim++)
    {
        for (int basis = 0; basis < num_coord; basis++)
        {
            // random coordinate assigned
            x(dim, basis) = (0.5 + dim * 4.5) * (basis + 1);
            basis_coord_mirror(0, basis, dim) = x(dim, basis);
        }
    }
    Kokkos::deep_copy(basis_coord_view, basis_coord_mirror);

    // Create mms evaluator.
    auto mms_eval = Teuchos::rcp(
        new InitialCondition::
            MethodManufacturedSolution<EvalType, panzer::Traits, num_space_dim>(
                *test_fixture.basis_ir_layout->getBasis()));
    test_fixture.registerEvaluator<EvalType>(mms_eval);

    // Coefficients to be used with the above function to compute reference
    // values for each primitive variable. They are hard coded in the source
    // code
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
    test_fixture.registerTestField<EvalType>(mms_eval->_lagrange_pressure);
    for (int dim = 0; dim < num_space_dim; ++dim)
        test_fixture.registerTestField<EvalType>(mms_eval->_velocity[dim]);
    test_fixture.registerTestField<EvalType>(mms_eval->_temperature);

    // Evaluate MMS IC.
    test_fixture.evaluate<EvalType>();

    // Check the IC value for all scalars.
    const auto phi_result = test_fixture.getTestFieldData<EvalType>(
        mms_eval->_lagrange_pressure);
    const auto velocity_0_result
        = test_fixture.getTestFieldData<EvalType>(mms_eval->_velocity[0]);
    const auto velocity_1_result
        = test_fixture.getTestFieldData<EvalType>(mms_eval->_velocity[1]);
    const auto temperature_result
        = test_fixture.getTestFieldData<EvalType>(mms_eval->_temperature);

    // Assert number of points
    EXPECT_EQ(num_coord, phi_result.extent(1));
    EXPECT_EQ(num_coord, velocity_0_result.extent(1));
    EXPECT_EQ(num_coord, velocity_1_result.extent(1));
    EXPECT_EQ(num_coord, temperature_result.extent(1));

    // Loop over number of points and compare against reference values
    const int num_point = phi_result.extent(1);
    for (int d = 0; d < num_point; ++d)
    {
        auto x_subview = Kokkos::subview(x, Kokkos::ALL(), d);

        // Compute reference values
        const double phi_ref
            = set_function<num_space_dim, num_coeff>(phi_coeff, x_subview);
        const double u_ref
            = set_function<num_space_dim, num_coeff>(vel_coeff[0], x_subview);
        const double v_ref
            = set_function<num_space_dim, num_coeff>(vel_coeff[1], x_subview);
        const double T_ref
            = set_function<num_space_dim, num_coeff>(T_coeff, x_subview);

        // Assert
        EXPECT_DOUBLE_EQ(phi_ref, fieldValue(phi_result, 0, d));
        EXPECT_DOUBLE_EQ(u_ref, fieldValue(velocity_0_result, 0, d));
        EXPECT_DOUBLE_EQ(v_ref, fieldValue(velocity_1_result, 0, d));
        EXPECT_DOUBLE_EQ(T_ref, fieldValue(temperature_result, 0, d));
    }

    if (num_space_dim == 3)
    {
        const auto velocity_2_result
            = test_fixture.getTestFieldData<EvalType>(mms_eval->_velocity[2]);
        EXPECT_EQ(num_coord, velocity_2_result.extent(1));

        for (int d = 0; d < num_point; ++d)
        {
            auto x_subview = Kokkos::subview(x, Kokkos::ALL(), d);
            const double w_ref = set_function<num_space_dim, num_coeff>(
                vel_coeff[2], x_subview);

            // Assert
            EXPECT_DOUBLE_EQ(w_ref, fieldValue(velocity_2_result, 0, d));
        }
    }
}

//---------------------------------------------------------------------------//
// Method of Manufactured Solution IC
// Residual
TEST(MethodManufacturedSolutionIC2D, residual_mms_ic_test)
{
    testEval<panzer::Traits::Residual, 2>();
}

// Jacobian
TEST(MethodManufacturedSolutionIC2D, jacobian_mms_ic_test)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

TEST(MethodManufacturedSolutionIC3D, residual_mms_ic_test)
{
    testEval<panzer::Traits::Residual, 3>();
}

// Jacobian
TEST(MethodManufacturedSolutionIC3D, jacobian_mms_ic_test)
{
    testEval<panzer::Traits::Jacobian, 3>();
}
//---------------------------------------------------------------------------//
} // end namespace Test
} // end namespace VertexCFD
