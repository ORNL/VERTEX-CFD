#include <VertexCFD_EvaluatorTestHarness.hpp>

#include "full_induction_mhd_solver/initial_conditions/VertexCFD_InitialCondition_MHDVortexProblem.hpp"

#include <gtest/gtest.h>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//¬
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
template<class SubviewType>
auto compute_ref_sol(const Teuchos::ParameterList& params, const SubviewType& x)
{
    using std::exp;
    using std::sqrt;

    constexpr int num_conserve = 5;
    Kokkos::Array<double, num_conserve> ref_solution;

    // Get MHD vortex problem properties
    const auto vel_0 = params.get<Teuchos::Array<double>>("velocity_0");
    const auto xy_0 = params.get<Teuchos::Array<double>>("center_0");

    const double dx = x[0] - xy_0[0];
    const double dy = x[1] - xy_0[1];
    const double r2 = dx * dx + dy * dy;

    // Lagrange pressure
    ref_solution[0] = 1.0 + 0.5 * exp(1.) * (1. - r2 * exp(-r2));
    // Magnetic field
    ref_solution[1] = -exp(0.5 * (1.0 - r2)) * dy;
    ref_solution[2] = exp(0.5 * (1.0 - r2)) * dx;
    // Velocity
    ref_solution[3] = ref_solution[1] + vel_0[0];
    ref_solution[4] = ref_solution[2] + vel_0[1];

    return ref_solution;
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval()
{
    // Test fixture
    constexpr int num_space_dim = NumSpaceDim;
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
    Kokkos::Array<double, num_space_dim> _xy_0{};
    for (int dim = 0; dim < num_space_dim; dim++)
    {
        for (int basis = 0; basis < num_coord; basis++)
        {
            // random coordinate assigned
            x(dim, basis) = (0.5 + dim * 4.5) * (basis + 1);
            basis_coord_mirror(0, basis, dim) = x(dim, basis);
            _xy_0[dim] += x(dim, basis);
        }
    }
    Kokkos::deep_copy(basis_coord_view, basis_coord_mirror);

    // Create the param list to initialize the evaluator
    Teuchos::ParameterList model_params;
    Teuchos::Array<double> xy_0(2);
    Teuchos::Array<double> vel_0(2);
    vel_0[0] = 1.2;
    vel_0[1] = 2.4;

    // center of vortex is center of the cell
    for (int dim = 0; dim < 2; ++dim)
        xy_0[dim] = _xy_0[dim] / std::pow(num_space_dim, 2);

    model_params.set<Teuchos::Array<double>>("velocity_0", vel_0);
    model_params.set<Teuchos::Array<double>>("center_0", xy_0);

    // Create evaluator.
    auto mhd_val = Teuchos::rcp(
        new InitialCondition::
            MHDVortexProblem<EvalType, panzer::Traits, num_space_dim>(
                model_params, *test_fixture.basis_ir_layout->getBasis()));
    test_fixture.registerEvaluator<EvalType>(mhd_val);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(mhd_val->_lagrange_pressure);
    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        test_fixture.registerTestField<EvalType>(mhd_val->_velocity[dim]);
        test_fixture.registerTestField<EvalType>(
            mhd_val->_induced_magnetic_field[dim]);
    }

    // Evaluate IC.
    test_fixture.evaluate<EvalType>();

    // Check the IC value for all scalars.
    const auto lg_pressure_result
        = test_fixture.getTestFieldData<EvalType>(mhd_val->_lagrange_pressure);
    const auto velocity_0_result
        = test_fixture.getTestFieldData<EvalType>(mhd_val->_velocity[0]);
    const auto velocity_1_result
        = test_fixture.getTestFieldData<EvalType>(mhd_val->_velocity[1]);
    const auto magn_field_0_result = test_fixture.getTestFieldData<EvalType>(
        mhd_val->_induced_magnetic_field[0]);
    const auto magn_field_1_result = test_fixture.getTestFieldData<EvalType>(
        mhd_val->_induced_magnetic_field[1]);

    // Assert number of points
    EXPECT_EQ(num_coord, lg_pressure_result.extent(1));
    EXPECT_EQ(num_coord, velocity_0_result.extent(1));
    EXPECT_EQ(num_coord, velocity_1_result.extent(1));
    EXPECT_EQ(num_coord, magn_field_0_result.extent(1));
    EXPECT_EQ(num_coord, magn_field_1_result.extent(1));

    // Loop over number of points and compare against reference values
    const int num_point = lg_pressure_result.extent(1);
    const double tol = 1.0e-15;
    for (int d = 0; d < num_point; ++d)
    {
        const auto x_subview = Kokkos::subview(x, Kokkos::ALL(), d);

        // Compute reference values
        const auto ref_solution = compute_ref_sol(model_params, x_subview);

        // Assert
        EXPECT_NEAR(ref_solution[0], fieldValue(lg_pressure_result, 0, d), tol);
        EXPECT_NEAR(
            ref_solution[1], fieldValue(magn_field_0_result, 0, d), tol);
        EXPECT_NEAR(
            ref_solution[2], fieldValue(magn_field_1_result, 0, d), tol);
        EXPECT_DOUBLE_EQ(ref_solution[3], fieldValue(velocity_0_result, 0, d));
        EXPECT_DOUBLE_EQ(ref_solution[4], fieldValue(velocity_1_result, 0, d));

        if (num_space_dim == 3)
        {
            const auto velocity_2_result
                = test_fixture.getTestFieldData<EvalType>(
                    mhd_val->_velocity[2]);
            EXPECT_EQ(num_coord, velocity_2_result.extent(1));
            EXPECT_DOUBLE_EQ(0.0, fieldValue(velocity_2_result, 0, d));

            const auto magn_field_2_result
                = test_fixture.getTestFieldData<EvalType>(
                    mhd_val->_induced_magnetic_field[2]);
            EXPECT_EQ(num_coord, magn_field_2_result.extent(1));
            EXPECT_DOUBLE_EQ(0.0, fieldValue(magn_field_2_result, 0, d));
        }
    }
}

//---------------------------------------------------------------------------//
// MHD vortex probem 2D
// Residual
TEST(MHDVortexProblemIC2D, residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

// Jacobian
TEST(MHDVortexProblemIC2D, jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

// MHD vortex probem 3D
// Residual
TEST(MHDVortexProblemIC3D, residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

// Jacobian
TEST(MHDVortexProblemIC3D, jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}
//---------------------------------------------------------------------------//
} // end namespace Test
} // end namespace VertexCFD
