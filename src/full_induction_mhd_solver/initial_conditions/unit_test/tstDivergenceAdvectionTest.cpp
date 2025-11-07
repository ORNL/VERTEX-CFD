#include <VertexCFD_EvaluatorTestHarness.hpp>

#include "full_induction_mhd_solver/initial_conditions/VertexCFD_InitialCondition_DivergenceAdvectionTest.hpp"

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
template<class SubviewType>
auto compute_ref_sol(const Teuchos::ParameterList& params, const SubviewType& x)
{
    using Constants::pi;
    using std::pow;
    using std::sqrt;

    constexpr int num_conserve = 6;
    Kokkos::Array<double, num_conserve> ref_solution;

    const auto psi = params.isType<double>("Lagrange Pressure")
                         ? params.get<double>("Lagrange Pressure")
                         : 6.0;

    const auto r0 = params.isType<double>("Divergence Bubble Radius")
                        ? params.get<double>("Divergence Bubble Radius")
                        : 1.0 / sqrt(8.0);

    const auto x_c
        = params.isType<Teuchos::Array<double>>("Divergence Bubble Center")
              ? params.get<Teuchos::Array<double>>("Divergence Bubble Center")
              : Teuchos::Array<double>(3, 0);

    const auto vel = params.isType<Teuchos::Array<double>>("Velocity")
                         ? params.get<Teuchos::Array<double>>("Velocity")
                         : Teuchos::Array<double>({1.0, 1.0, 0.0});

    const double dx = x[0] - x_c[0];
    const double dy = x[1] - x_c[1];
    const double r = sqrt(dx * dx + dy * dy);

    // Lagrange pressure
    ref_solution[0] = psi;
    // Magnetic field
    if (r < r0)
    {
        ref_solution[1] = (pow(r / r0, 8) - 2.0 * pow(r / r0, 4) + 1)
                          / sqrt(4.0 * pi);
    }
    else
    {
        ref_solution[1] = 0.0;
    }
    ref_solution[2] = 0.0;
    // Velocity
    ref_solution[3] = vel[0];
    ref_solution[4] = vel[1];
    ref_solution[5] = vel[2];

    return ref_solution;
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval(const bool set_input_params = false)
{
    // Test fixture
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const double r0 = 1.5;
    const double nanval = std::numeric_limits<double>::signaling_NaN();
    const double x_z = num_space_dim == 2 ? nanval : 0.5;
    const Teuchos::Array<double> x0({0.3, 0.4, x_z});
    Teuchos::ParameterList model_params;
    if (set_input_params)
    {
        model_params.set("Lagrange Pressure", 2.3);
        model_params.set("Divergence Bubble Radius", r0);
        model_params.set("Divergence Bubble Center", x0);
        const double v_z = num_space_dim == 2 ? nanval : 0.325;
        model_params.set("Velocity",
                         Teuchos::Array<double>({0.2, -0.125, v_z}));
    }
    Teuchos::ParameterList mhd_params;
    mhd_params.set("Divergence Advection Test", model_params);

    const int num_coord = test_fixture.cell_topo->getNodeCount();

    // Set non-trivial coordinates for the degree of freedom
    const Kokkos::View<double**, Kokkos::HostSpace> x(
        "coordinate", num_space_dim, num_coord);
    auto basis_coord_view
        = test_fixture.workset->bases[0]->basis_coordinates.get_static_view();
    auto basis_coord_mirror = Kokkos::create_mirror(basis_coord_view);
    for (int basis = 0; basis < num_coord; basis++)
    {
        // set coordinates inside and outside of bubble radius
        x(0, basis) = r0 / (num_coord - 2) * basis * std::cos(basis) + x0[0];
        x(1, basis) = r0 / (num_coord - 2) * basis * std::sin(basis) + x0[1];
        if (num_space_dim > 2)
            x(2, basis) = r0 * (basis);
        for (int dim = 0; dim < num_space_dim; ++dim)
        {
            basis_coord_mirror(0, basis, dim) = x(dim, basis);
        }
    }
    Kokkos::deep_copy(basis_coord_view, basis_coord_mirror);

    // Create evaluator.
    auto mhd_val = Teuchos::rcp(
        new InitialCondition::
            DivergenceAdvectionTest<EvalType, panzer::Traits, num_space_dim>(
                mhd_params, *test_fixture.basis_ir_layout->getBasis()));
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
    for (int qp = 0; qp < num_point; ++qp)
    {
        const auto x_subview = Kokkos::subview(x, Kokkos::ALL(), qp);

        // Compute reference values
        const auto ref_solution = compute_ref_sol(model_params, x_subview);

        // Assert
        EXPECT_NEAR(
            ref_solution[0], fieldValue(lg_pressure_result, 0, qp), tol);
        EXPECT_NEAR(
            ref_solution[1], fieldValue(magn_field_0_result, 0, qp), tol);
        EXPECT_NEAR(
            ref_solution[2], fieldValue(magn_field_1_result, 0, qp), tol);
        EXPECT_DOUBLE_EQ(ref_solution[3], fieldValue(velocity_0_result, 0, qp));
        EXPECT_DOUBLE_EQ(ref_solution[4], fieldValue(velocity_1_result, 0, qp));

        if (num_space_dim == 3)
        {
            const auto velocity_2_result
                = test_fixture.getTestFieldData<EvalType>(
                    mhd_val->_velocity[2]);
            EXPECT_EQ(num_coord, velocity_2_result.extent(1));
            EXPECT_DOUBLE_EQ(ref_solution[5],
                             fieldValue(velocity_2_result, 0, qp));

            const auto magn_field_2_result
                = test_fixture.getTestFieldData<EvalType>(
                    mhd_val->_induced_magnetic_field[2]);
            EXPECT_EQ(num_coord, magn_field_2_result.extent(1));
            EXPECT_DOUBLE_EQ(0.0, fieldValue(magn_field_2_result, 0, qp));
        }
    }
}

//---------------------------------------------------------------------------//
// Divergence Advection Test 2D, default values
// Residual
TEST(DivergenceAdvectionTestDefaultIC2D, residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

// Jacobian
TEST(DivergenceAdvectionTestDefaultIC2D, jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

// Divergence Advection Test 2D, input values
// Residual
TEST(DivergenceAdvectionTestInputsIC2D, residual)
{
    testEval<panzer::Traits::Residual, 2>(true);
}

// Jacobian
TEST(DivergenceAdvectionTestInputsIC2D, jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(true);
}

// Divergence Advection Test 3D, default values
// Residual
TEST(DivergenceAdvectionTestDefaultIC3D, residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

// Jacobian
TEST(DivergenceAdvectionTestDefaultIC3D, jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

// Divergence Advection Test 3D, input values
// Residual
TEST(DivergenceAdvectionTestInputsIC3D, residual)
{
    testEval<panzer::Traits::Residual, 3>(true);
}

// Jacobian
TEST(DivergenceAdvectionTestInputsIC3D, jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(true);
}
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
} // end namespace Test
} // end namespace VertexCFD
