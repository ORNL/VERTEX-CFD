#include <VertexCFD_EvaluatorTestHarness.hpp>

#include "incompressible_solver/initial_conditions/VertexCFD_InitialCondition_IncompressibleTaylorGreenVortex.hpp"

#include <Teuchos_ParameterList.hpp>

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval()
{
    const int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Set non-trivial coordinates for the degree of freedom
    const int num_coord = test_fixture.cell_topo->getNodeCount();
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
            x(dim, basis) = 0.1 * (dim + 1) * (basis + 1) - 0.25;
            basis_coord_mirror(0, basis, dim) = x(dim, basis);
        }
    }

    Kokkos::deep_copy(basis_coord_view, basis_coord_mirror);

    // Register and evaluate fields
    const auto eval = Teuchos::rcp(
        new InitialCondition::IncompressibleTaylorGreenVortex<EvalType,
                                                              panzer::Traits,
                                                              num_space_dim>(
            *test_fixture.basis_ir_layout->getBasis()));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_lagrange_pressure);
    for (int dim = 0; dim < num_space_dim; ++dim)
        test_fixture.registerTestField<EvalType>(eval->_velocity[dim]);

    test_fixture.evaluate<EvalType>();

    const auto phi
        = test_fixture.getTestFieldData<EvalType>(eval->_lagrange_pressure);
    const auto u = test_fixture.getTestFieldData<EvalType>(eval->_velocity[0]);
    const auto v = test_fixture.getTestFieldData<EvalType>(eval->_velocity[1]);

    // Check number of degree of freedoms
    const int num_dofs = num_space_dim == 3 ? 8 : 4;
    EXPECT_EQ(num_dofs, phi.extent(1));
    EXPECT_EQ(num_dofs, u.extent(1));
    EXPECT_EQ(num_dofs, v.extent(1));

    // Reference values
    const double phi_ref_2d[4] = {-0.487585163600908,
                                  -0.487585163600908,
                                  -0.4399615881406286,
                                  -0.3522331526377958};
    const double u_ref_2d[4] = {-0.04941795707411653,
                                0.14925137372094469,
                                0.34246927448499503,
                                0.5168180147731708};
    const double v_ref_2d[4] = {0.14925137372094469,
                                0.04941795707411653,
                                -0.04694906782365546,
                                -0.12739967246452027};

    const double phi_ref_3d[8] = {0.06153886762615066,
                                  0.11020304081810303,
                                  0.10861329195646167,
                                  0.0639243560275087,
                                  0.012493863917202034,
                                  -0.010431066422232835,
                                  0.0023302078434405773,
                                  0.02815077122950177};
    const double u_ref_3d[8] = {-0.04935619749596115,
                                0.1402026678284418,
                                0.2726342409183541,
                                0.300624299523849,
                                0.20825410612100023,
                                0.015889334774286345,
                                -0.2265064237080141,
                                -0.45530748103658736};
    const double v_ref_3d[8] = {0.14906484836809994,
                                0.04642188040008816,
                                -0.03737539225139289,
                                -0.07410623507584901,
                                -0.057080512695698894,
                                -0.004147691769724388,
                                0.048966312273462,
                                0.06265712580377152};
    const double w_ref_3d[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    const double* phi_ref = num_space_dim == 3 ? phi_ref_3d : phi_ref_2d;
    const double* u_ref = num_space_dim == 3 ? u_ref_3d : u_ref_2d;
    const double* v_ref = num_space_dim == 3 ? v_ref_3d : v_ref_2d;

    // Check values
    for (int n = 0; n < num_dofs; ++n)
    {
        EXPECT_FLOAT_EQ(phi_ref[n], fieldValue(phi, 0, n));
        EXPECT_FLOAT_EQ(u_ref[n], fieldValue(u, 0, n));
        EXPECT_FLOAT_EQ(v_ref[n], fieldValue(v, 0, n));

        if (num_space_dim == 3)
        {
            const auto w
                = test_fixture.getTestFieldData<EvalType>(eval->_velocity[2]);
            EXPECT_EQ(num_dofs, w.extent(1));
            EXPECT_FLOAT_EQ(w_ref_3d[n], fieldValue(w, 0, n));
        }
    }
}

//---------------------------------------------------------------------------//
TEST(IncompressibleTaylorGreenVortex2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

//---------------------------------------------------------------------------//
TEST(IncompressibleTaylorGreenVortex2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//---------------------------------------------------------------------------//
TEST(IncompressibleTaylorGreenVortex3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

//---------------------------------------------------------------------------//
TEST(IncompressibleTaylorGreenVortex3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//---------------------------------------------------------------------------//

} // namespace Test
} // namespace VertexCFD
