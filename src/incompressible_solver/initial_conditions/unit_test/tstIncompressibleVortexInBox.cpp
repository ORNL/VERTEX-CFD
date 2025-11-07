#include <VertexCFD_EvaluatorTestHarness.hpp>

#include "incompressible_solver/initial_conditions/VertexCFD_InitialCondition_IncompressibleVortexInBox.hpp"

#include <Teuchos_ParameterList.hpp>

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
template<class EvalType>
void testEval()
{
    const int num_space_dim = 2;
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
    auto eval = Teuchos::rcp(
        new InitialCondition::IncompressibleVortexInBox<EvalType, panzer::Traits>(
            *test_fixture.basis_ir_layout->getBasis()));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_lagrange_pressure);
    test_fixture.registerTestField<EvalType>(eval->_velocity_0);
    test_fixture.registerTestField<EvalType>(eval->_velocity_1);

    test_fixture.evaluate<EvalType>();

    const auto phi
        = test_fixture.getTestFieldData<EvalType>(eval->_lagrange_pressure);
    const auto u = test_fixture.getTestFieldData<EvalType>(eval->_velocity_0);
    const auto v = test_fixture.getTestFieldData<EvalType>(eval->_velocity_1);

    // Check number of degree of freedoms
    const int num_dofs = 4;
    EXPECT_EQ(num_dofs, phi.extent(1));
    EXPECT_EQ(num_dofs, u.extent(1));
    EXPECT_EQ(num_dofs, v.extent(1));

    // Reference values
    const double phi_ref = 0.0;
    const double u_ref[4] = {0.0636906811868036,
                             -0.019798055040567034,
                             -0.019798055040567034,
                             0.06369068118680365};
    const double v_ref[4] = {-0.019798055040567034,
                             -0.0636906811868036,
                             0.2453263131881438,
                             0.7892189393343801};

    // Check values
    for (int n = 0; n < num_dofs; ++n)
    {
        EXPECT_DOUBLE_EQ(phi_ref, fieldValue(phi, 0, n));
        EXPECT_FLOAT_EQ(u_ref[n], fieldValue(u, 0, n));
        EXPECT_FLOAT_EQ(v_ref[n], fieldValue(v, 0, n));
    }
}

//---------------------------------------------------------------------------//
TEST(IncompressibleVortexInBox, residual_test)
{
    testEval<panzer::Traits::Residual>();
}

//---------------------------------------------------------------------------//
TEST(IncompressibleVortexInBox, jacobian_test)
{
    testEval<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//

} // namespace Test
} // namespace VertexCFD
