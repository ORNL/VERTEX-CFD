#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "incompressible_solver/initial_conditions/VertexCFD_InitialCondition_IncompressibleTurbulentChannel.hpp"

#include <Teuchos_ParameterList.hpp>

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
template<class EvalType>
void testEval(const bool build_temp_equ)
{
    const int num_space_dim = 3;
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
    for (int basis = 0; basis < num_coord; basis++)
    {
        // Arbitrary coordinate assigned
        basis_coord_mirror(0, basis, 0) = 2.0;
        basis_coord_mirror(0, basis, 1) = 3.0;
        basis_coord_mirror(0, basis, 2) = -4.0;
    }

    Kokkos::deep_copy(basis_coord_view, basis_coord_mirror);

    // Build IC parameter list
    Teuchos::ParameterList ic_params;
    ic_params.set("Kinematic Viscosity", 1.5e-5);
    ic_params.set("Friction Reynolds Number", 125.0);
    ic_params.set("Half Width", 5.0);
    ic_params.set("L_x", 13.0);
    ic_params.set("L_z", 15.0);
    ic_params.set("Number of Modes", 5);
    ic_params.set("Add Random Perturbations", false);
    if (build_temp_equ)
        ic_params.set("Temperature", 4.0);

    // Register and evaluate fields
    auto eval = Teuchos::rcp(
        new InitialCondition::IncompressibleTurbulentChannel<EvalType,
                                                             panzer::Traits,
                                                             num_space_dim>(
            ic_params,
            build_temp_equ,
            *test_fixture.basis_ir_layout->getBasis()));
    test_fixture.registerEvaluator<EvalType>(eval);

    test_fixture.registerTestField<EvalType>(eval->_lagrange_pressure);
    for (int dim = 0; dim < num_space_dim; ++dim)
        test_fixture.registerTestField<EvalType>(eval->_velocity[dim]);
    if (build_temp_equ)
        test_fixture.registerTestField<EvalType>(eval->_temperature);

    test_fixture.evaluate<EvalType>();

    const auto phi
        = test_fixture.getTestFieldData<EvalType>(eval->_lagrange_pressure);
    const auto u = test_fixture.getTestFieldData<EvalType>(eval->_velocity[0]);
    const auto v = test_fixture.getTestFieldData<EvalType>(eval->_velocity[1]);
    const auto w = test_fixture.getTestFieldData<EvalType>(eval->_velocity[2]);

    // Check number of degree of freedoms
    const int num_dofs = 8;
    EXPECT_EQ(num_dofs, phi.extent(1));
    EXPECT_EQ(num_dofs, u.extent(1));
    EXPECT_EQ(num_dofs, v.extent(1));
    EXPECT_EQ(num_dofs, w.extent(1));

    // Check values
    for (int n = 0; n < num_dofs; ++n)
    {
        EXPECT_DOUBLE_EQ(0.0, fieldValue(phi, 0, n));
        EXPECT_DOUBLE_EQ(0.8244734751684003, fieldValue(u, 0, n));
        EXPECT_DOUBLE_EQ(0.0, fieldValue(v, 0, n));
        EXPECT_DOUBLE_EQ(0.005832209590991697, fieldValue(w, 0, n));

        if (build_temp_equ)
        {
            const auto temp
                = test_fixture.getTestFieldData<EvalType>(eval->_temperature);
            EXPECT_EQ(num_dofs, temp.extent(1));
            EXPECT_FLOAT_EQ(4.0, fieldValue(temp, 0, n));
        }
    }
}

//---------------------------------------------------------------------------//
TEST(IncompressibleTurbulentChannelIsothermal, Residual)
{
    testEval<panzer::Traits::Residual>(false);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleTurbulentChannelIsothermal, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(false);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleTurbulentChannel, Residual)
{
    testEval<panzer::Traits::Residual>(true);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleTurbulentChannel, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(true);
}

//---------------------------------------------------------------------------//

} // namespace Test
} // namespace VertexCFD
