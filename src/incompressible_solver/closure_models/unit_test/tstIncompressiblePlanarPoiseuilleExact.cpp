#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressiblePlanarPoiseuilleExact.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{

template<class EvalType, int NumSpaceDim>
void testEval()
{
    // Set up test fixture
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Set non-trivial quadrature point coordinates
    auto ip_coord_view
        = test_fixture.int_values->ip_coordinates.get_static_view();
    auto ip_coord_mirror = Kokkos::create_mirror(ip_coord_view);
    ip_coord_mirror(0, 0, 0) = 2.25;
    ip_coord_mirror(0, 0, 1) = 2.20;
    Kokkos::deep_copy(ip_coord_view, ip_coord_mirror);

    // Get test fixture integrator rule
    const auto& ir = *test_fixture.ir;

    // Set list of parameters to pass to the test evaluator
    Teuchos::ParameterList closure_params;
    closure_params.set("H min", -4.0)
        .set("H max", 4.0)
        .set("Lower wall temperature", 305.0)
        .set("Upper wall temperature", 300.0)
        .set("Momentum source", 24000.0)
        .set("Density", 1.0)
        .set("Kinematic viscosity", 50.0)
        .set("Thermal conductivity", 0.5)
        .set("Specific heat capacity", 100.0);

    // Create test evaluator
    auto eval = Teuchos::rcp(
        new ClosureModel::IncompressiblePlanarPoiseuilleExact<EvalType,
                                                              panzer::Traits,
                                                              num_space_dim>(
            ir, closure_params));

    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_lagrange_pressure);
    for (int dim = 0; dim < num_space_dim; ++dim)
        test_fixture.registerTestField<EvalType>(eval->_velocity[dim]);
    test_fixture.registerTestField<EvalType>(eval->_temperature);

    test_fixture.evaluate<EvalType>();

    const auto fc_pressure
        = test_fixture.getTestFieldData<EvalType>(eval->_lagrange_pressure);
    const auto fc_mom_0
        = test_fixture.getTestFieldData<EvalType>(eval->_velocity[0]);
    const auto fc_mom_1
        = test_fixture.getTestFieldData<EvalType>(eval->_velocity[1]);
    const auto fc_energy
        = test_fixture.getTestFieldData<EvalType>(eval->_temperature);

    // Assert values
    const int num_point = ir.num_points;
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(0.0, fieldValue(fc_pressure, 0, qp));
        EXPECT_DOUBLE_EQ(2678.4, fieldValue(fc_mom_0, 0, qp));
        EXPECT_DOUBLE_EQ(0.0, fieldValue(fc_mom_1, 0, qp));
        if (num_space_dim == 3)
        {
            const auto fc_mom_2
                = test_fixture.getTestFieldData<EvalType>(eval->_velocity[2]);
            EXPECT_EQ(0.0, fieldValue(fc_mom_2, 0, qp));
        }

        EXPECT_EQ(446543149.12499994, fieldValue(fc_energy, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(IncompressiblePlanarPoiseuilleExact2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressiblePlanarPoiseuilleExact2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressiblePlanarPoiseuilleExact3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(IncompressiblePlanarPoiseuilleExact3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.type_name = "IncompressiblePlanarPoiseuilleExact";
    test_fixture.eval_name = "Exact Solution Planar Poiseuille";
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.model_params.set("H min", -4.0)
        .set("H max", 4.0)
        .set("Lower wall temperature", 305.0)
        .set("Upper wall temperature", 300.0)
        .set("Momentum source", 24000.0)
        .set("Density", 1.0)
        .set("Kinematic viscosity", 0.1)
        .set("Thermal conductivity", 0.5)
        .set("Specific heat capacity", 2.0);
    test_fixture.template buildAndTest<
        ClosureModel::IncompressiblePlanarPoiseuilleExact<EvalType,
                                                          panzer::Traits,
                                                          num_space_dim>,
        num_space_dim>();
}

TEST(IncompressiblePlanarPoiseuilleExact_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(IncompressiblePlanarPoiseuilleExact_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(IncompressiblePlanarPoiseuilleExact_Factory3D, residual_test)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(IncompressiblePlanarPoiseuilleExact_Factory3D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
