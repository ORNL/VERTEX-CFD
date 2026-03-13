#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleTaylorGreenVortexExactSolution.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{

template<class EvalType>
void testEval()
{
    constexpr int num_space_dim = 2;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);
    const auto& ir = *test_fixture.ir;
    const double time = 0.5;
    test_fixture.setTime(time);

    // Set non-trivial coordinates for the degree of freedom
    const int num_point = ir.num_points;
    auto ip_coord_view
        = test_fixture.int_values->ip_coordinates.get_static_view();
    auto ip_coord_mirror = Kokkos::create_mirror(ip_coord_view);
    for (int dim = 0; dim < num_space_dim; dim++)
    {
        for (int qp = 0; qp < num_point; ++qp)
        {
            ip_coord_mirror(0, qp, dim) = 0.1 * (dim + 1) * (qp + 1) - 0.25;
        }
    }
    Kokkos::deep_copy(ip_coord_view, ip_coord_mirror);

    // Initialize class object to test
    Teuchos::ParameterList closure_params;
    closure_params.set("Kinematic Viscosity", 0.325);
    const auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleTaylorGreenVortexExactSolution<
            EvalType,
            panzer::Traits,
            num_space_dim>(ir, closure_params));

    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_lagrange_pressure);
    for (int dim = 0; dim < num_space_dim; ++dim)
        test_fixture.registerTestField<EvalType>(eval->_velocity[dim]);
    test_fixture.evaluate<EvalType>();

    const auto exact_cont
        = test_fixture.getTestFieldData<EvalType>(eval->_lagrange_pressure);
    const auto exact_mom_0
        = test_fixture.getTestFieldData<EvalType>(eval->_velocity[0]);
    const auto exact_mom_1
        = test_fixture.getTestFieldData<EvalType>(eval->_velocity[1]);

    // Reference values
    const double phi_ref[4] = {-0.2545417754691831,
                               -0.2545417754691831,
                               -0.2296800890258847,
                               -0.18388182976977963};
    const double u_ref[4] = {-0.035705825747158935,
                             0.10783820008203808,
                             0.2474434185973639,
                             0.37341515252860846};
    const double v_ref[4] = {0.10783820008203808,
                             0.035705825747158935,
                             -0.03392198573058794,
                             -0.09204974820065662};

    // Assert values
    const double tol = 1.0e-16;
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_NEAR(phi_ref[qp], fieldValue(exact_cont, 0, qp), tol);
        EXPECT_NEAR(u_ref[qp], fieldValue(exact_mom_0, 0, qp), tol);
        EXPECT_NEAR(v_ref[qp], fieldValue(exact_mom_1, 0, qp), tol);
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleTaylorGreenVortexExactSolution, residual_test)
{
    testEval<panzer::Traits::Residual>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleTaylorGreenVortexExactSolution, jacobian_test)
{
    testEval<panzer::Traits::Jacobian>();
}

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.type_name = "IncompressibleTaylorGreenVortexExactSolution";
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.model_params.set("Kinematic Viscosity", 0.325);
    test_fixture.eval_name
        = "Incompressible Taylor Green Vortex Exact Solution";
    test_fixture.template buildAndTest<
        ClosureModel::IncompressibleTaylorGreenVortexExactSolution<EvalType,
                                                                   panzer::Traits,
                                                                   num_space_dim>,
        num_space_dim>();
}

TEST(IncompressibleTaylorGreenVortexExactSolution_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(IncompressibleTaylorGreenVortexExactSolution_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

} // namespace Test
} // namespace VertexCFD
