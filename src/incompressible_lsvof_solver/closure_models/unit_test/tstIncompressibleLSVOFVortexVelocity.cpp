#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFVortexVelocity.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
// Vortex direction test options
enum class VortexDirection
{
    Forward,
    Reverse
};

//---------------------------------------------------------------------------//
// Test function
template<class EvalType, int NumSpaceDim>
void testEval(const VortexDirection vortex_direction)
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
    ip_coord_mirror(0, 0, 0) = 0.625;
    ip_coord_mirror(0, 0, 1) = 0.750;
    Kokkos::deep_copy(ip_coord_view, ip_coord_mirror);

    const auto& ir = *test_fixture.ir;

    // Parameter list
    std::string direction = "";
    if (vortex_direction == VortexDirection::Forward)
    {
        direction = "Forward";
    }
    else
    {
        direction = "Reverse";
    }

    Teuchos::ParameterList params;
    params.set("Direction", direction);

    if (num_space_dim == 3)
    {
        const std::string msg
            = "ERROR: Incompressible LSVOF Vortex Velocity must be used "
              "for 2D "
              "problem.\n";

        using eval
            = ClosureModel::IncompressibleLSVOFVortexVelocity<EvalType,
                                                              panzer::Traits,
                                                              num_space_dim>;

        ASSERT_THROW(
            try {
                auto vortex_eval = eval(*test_fixture.ir, params);
            } catch (const std::runtime_error& e) {
                EXPECT_EQ(msg, e.what());
                throw;
            },
            std::runtime_error);
    }
    else
    {
        // Initialize class object to test
        const auto eval = Teuchos::rcp(
            new ClosureModel::IncompressibleLSVOFVortexVelocity<EvalType,
                                                                panzer::Traits,
                                                                num_space_dim>(
                ir, params));
        test_fixture.registerEvaluator<EvalType>(eval);
        test_fixture.registerTestField<EvalType>(eval->_vel_0);
        test_fixture.registerTestField<EvalType>(eval->_vel_1);

        test_fixture.evaluate<EvalType>();

        const auto fc_vel_0
            = test_fixture.getTestFieldData<EvalType>(eval->_vel_0);
        const auto fc_vel_1
            = test_fixture.getTestFieldData<EvalType>(eval->_vel_1);

        // Expected values
        const double exp_vel_0 = vortex_direction == VortexDirection::Reverse
                                     ? 0.8535533905932737
                                     : -0.8535533905932737;
        const double exp_vel_1 = vortex_direction == VortexDirection::Reverse
                                     ? -0.3535533905932738
                                     : 0.3535533905932738;

        // Assert values
        const int num_points = ir.num_points;
        for (int qp = 0; qp < num_points; ++qp)
        {
            EXPECT_EQ(exp_vel_0, fieldValue(fc_vel_0, 0, qp));
            EXPECT_EQ(exp_vel_1, fieldValue(fc_vel_1, 0, qp));
        }
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleLSVOFVortexVelocityFwd, Residual)
{
    testEval<panzer::Traits::Residual, 2>(VortexDirection::Forward);
}

//-----------------------------------------------------------------//
TEST(IncompressibleLSVOFVortexVelocityFwd, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(VortexDirection::Forward);
}

//-----------------------------------------------------------------//
TEST(IncompressibleLSVOFVortexVelocityRev, Residual)
{
    testEval<panzer::Traits::Residual, 2>(VortexDirection::Reverse);
}

//-----------------------------------------------------------------//
TEST(IncompressibleLSVOFVortexVelocityRev, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(VortexDirection::Reverse);
}

//-----------------------------------------------------------------//
TEST(IncompressibleLSVOFVortexVelocityDimensionError, Residual)
{
    testEval<panzer::Traits::Residual, 3>(VortexDirection::Forward);
}

//-----------------------------------------------------------------//
TEST(IncompressibleLSVOFVortexVelocityDimensionError, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(VortexDirection::Forward);
}

//-----------------------------------------------------------------//

} // namespace Test
} // namespace VertexCFD
