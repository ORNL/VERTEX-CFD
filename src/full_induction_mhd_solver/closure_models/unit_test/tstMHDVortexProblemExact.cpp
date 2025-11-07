#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_MHDVortexProblemExact.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
template<class EvalType, int NumSpaceDim>
void testEval()
{
    const int integration_order = 1;
    const int basis_order = 1;
    static constexpr int num_space_dim = NumSpaceDim;

    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;
    const double time = 0.6;
    test_fixture.setTime(time);

    // Set non-trivial coordinates for the degree of freedom
    const int num_point = ir.num_points;
    auto ip_coord_view
        = test_fixture.int_values->ip_coordinates.get_static_view();
    auto ip_coord_mirror = Kokkos::create_mirror(ip_coord_view);
    ip_coord_mirror(0, 0, 0) = 0.2;
    ip_coord_mirror(0, 0, 1) = 0.4;
    Kokkos::deep_copy(ip_coord_view, ip_coord_mirror);

    // Closure parameters
    Teuchos::Array<double> xy_0(2);
    xy_0[0] = 0.1;
    xy_0[1] = -0.2;
    Teuchos::Array<double> vel_0(2);
    vel_0[0] = 2.0;
    vel_0[1] = -0.5;
    Teuchos::ParameterList mhd_params;
    mhd_params.set<Teuchos::Array<double>>("velocity_0", vel_0);
    mhd_params.set<Teuchos::Array<double>>("center_0", xy_0);

    // Initialize and register
    auto eval = Teuchos::rcp(
        new ClosureModel::MHDVortexProblemExact<EvalType,
                                                panzer::Traits,
                                                num_space_dim>(ir, mhd_params));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_lagrange_pressure);
    test_fixture.registerTestField<EvalType>(eval->_induced_magnetic_field[0]);
    test_fixture.registerTestField<EvalType>(eval->_induced_magnetic_field[1]);
    test_fixture.registerTestField<EvalType>(eval->_velocity[0]);
    test_fixture.registerTestField<EvalType>(eval->_velocity[1]);
    test_fixture.evaluate<EvalType>();

    // Evaluate test fields
    const auto fv_lag_pres
        = test_fixture.getTestFieldData<EvalType>(eval->_lagrange_pressure);
    const auto fv_mgn_field_0 = test_fixture.getTestFieldData<EvalType>(
        eval->_induced_magnetic_field[0]);
    const auto fv_mgn_field_1 = test_fixture.getTestFieldData<EvalType>(
        eval->_induced_magnetic_field[1]);
    const auto fv_vel_0
        = test_fixture.getTestFieldData<EvalType>(eval->_velocity[0]);
    const auto fv_vel_1
        = test_fixture.getTestFieldData<EvalType>(eval->_velocity[1]);

    // Assert values
    const double tol = 1.0e-15;
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_NEAR(1.9152034448503858, fieldValue(fv_lag_pres, 0, qp), tol);
        EXPECT_NEAR(
            -0.45120855259162973, fieldValue(fv_mgn_field_0, 0, qp), tol);
        EXPECT_NEAR(
            0.07520142543193828, fieldValue(fv_mgn_field_1, 0, qp), tol);
        EXPECT_NEAR(1.5487914474083704, fieldValue(fv_vel_0, 0, qp), tol);
        EXPECT_NEAR(0.07520142543193828, fieldValue(fv_vel_1, 0, qp), tol);

        if (num_space_dim == 3)
        {
            const auto fv_mgn_field_2 = test_fixture.getTestFieldData<EvalType>(
                eval->_induced_magnetic_field[2]);
            const auto fv_vel_2
                = test_fixture.getTestFieldData<EvalType>(eval->_velocity[2]);
            EXPECT_EQ(0.0, fieldValue(fv_mgn_field_2, 0, qp));
            EXPECT_EQ(0.0, fieldValue(fv_vel_2, 0, qp));
        }
    }
}

//-----------------------------------------------------------------//
TEST(MHDVortexProblemExact2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(MHDVortexProblemExact2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(MHDVortexProblemExact3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(MHDVortexProblemExact3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    static constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.type_name = "MHDVortexProblemExact";
    test_fixture.eval_name = "MHD Vortex Problem Exact Solution "
                             + std::to_string(num_space_dim) + "D.";
    const Teuchos::Array<double> dummy(2);
    test_fixture.model_params.set("velocity_0", dummy).set("center_0", dummy);
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Full Induction MHD Properties")
        .set("Vacuum Magnetic Permeability", 0.1)
        .set("Build Magnetic Correction Potential Equation", false);
    test_fixture.factory_type = "Full Induction MHD";
    test_fixture.template buildAndTest<
        ClosureModel::MHDVortexProblemExact<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(MHDVortexProblemExact_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(MHDVortexProblemExact_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(MHDVortexProblemExact_Factory3D, residual_test)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(MHDVortexProblemExact_Factory3D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

//-----------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD
