#include <VertexCFD_EvaluatorTestHarness.hpp>
#include <closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp>

#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"
#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_HartmannProblemExact.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{

template<class EvalType, int NumSpaceDim>
void testEval()
{
    // Test fixture
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Set non-trivial y values
    auto ip_coord_view
        = test_fixture.int_values->ip_coordinates.get_static_view();
    auto ip_coord_mirror = Kokkos::create_mirror(ip_coord_view);
    ip_coord_mirror(0, 0, 1) = 0.5;
    Kokkos::deep_copy(ip_coord_view, ip_coord_mirror);

    // Initialize class object to test
    Teuchos::ParameterList user_params;
    user_params.set("Hartmann Solution Half-Width Channel", 2.5);
    Teuchos::Array<double> ext_magn_vct(3);
    ext_magn_vct[0] = 1.5;
    ext_magn_vct[1] = 2.0;
    ext_magn_vct[2] = 3.0;
    user_params.set("External Magnetic Field", ext_magn_vct);
    Teuchos::ParameterList fluid_prop_list;
    fluid_prop_list.set("Kinematic viscosity", 4.5);
    fluid_prop_list.set("Density", 4.0);
    fluid_prop_list.set("Artificial compressibility", 2.0);
    fluid_prop_list.set("Build Temperature Equation", false);
    fluid_prop_list.set("Build Inductionless MHD Equation", true);
    fluid_prop_list.set("Electrical conductivity", 3.5);
    const FluidProperties::ConstantFluidProperties fluid_prop(fluid_prop_list);

    const auto eval = Teuchos::rcp(
        new ClosureModel::
            HartmannProblemExact<EvalType, panzer::Traits, num_space_dim>(
                ir, fluid_prop, user_params));

    // Register
    test_fixture.registerEvaluator<EvalType>(eval);
    for (int dim = 0; dim < num_space_dim; ++dim)
        test_fixture.registerTestField<EvalType>(eval->_exact_velocity[dim]);

    test_fixture.evaluate<EvalType>();
    const auto value
        = test_fixture.getTestFieldData<EvalType>(eval->_exact_velocity[0]);

    // Assert values
    const int num_point = ir.num_points;
    const double exp_value[3] = {0.9890644233322864, 0.0, 0.0};
    for (int qp = 0; qp < num_point; ++qp)
    {
        const auto lp = test_fixture.getTestFieldData<EvalType>(
            eval->_exact_lagrange_pressure);
        EXPECT_EQ(0.0, fieldValue(lp, 0, qp));

        for (int dim = 0; dim < num_space_dim; ++dim)
        {
            const auto value = test_fixture.getTestFieldData<EvalType>(
                eval->_exact_velocity[dim]);
            EXPECT_EQ(exp_value[dim], fieldValue(value, 0, qp));
        }

        const auto ep
            = test_fixture.getTestFieldData<EvalType>(eval->_exact_elec_pot);
        EXPECT_EQ(0.0, fieldValue(ep, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(HartmannProblemExact2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(HartmannProblemExact2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(HartmannProblemExact3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(HartmannProblemExact3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.user_params.set("Build Inductionless MHD Equation", true);
    test_fixture.user_params.set("Build Temperature Equation", false);
    Teuchos::Array<double> ext_magn_vct(3);
    test_fixture.user_params.set("External Magnetic Field", ext_magn_vct);
    test_fixture.user_params.set("Hartmann Solution Half-Width Channel", 2.5);
    test_fixture.user_params.sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0)
        .set("Electrical conductivity", 3.5);
    test_fixture.type_name = "HartmannProblemExact";
    test_fixture.eval_name = "Exact Solution Hartmann Problem";
    test_fixture.template buildAndTest<
        ClosureModel::HartmannProblemExact<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(HartmannProblemExact_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(HartmannProblemExact_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

} // namespace Test
} // namespace VertexCFD
