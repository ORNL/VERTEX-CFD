#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "conduction/closure_models/VertexCFD_Closure_ConductionVolumetricSource.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{

enum HeatSourceType
{
    constant,
    xlinear
};

template<class EvalType>
void testEval(const HeatSourceType heat_source_type = HeatSourceType::constant)
{
    // Test fixture
    const int num_space_dim = 2;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Set x
    const double x = 0.1;
    auto ip_coord_view
        = test_fixture.int_values->ip_coordinates.get_static_view();
    auto ip_coord_mirror = Kokkos::create_mirror(ip_coord_view);
    ip_coord_mirror(0, 0, 0) = x;
    Kokkos::deep_copy(ip_coord_view, ip_coord_mirror);

    // Initialize class object to test and expected values
    const double q = 10.0;
    double exp_value = q;
    Teuchos::ParameterList closure_params;
    if (heat_source_type == HeatSourceType::constant)
    {
        closure_params.set("Heat Source Type", "constant");
    }
    else
    {
        closure_params.set("Heat Source Type", "xlinear");
        closure_params.set("X-value of the left boundary", 0.0);
        closure_params.set("X-value of the right boundary", 1.0);
        exp_value *= x;
    }
    closure_params.set("Volumetric Heat Source Value", q);

    const auto eval = Teuchos::rcp(
        new ClosureModel::ConductionVolumetricSource<EvalType, panzer::Traits>(
            ir, closure_params));

    // Register
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_source);

    test_fixture.evaluate<EvalType>();

    const auto source = test_fixture.getTestFieldData<EvalType>(eval->_source);

    // Assert values
    const int num_point = ir.num_points;
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_EQ(exp_value, fieldValue(source, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(ConstantVolumetricSource, Residual)
{
    testEval<panzer::Traits::Residual>();
}

//-----------------------------------------------------------------//
TEST(ConstantVolumetricSource, Jacobian)
{
    testEval<panzer::Traits::Jacobian>();
}

//-----------------------------------------------------------------//
TEST(LinearVolumetricSource, Residual)
{
    testEval<panzer::Traits::Residual>(HeatSourceType::xlinear);
}

//-----------------------------------------------------------------//
TEST(LinearVolumetricSource, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(HeatSourceType::xlinear);
}

} // namespace Test
} // namespace VertexCFD
