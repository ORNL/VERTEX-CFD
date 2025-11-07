#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "conduction/closure_models/VertexCFD_Closure_ConductionExactSolution.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{

template<class EvalType>
void testEval()
{
    constexpr int num_space_dim = 2;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Set non-trivial coordinates for the degree of freedom
    const int num_point = ir.num_points;
    auto ip_coord_view
        = test_fixture.int_values->ip_coordinates.get_static_view();
    auto ip_coord_mirror = Kokkos::create_mirror(ip_coord_view);
    for (int qp = 0; qp < num_point; ++qp)
    {
        ip_coord_mirror(0, qp, 0) = 0.1 * (qp + 1);
    }
    Kokkos::deep_copy(ip_coord_view, ip_coord_mirror);

    // Initialize class object to test
    Teuchos::ParameterList closure_params;
    closure_params.set("Volumetric Heat Source Value", 10.0);
    closure_params.set("Thermal Conductivity Coefficient", 2.0);
    closure_params.set("Right Temperature Boundary Value", 300.0);

    const auto eval = Teuchos::rcp(
        new ClosureModel::
            ConductionExactSolution<EvalType, panzer::Traits, num_space_dim>(
                ir, closure_params));

    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_temperature);
    test_fixture.evaluate<EvalType>();

    const auto temp
        = test_fixture.getTestFieldData<EvalType>(eval->_temperature);

    // Reference value
    const double temp_ref = 689.7177629135476;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(temp_ref, fieldValue(temp, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(ConductionExactSolution, Residual)
{
    testEval<panzer::Traits::Residual>();
}

//-----------------------------------------------------------------//
TEST(ConductionExactSolution, Jacobian)
{
    testEval<panzer::Traits::Jacobian>();
}

} // namespace Test
} // namespace VertexCFD
