#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_SolidElectricConductivity.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{

enum ElecCondType
{
    constant,
};

template<class EvalType>
void testEval(const int num_space_dim,
              const ElecCondType elec_cond_type = ElecCondType::constant)
{
    // Test fixture
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Initialize class object to test and expected values
    Teuchos::ParameterList closure_params;
    const double sigma = 10.0;
    if (elec_cond_type == ElecCondType::constant)
    {
        closure_params.set("Electric Conductivity Type", "constant");
        closure_params.set("Electric Conductivity Value", sigma);
    }

    const auto eval = Teuchos::rcp(
        new ClosureModel::SolidElectricConductivity<EvalType, panzer::Traits>(
            ir, closure_params));

    // Register
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_sigma);

    test_fixture.evaluate<EvalType>();

    const auto thermal_cond
        = test_fixture.getTestFieldData<EvalType>(eval->_sigma);

    const int num_point = ir.num_points;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_EQ(sigma, fieldValue(thermal_cond, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(ConstantElectricConductivity, Residual)
{
    testEval<panzer::Traits::Residual>(2);
}

//-----------------------------------------------------------------//
TEST(ConstantElectricConductivity, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2);
}

} // namespace Test
} // namespace VertexCFD
