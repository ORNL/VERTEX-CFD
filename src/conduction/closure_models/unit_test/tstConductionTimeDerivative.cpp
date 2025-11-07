#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "conduction/closure_models/VertexCFD_Closure_ConductionTimeDerivative.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
template<class EvalType>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> dxdt_temperature;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> specific_heat;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> solid_density;

    Dependencies(const panzer::IntegrationRule& ir)
        : dxdt_temperature("DXDT_temperature", ir.dl_scalar)
        , specific_heat("solid_specific_heat_capacity", ir.dl_scalar)
        , solid_density("solid_density", ir.dl_scalar)
    {
        this->addEvaluatedField(dxdt_temperature);
        this->addEvaluatedField(specific_heat);
        this->addEvaluatedField(solid_density);
    }

    void evaluateFields(typename panzer::Traits::EvalData /**d**/) override
    {
        dxdt_temperature.deep_copy(0.5);
        specific_heat.deep_copy(2.0);
        solid_density.deep_copy(3.0);
    }
};

template<class EvalType>
void testEval(const int num_space_dim)
{
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    auto eval = Teuchos::rcp(
        new ClosureModel::ConductionTimeDerivative<EvalType, panzer::Traits>(
            ir));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_dqdt_energy);

    test_fixture.evaluate<EvalType>();

    auto time_derivative
        = test_fixture.getTestFieldData<EvalType>(eval->_dqdt_energy);

    const int num_points = ir.num_points;

    const double exp_value = 3.0;

    for (int qp = 0; qp < num_points; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_value, fieldValue(time_derivative, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(ConductionTimeDerivative2d, Residual)
{
    testEval<panzer::Traits::Residual>(2);
}

//-----------------------------------------------------------------//
TEST(ConductionTimeDerivative3d, Residual)
{
    testEval<panzer::Traits::Residual>(3);
}

//-----------------------------------------------------------------//
TEST(ConductionTimeDerivative2d, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2);
}

//-----------------------------------------------------------------//
TEST(ConductionTimeDerivative3d, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(3);
}

} // namespace Test
} // namespace VertexCFD
