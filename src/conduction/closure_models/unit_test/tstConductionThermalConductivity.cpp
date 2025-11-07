#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "conduction/closure_models/VertexCFD_Closure_ConductionThermalConductivity.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{

enum ThermCondType
{
    constant,
    inverse_proportional
};

template<class EvalType>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> temperature;

    Dependencies(const panzer::IntegrationRule& ir)
        : temperature("temperature", ir.dl_scalar)
    {
        this->addEvaluatedField(temperature);
    }

    void evaluateFields(typename panzer::Traits::EvalData /**d**/) override
    {
        temperature.deep_copy(2.0);
    }
};

template<class EvalType>
void testEval(const int num_space_dim,
              const ThermCondType therm_cond_type = ThermCondType::constant)
{
    // Test fixture
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize class object to test and expected values
    Teuchos::ParameterList closure_params;
    const double k = 10.0;
    double exp_value = k;
    if (therm_cond_type == ThermCondType::constant)
    {
        closure_params.set("Thermal Conductivity Type", "constant");
        closure_params.set("Thermal Conductivity Value", k);
    }
    else if (therm_cond_type == ThermCondType::inverse_proportional)
    {
        closure_params.set("Thermal Conductivity Type", "inverse_proportional");
        closure_params.set("Thermal Conductivity Coefficient", k);
        exp_value = k / 2.0;
    }

    const auto eval = Teuchos::rcp(
        new ClosureModel::ConductionThermalConductivity<EvalType, panzer::Traits>(
            ir, closure_params));

    // Register
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_thermal_conductivity);

    test_fixture.evaluate<EvalType>();

    const auto thermal_cond
        = test_fixture.getTestFieldData<EvalType>(eval->_thermal_conductivity);

    const int num_point = ir.num_points;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_EQ(exp_value, fieldValue(thermal_cond, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(ConstantThermalConductivity, Residual)
{
    testEval<panzer::Traits::Residual>(2);
}

//-----------------------------------------------------------------//
TEST(ConstantThermalConductivity, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2);
}

//-----------------------------------------------------------------//
TEST(OneOverTempThermalConductivity, Residual)
{
    testEval<panzer::Traits::Residual>(2, ThermCondType::inverse_proportional);
}

//-----------------------------------------------------------------//
TEST(OneOverTempThermalConductivity, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2, ThermCondType::inverse_proportional);
}

} // namespace Test
} // namespace VertexCFD
