#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleCLSTimeDerivative.hpp"

#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
template<class EvalType>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> sign;
    PHX::MDField<double, panzer::Cell, panzer::Point> star_sign;

    Dependencies(const panzer::IntegrationRule& ir)
        : sign("CLS_sign", ir.dl_scalar)
        , star_sign("STAR_CLS_sign", ir.dl_scalar)
    {
        this->addEvaluatedField(sign);
        this->addEvaluatedField(star_sign);
        this->setName(
            "Incompressible CLS Time Derivative Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData) override
    {
        sign.deep_copy(2.00);
        star_sign.deep_copy(0.75);
    }
};

template<class EvalType>
void testEval()
{
    const int integration_order = 2;
    const int basis_order = 1;
    constexpr int num_space_dim = 2;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Set test fixture workset alpha
    test_fixture.setAlpha(2.0);

    // Eval dependencies
    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    const auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleCLSTimeDerivative<EvalType,
                                                          panzer::Traits>(ir));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_dqdt_sign);

    test_fixture.evaluate<EvalType>();

    const auto fc_dqdt
        = test_fixture.getTestFieldData<EvalType>(eval->_dqdt_sign);

    const int num_point = ir.num_points;

    // Check on values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(2.5, fieldValue(fc_dqdt, 0, qp));
    }
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSTimeDerivative, Residual)
{
    testEval<panzer::Traits::Residual>();
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSTimeDerivative, Jacobian)
{
    testEval<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD
