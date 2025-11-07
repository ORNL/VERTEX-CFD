#include <VertexCFD_ClosureModelFactoryTestHarness.hpp>
#include <VertexCFD_EvaluatorTestHarness.hpp>

#include <closure_models/VertexCFD_Closure_ElementLength.hpp>

#include <Panzer_Traits.hpp>

#include <Teuchos_RCP.hpp>

#include <gtest/gtest.h>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
template<class EvalType>
void testEval()
{
    // Setup test fixture.
    const int num_space_dim = 2;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Create evaluator.
    auto length_eval = Teuchos::rcp(
        new ClosureModel::ElementLength<EvalType, panzer::Traits>(
            *test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(length_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(length_eval->_element_length);

    // Evaluate metric tensor.
    test_fixture.evaluate<EvalType>();

    // Check the metric tensor.
    auto element_length_result = test_fixture.getTestFieldData<EvalType>(
        length_eval->_element_length);
    EXPECT_EQ(1, element_length_result.extent(1));
    EXPECT_DOUBLE_EQ(1.0, fieldValue(element_length_result, 0, 0, 0));
}

//---------------------------------------------------------------------------//
TEST(ElementLength, residual_test)
{
    testEval<panzer::Traits::Residual>();
}

//---------------------------------------------------------------------------//
TEST(ElementLength, jacobian_test)
{
    testEval<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//

template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.type_name = "ElementLength";
    test_fixture.eval_name = "Element Length";
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.template buildAndTest<
        ClosureModel::ElementLength<EvalType, panzer::Traits>,
        num_space_dim>();
}

TEST(ElementLength_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(ElementLength_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

} // end namespace Test
} // end namespace VertexCFD
