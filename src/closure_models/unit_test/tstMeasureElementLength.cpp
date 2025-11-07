#include <VertexCFD_ClosureModelFactoryTestHarness.hpp>
#include <VertexCFD_EvaluatorTestHarness.hpp>

#include <closure_models/VertexCFD_Closure_MeasureElementLength.hpp>

#include <Panzer_Dimension.hpp>
#include <Panzer_Traits.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_MDField.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <iostream>

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
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Create evaluator.
    auto measure_element_length_eval = Teuchos::rcp(
        new ClosureModel::MeasureElementLength<EvalType, panzer::Traits>(
            *test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(measure_element_length_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(
        measure_element_length_eval->_element_length);

    // Evaluate MFEM element length.
    test_fixture.evaluate<EvalType>();

    // Check the MFEM element length.
    auto element_length_result = test_fixture.getTestFieldData<EvalType>(
        measure_element_length_eval->_element_length);
    const int num_point = element_length_result.extent(1);
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(0.5, fieldValue(element_length_result, 0, qp, 0));
        EXPECT_DOUBLE_EQ(0.5, fieldValue(element_length_result, 0, qp, 1));
    }
}

//---------------------------------------------------------------------------//
TEST(MeasureElementLength, residual_test)
{
    testEval<panzer::Traits::Residual>();
}

//---------------------------------------------------------------------------//
TEST(MeasureElementLength, jacobian_test)
{
    testEval<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//

template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.type_name = "MeasureElementLength";
    test_fixture.eval_name = "Measure Element Length";
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.template buildAndTest<
        ClosureModel::MeasureElementLength<EvalType, panzer::Traits>,
        num_space_dim>();
}

TEST(MeasureElementLength_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(MeasureElementLength_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

} // end namespace Test
} // end namespace VertexCFD
