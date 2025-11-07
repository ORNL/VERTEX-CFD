#include <VertexCFD_ClosureModelFactoryTestHarness.hpp>
#include <VertexCFD_EvaluatorTestHarness.hpp>

#include <closure_models/VertexCFD_Closure_MetricTensorElementLength.hpp>

#include <Panzer_Dimension.hpp>
#include <Panzer_IntegrationRule.hpp>
#include <Panzer_Traits.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_KokkosDeviceTypes.hpp>
#include <Phalanx_MDField.hpp>

#include <Teuchos_RCP.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <iostream>
#include <limits>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
// Test data dependencies.
template<class EvalType>
struct Dependencies : public PHX::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    static constexpr double _unused
        = std::numeric_limits<double>::signaling_NaN();

    PHX::MDField<double, panzer::Cell, panzer::Point, panzer::Dim, panzer::Dim>
        _metric_tensor;

    Dependencies(const panzer::IntegrationRule& ir)
        : _metric_tensor("metric_tensor", ir.dl_tensor)
    {
        this->addEvaluatedField(_metric_tensor);
        this->setName("Metric Tensor Element Length Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "metric tensor element length test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = _metric_tensor.extent(1);
        const int num_space_dim = _metric_tensor.extent(2);
        for (int qp = 0; qp < num_point; ++qp)
        {
            for (int dim_i = 0; dim_i < num_space_dim; ++dim_i)
                for (int dim_j = 0; dim_j < num_space_dim; ++dim_j)
                    _metric_tensor(c, qp, dim_i, dim_j) = _unused;

            for (int d = 0; d < num_space_dim; ++d)
            {
                _metric_tensor(c, qp, d, d) = 0.0625 * (d + 1) * (d + 1);
            }
        }
    }
};

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

    // Create dependencies.
    auto dep_eval = Teuchos::rcp(new Dependencies<EvalType>(*test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Create evaluator.
    auto metric_tensor_eval = Teuchos::rcp(
        new ClosureModel::MetricTensorElementLength<EvalType, panzer::Traits>(
            *test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(metric_tensor_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(
        metric_tensor_eval->_element_length);

    // Evaluate metric tensor.
    test_fixture.evaluate<EvalType>();

    // Check the metric tensor.
    auto element_length_result = test_fixture.getTestFieldData<EvalType>(
        metric_tensor_eval->_element_length);
    const int num_point = element_length_result.extent(1);
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(0.25, fieldValue(element_length_result, 0, qp, 0));
        EXPECT_DOUBLE_EQ(0.50, fieldValue(element_length_result, 0, qp, 1));
    }
}

//---------------------------------------------------------------------------//
TEST(MetricTensorElementLength, residual_test)
{
    testEval<panzer::Traits::Residual>();
}

//---------------------------------------------------------------------------//
TEST(MetricTensorElementLength, jacobian_test)
{
    testEval<panzer::Traits::Jacobian>();
}

//---------------------------------------------------------------------------//

template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.type_name = "MetricTensorElementLength";
    test_fixture.eval_name = "Metric Tensor Element Length";
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.template buildAndTest<
        ClosureModel::MetricTensorElementLength<EvalType, panzer::Traits>,
        num_space_dim>();
}

TEST(MetricTensorElementLength_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(MetricTensorElementLength_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

} // end namespace Test
} // end namespace VertexCFD
