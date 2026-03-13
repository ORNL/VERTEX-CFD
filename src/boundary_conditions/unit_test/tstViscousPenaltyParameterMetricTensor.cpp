#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "boundary_conditions/VertexCFD_BoundaryState_ViscousPenaltyParameterMetricTensor.hpp"

#include <Panzer_Dimension.hpp>
#include <Panzer_Traits.hpp>

#include <Phalanx_Evaluator_Derived.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>

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
    using scalar_type = typename EvalType::ScalarT;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> _normals;

    PHX::MDField<double, panzer::Cell, panzer::Point, panzer::Dim, panzer::Dim>
        _metric_tensor;

    Dependencies(const panzer::IntegrationRule& ir)
        : _normals("Side Normal", ir.dl_vector)
        , _metric_tensor("metric_tensor", ir.dl_tensor)
    {
        this->addEvaluatedField(_normals);
        this->addEvaluatedField(_metric_tensor);
        this->setName(
            "ViscousPenaltyParameterMetricTensor Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "viscous penalty parameter test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int) const
    {
        const int num_space_dim = _metric_tensor.extent(2);
        if (num_space_dim == 2)
        {
            _normals(0, 0, 0) = 0.6;
            _normals(0, 0, 1) = 0.8;

            _metric_tensor(0, 0, 0, 0) = 0.5;
            _metric_tensor(0, 0, 0, 1) = -0.125;
            _metric_tensor(0, 0, 1, 0) = -0.125;
            _metric_tensor(0, 0, 1, 1) = 0.3125;
        }
        if (num_space_dim == 3)
        {
            _normals(0, 0, 0) = 1.0;
            _normals(0, 0, 1) = 2.0;
            _normals(0, 0, 2) = 2.0;

            _metric_tensor(0, 0, 0, 0) = 0.5;
            _metric_tensor(0, 0, 0, 1) = -0.125;
            _metric_tensor(0, 0, 0, 2) = 0.25;
            _metric_tensor(0, 0, 1, 0) = -0.125;
            _metric_tensor(0, 0, 1, 1) = -0.125;
            _metric_tensor(0, 0, 1, 2) = 0.3125;
            _metric_tensor(0, 0, 2, 0) = -0.125;
            _metric_tensor(0, 0, 2, 1) = -0.125;
            _metric_tensor(0, 0, 2, 2) = 0.125;
        }
    }
};

//---------------------------------------------------------------------------//
template<class EvalType>
void testEval(const int num_space_dim)
{
    // Test fixture
    const int side_id = 0;
    const int integration_order = 0;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order, side_id);

    // Create dependencies
    auto dep_eval = Teuchos::rcp(new Dependencies<EvalType>(*test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Create viscous penalty parameter evaluator.
    auto viscous_penalty_param = Teuchos::rcp(
        new BoundaryCondition::ViscousPenaltyParameterMetricTensor<EvalType,
                                                                   panzer::Traits>(
            *test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(viscous_penalty_param);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(
        viscous_penalty_param->_penalty_param);

    // Evaluate viscous penalty parameter.
    test_fixture.evaluate<EvalType>();

    // Check viscous penalty parameter
    auto boundary_penalty_param_result
        = test_fixture.getTestFieldData<EvalType>(
            viscous_penalty_param->_penalty_param);

    // Check result. Verified via python.
    const double n_dot_m_dot_n = (2 == num_space_dim) ? 0.26 : 1.0;
    EXPECT_DOUBLE_EQ(1.0 / std::sqrt(n_dot_m_dot_n),
                     fieldValue(boundary_penalty_param_result, 0, 0));
}

//---------------------------------------------------------------------------//
// 2-D: residual
TEST(ViscousPenaltyParameterMetricTensor2D, Residual)
{
    testEval<panzer::Traits::Residual>(2);
}

// 2-D: jacobian
TEST(ViscousPenaltyParameterMetricTensor2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2);
}

//---------------------------------------------------------------------------//
// 3-D: residual
TEST(ViscousPenaltyParameterMetricTensor3D, Residual)
{
    testEval<panzer::Traits::Residual>(3);
}

// 3-D: jacobian
TEST(ViscousPenaltyParameterMetricTensor3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(3);
}

//---------------------------------------------------------------------------//

} // end namespace Test
} // end namespace VertexCFD
