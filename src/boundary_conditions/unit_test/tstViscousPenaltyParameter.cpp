#include <VertexCFD_EvaluatorTestHarness.hpp>

#include <boundary_conditions/VertexCFD_BoundaryState_ViscousPenaltyParameter.hpp>

#include <Panzer_Dimension.hpp>
#include <Panzer_Traits.hpp>

#include <Phalanx_Evaluator_Derived.hpp>

#include <Teuchos_RCP.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
template<class EvalType>
void testEval(int num_space_dim)
{
    // Test fixture
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Define penalty
    const double penalty = 5.0;

    // Initial gradient of basis functions (one cell) and compute reference
    // values 'h_ref'
    const int num_basis = test_fixture.workset->bases[0]->grad_basis.extent(1);
    const int num_point = test_fixture.workset->bases[0]->grad_basis.extent(2);
    EXPECT_EQ((num_space_dim - 1) * 4, num_point);
    std::array<double, 8> one_over_h_ref;
    auto grad_basis_view
        = test_fixture.workset->bases[0]->grad_basis.get_static_view();
    auto grad_basis_mirror = Kokkos::create_mirror(grad_basis_view);
    for (int qp = 0; qp < num_point; ++qp)
    {
        one_over_h_ref[qp] = 0.0;
        for (int basis = 0; basis < num_basis; basis++)
        {
            double one_over_h2_ref_basis = 0.0;
            for (int dim = 0; dim < num_space_dim; dim++)
            {
                const double one_over_h2_ref_basis_dim
                    = 0.5 * (basis + 1) * (dim + 1 + num_space_dim) * (qp + 1);
                grad_basis_mirror(0, basis, qp, dim)
                    = one_over_h2_ref_basis_dim;
                one_over_h2_ref_basis += one_over_h2_ref_basis_dim
                                         * one_over_h2_ref_basis_dim;
            }

            one_over_h_ref[qp] += sqrt(one_over_h2_ref_basis);
        }
    }
    Kokkos::deep_copy(grad_basis_view, grad_basis_mirror);

    // Create viscous penalty parameter evaluator.
    const std::string& dof_name = "temperature";
    auto viscous_penalty_param = Teuchos::rcp(
        new BoundaryCondition::ViscousPenaltyParameter<EvalType, panzer::Traits>(
            *test_fixture.ir,
            *(test_fixture.workset->bases[0]->basis_layout->getBasis()),
            dof_name,
            penalty));
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

    const int num_point_rslt = boundary_penalty_param_result.extent(1);

    // Loop over quadrature points
    for (int qp = 0; qp < num_point_rslt; ++qp)
    {
        // Compare to reference values
        EXPECT_DOUBLE_EQ(5.0 * one_over_h_ref[qp],
                         fieldValue(boundary_penalty_param_result, 0, qp));
    }
}

//---------------------------------------------------------------------------//
// 2-D: viscousPenaltyParameter residual
TEST(ViscousPenaltyParameter2D, Residual)
{
    testEval<panzer::Traits::Residual>(2);
}

// 2-D: viscousPenaltyParameter jacobian
TEST(ViscousPenaltyParameter2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2);
}

//---------------------------------------------------------------------------//
// 3-D: viscousPenaltyParameter residual
TEST(ViscousPenaltyParameter3D, Residual)
{
    testEval<panzer::Traits::Residual>(3);
}

// 3-D: viscousPenaltyParameter jacobian
TEST(ViscousPenaltyParameter3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(3);
}

//---------------------------------------------------------------------------//

} // end namespace Test
} // end namespace VertexCFD
