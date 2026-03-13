#include <VertexCFD_EvaluatorTestHarness.hpp>

#include <boundary_conditions/VertexCFD_BoundaryState_ViscousGradient.hpp>

#include <Panzer_Dimension.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>

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
// Test data dependencies.
template<class EvalType>
struct Dependencies : public PHX::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dof;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _bnd_dof;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _penalty_param;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> _normals;

    Dependencies(const panzer::IntegrationRule& ir)
        : _dof("lagrange_pressure", ir.dl_scalar)
        , _bnd_dof("BOUNDARY_lagrange_pressure", ir.dl_scalar)
        , _penalty_param("viscous_penalty_parameter", ir.dl_scalar)
        , _normals("Side Normal", ir.dl_vector)
    {
        this->addEvaluatedField(_dof);
        this->addEvaluatedField(_bnd_dof);
        this->addEvaluatedField(_penalty_param);
        this->addEvaluatedField(_normals);

        this->setName("Viscous Gradient Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "viscous gradient test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = _dof.extent(1);
        const int num_space_dim = _normals.extent(2);
        for (int qp = 0; qp < num_point; ++qp)
        {
            // Set scalar variables
            _dof(c, qp) = 0.1 * (qp + 1);
            _bnd_dof(c, qp) = 1.2 * (qp + 1);
            _penalty_param(c, qp) = 2.1 * (qp + 1);

            // Set normal vector
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                _normals(c, qp, dim) = (dim + 1) * (dim * num_point + 1) * 0.2
                                       * (qp + 1);
            }
        }
    }
};

//---------------------------------------------------------------------------//
template<class EvalType>
void testEval(const int num_space_dim)
{
    // Test fixture
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Create dependencies
    auto dep_eval = Teuchos::rcp(new Dependencies<EvalType>(*test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Create viscous_gradient evaluator.
    const std::string& dof_name = "lagrange_pressure";
    auto viscous_gradient_eval = Teuchos::rcp(
        new BoundaryCondition::ViscousGradient<EvalType, panzer::Traits>(
            *test_fixture.ir, dof_name, 2.5));
    test_fixture.registerEvaluator<EvalType>(viscous_gradient_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(viscous_gradient_eval->_grad);
    test_fixture.registerTestField<EvalType>(
        viscous_gradient_eval->_scaled_grad);

    // Evaluate viscous_gradient.
    test_fixture.evaluate<EvalType>();

    // Check viscous_gradient
    auto boundary_grad_result = test_fixture.getTestFieldData<EvalType>(
        viscous_gradient_eval->_grad);
    auto boundary_scaled_grad_result = test_fixture.getTestFieldData<EvalType>(
        viscous_gradient_eval->_scaled_grad);

    const int num_point = boundary_grad_result.extent(1);

    // Loop over quadrature points and mesh dimension
    for (int qp = 0; qp < num_point; ++qp)
    {
        // Initialize variables to calculate reference values
        const double u = 0.1 * (qp + 1);
        const double u_bnd = 1.2 * (qp + 1);
        const double delta = 5.25 * (qp + 1);

        // Compare to reference values
        for (int dim = 0; dim < num_space_dim; ++dim)
        {
            const double n = (dim + 1) * (dim * num_point + 1) * 0.2 * (qp + 1);
            EXPECT_DOUBLE_EQ(n * (u - u_bnd),
                             fieldValue(boundary_grad_result, 0, qp, dim));
            EXPECT_DOUBLE_EQ(
                delta * n * (u - u_bnd),
                fieldValue(boundary_scaled_grad_result, 0, qp, dim));
        }
    }
}

//---------------------------------------------------------------------------//
// 2-D: viscousGradient residual
TEST(ViscousGradient2D, residual_viscous_gradient_test)
{
    testEval<panzer::Traits::Residual>(2);
}

// 2-D: viscousGradient jacobian
TEST(ViscousGradient2D, jacobian_viscous_gradient_test)
{
    testEval<panzer::Traits::Jacobian>(2);
}

//---------------------------------------------------------------------------//
// 3-D: viscousGradient residual
TEST(ViscousGradient3D, residual_viscous_gradient_test)
{
    testEval<panzer::Traits::Residual>(3);
}

// 3-D: viscousGradient jacobian
TEST(ViscousGradient3D, jacobian_viscous_gradient_test)
{
    testEval<panzer::Traits::Jacobian>(3);
}

//---------------------------------------------------------------------------//

} // end namespace Test
} // end namespace VertexCFD
