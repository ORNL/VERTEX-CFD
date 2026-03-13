#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "boundary_conditions/VertexCFD_BoundaryState_VariableFlux.hpp"

#include <Panzer_Dimension.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>

#include <Teuchos_ParameterList.hpp>
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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _variable;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_variable;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> _normals;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dependent;

    const bool _flux_dependence;

    Dependencies(const panzer::IntegrationRule& ir,
                 const std::string coefficient_name)
        : _variable("variable", ir.dl_scalar)
        , _grad_variable("GRAD_variable", ir.dl_vector)
        , _normals("Side Normal", ir.dl_vector)
        , _dependent(coefficient_name, ir.dl_scalar)
        , _flux_dependence(!coefficient_name.empty())
    {
        this->addEvaluatedField(_variable);
        this->addEvaluatedField(_grad_variable);
        this->addEvaluatedField(_normals);

        if (_flux_dependence)
            this->addEvaluatedField(_dependent);

        this->setName("Variable Flux Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        _variable.deep_copy(0.75);
        if (_flux_dependence)
            _dependent.deep_copy(0.5);

        Kokkos::parallel_for(
            "variable flux test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = _variable.extent(1);
        const int num_space_dim = _grad_variable.extent(2);
        for (int qp = 0; qp < num_point; ++qp)
        {
            for (int d = 0; d < num_space_dim; ++d)
            {
                const int sign = pow(-1, d + 1);
                const int dimqp = (d + 1) * sign;

                _normals(c, qp, d) = 0.75 * dimqp;
                _grad_variable(c, qp, d) = 0.75 * dimqp;
            }
        }
    }
};

//---------------------------------------------------------------------------//
template<class EvalType>
void testEval(const int num_space_dim, const std::string coefficient_name = "")
{
    // Test fixture
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Create dependencies
    const auto dep_eval = Teuchos::rcp(
        new Dependencies<EvalType>(*test_fixture.ir, coefficient_name));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Boundary condition
    Teuchos::ParameterList bc_params;
    bc_params.set("Flux Value", 2.0);

    // Create adiabatic wall evaluator.
    const auto variable_flux_wall_eval = Teuchos::rcp(
        new BoundaryCondition::VariableFlux<EvalType, panzer::Traits>(
            *test_fixture.ir, bc_params, "variable", coefficient_name));
    test_fixture.registerEvaluator<EvalType>(variable_flux_wall_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(
        variable_flux_wall_eval->_boundary_variable);
    test_fixture.registerTestField<EvalType>(
        variable_flux_wall_eval->_boundary_grad_variable);

    // Evaluate conduction temperature.
    test_fixture.evaluate<EvalType>();

    // Check conduction temperature.
    const auto boundary_variable_result
        = test_fixture.getTestFieldData<EvalType>(
            variable_flux_wall_eval->_boundary_variable);

    const auto boundary_grad_variable_result
        = test_fixture.getTestFieldData<EvalType>(
            variable_flux_wall_eval->_boundary_grad_variable);

    // Expected values
    const double exp_field_2d[2]
        = {!coefficient_name.empty() ? -1.640625 : -0.140625,
           !coefficient_name.empty() ? 3.28125 : 0.28125};
    const double exp_field_3d[3]
        = {!coefficient_name.empty() ? 2.15625 : 3.65625,
           !coefficient_name.empty() ? -4.3125 : -7.3125,
           !coefficient_name.empty() ? 6.46875 : 10.96875};
    const double* exp_field = num_space_dim == 3 ? exp_field_3d : exp_field_2d;

    const int num_point = boundary_variable_result.extent(1);
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(0.75, fieldValue(boundary_variable_result, 0, qp));
        for (int dim = 0; dim < num_space_dim; ++dim)
        {
            EXPECT_DOUBLE_EQ(
                exp_field[dim],
                fieldValue(boundary_grad_variable_result, 0, qp, dim));
        }
    }
}

//---------------------------------------------------------------------------//
// 2-D wall flux residual
TEST(VariableFlux2D, Residual)
{
    testEval<panzer::Traits::Residual>(2);
}

// 2-D wall flux jacobian
TEST(VariableFlux2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2);
}

// 3-D dependent wall flux residual
TEST(VariableFluxDependent3D, Residual)
{
    testEval<panzer::Traits::Residual>(3, "thermal_conductivity");
}

// 3-D dependent wall flux jacobian
TEST(VariableFluxDependent3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(3, "thermal_conductivity");
}

//---------------------------------------------------------------------------//

} // end namespace Test
} // end namespace VertexCFD
