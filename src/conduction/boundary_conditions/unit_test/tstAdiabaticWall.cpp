#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "conduction/boundary_conditions/VertexCFD_BoundaryState_AdiabaticWall.hpp"

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _temperature;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_temperature;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> _normals;

    Dependencies(const panzer::IntegrationRule& ir)
        : _temperature("temperature", ir.dl_scalar)
        , _grad_temperature("GRAD_temperature", ir.dl_vector)
        , _normals("Side Normal", ir.dl_vector)
    {
        this->addEvaluatedField(_temperature);
        this->addEvaluatedField(_grad_temperature);
        this->addEvaluatedField(_normals);

        this->setName("Adiabatic Wall Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        _temperature.deep_copy(2.0);
        Kokkos::parallel_for(
            "adiabatic wall test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = _temperature.extent(1);
        const int num_space_dim = _grad_temperature.extent(2);
        for (int qp = 0; qp < num_point; ++qp)
        {
            for (int d = 0; d < num_space_dim; ++d)
            {
                const int sign = pow(-1, d + 1);
                _normals(c, qp, d) = 0.6 * (d + 1) * sign;
                _grad_temperature(c, qp, d) = -0.75 * (d + 1) * sign;
            }
        }
    }
};

//---------------------------------------------------------------------------//
template<class EvalType>
void testEval(const int num_space_dim)
{
    // Test fixture
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Create dependencies
    auto dep_eval = Teuchos::rcp(new Dependencies<EvalType>(*test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Create adiabatic wall evaluator.
    auto heat_flux_wall_eval = Teuchos::rcp(
        new BoundaryCondition::AdiabaticWall<EvalType, panzer::Traits>(
            *test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(heat_flux_wall_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(
        heat_flux_wall_eval->_boundary_temperature);

    test_fixture.registerTestField<EvalType>(
        heat_flux_wall_eval->_boundary_grad_temperature);

    // Evaluate conduction temperature.
    test_fixture.evaluate<EvalType>();

    // Check conduction temperature.
    const auto boundary_temperature_result
        = test_fixture.getTestFieldData<EvalType>(
            heat_flux_wall_eval->_boundary_temperature);

    const auto boundary_grad_temperature_result
        = test_fixture.getTestFieldData<EvalType>(
            heat_flux_wall_eval->_boundary_grad_temperature);

    // Assert values
    const int num_point = boundary_temperature_result.extent(1);
    const double exp_field_2d[2] = {-0.6, 1.2};
    const double exp_field_3d[3] = {-3.03, 6.06, -9.09};

    const double* exp_field = num_space_dim == 3 ? exp_field_3d : exp_field_2d;

    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(2.0, fieldValue(boundary_temperature_result, 0, qp));

        for (int dim = 0; dim < num_space_dim; ++dim)
        {
            EXPECT_DOUBLE_EQ(
                exp_field[dim],
                fieldValue(boundary_grad_temperature_result, 0, qp, dim));
        }
    }
}

//---------------------------------------------------------------------------//
// 2-D adiabatic wall residual
TEST(AdiabaticWall2D, Residual)
{
    testEval<panzer::Traits::Residual>(2);
}

// 2-D adiabatic wall jacobian
TEST(AdiabaticWall2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2);
}

//---------------------------------------------------------------------------//
// 3-D adiabatic wall residual
TEST(AdiabaticWall3D, Residual)
{
    testEval<panzer::Traits::Residual>(3);
}

// 3-D adiabatic wall jacobian
TEST(AdiabaticWall3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(3);
}

//---------------------------------------------------------------------------//

} // end namespace Test
} // end namespace VertexCFD
