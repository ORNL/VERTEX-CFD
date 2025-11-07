#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "conduction/closure_models/VertexCFD_Closure_ConductionFlux.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
template<class EvalType>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> thermal_conductivity;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        grad_temperature;

    Dependencies(const panzer::IntegrationRule& ir)
        : thermal_conductivity("thermal_conductivity", ir.dl_scalar)
        , grad_temperature("GRAD_temperature", ir.dl_vector)
    {
        this->addEvaluatedField(thermal_conductivity);
        this->addEvaluatedField(grad_temperature);
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "conduction flux test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = thermal_conductivity.extent(1);
        const int num_space_dim = grad_temperature.extent(2);
        for (int qp = 0; qp < num_point; ++qp)
        {
            thermal_conductivity(c, qp) = 0.875;
            grad_temperature(c, qp, 0) = -26.0;
            grad_temperature(c, qp, 1) = 39.0;
            if (num_space_dim > 2)
                grad_temperature(c, qp, 2) = -52.0;
        }
    }
};

template<class EvalType>
void testEval(const int num_space_dim)
{
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    auto& ir = *test_fixture.ir;

    auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    auto eval = Teuchos::rcp(
        new ClosureModel::ConductionFlux<EvalType, panzer::Traits>(ir));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_conduction_flux);

    test_fixture.evaluate<EvalType>();

    auto fv_thermal
        = test_fixture.getTestFieldData<EvalType>(eval->_conduction_flux);

    const int num_points = ir.num_points;

    const double exp_conduction_flux[3] = {
        -22.75,
        34.125,
        num_space_dim == 2 ? std::numeric_limits<double>::quiet_NaN() : -45.5};

    for (int qp = 0; qp < num_points; ++qp)
    {
        for (int dim = 0; dim < num_space_dim; ++dim)
        {
            EXPECT_DOUBLE_EQ(exp_conduction_flux[dim],
                             fieldValue(fv_thermal, 0, qp, dim));
        }
    }
}

//-----------------------------------------------------------------//
TEST(ConductionFlux2d, Residual)
{
    testEval<panzer::Traits::Residual>(2);
}

//-----------------------------------------------------------------//
TEST(ConductionFlux3d, Residual)
{
    testEval<panzer::Traits::Residual>(3);
}

//-----------------------------------------------------------------//
TEST(ConductionFlux2d, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2);
}

//-----------------------------------------------------------------//
TEST(ConductionFlux3d, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(3);
}

} // namespace Test
} // namespace VertexCFD
