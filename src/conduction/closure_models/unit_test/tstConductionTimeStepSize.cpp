#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "conduction/closure_models/VertexCFD_Closure_ConductionTimeStepSize.hpp"

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
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> specific_heat;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> solid_density;

    PHX::MDField<double, panzer::Cell, panzer::Point, panzer::Dim> element_length;

    Dependencies(const panzer::IntegrationRule& ir)
        : thermal_conductivity("thermal_conductivity", ir.dl_scalar)
        , specific_heat("solid_specific_heat_capacity", ir.dl_scalar)
        , solid_density("solid_density", ir.dl_scalar)
        , element_length("element_length", ir.dl_vector)
    {
        this->addEvaluatedField(thermal_conductivity);
        this->addEvaluatedField(specific_heat);
        this->addEvaluatedField(solid_density);
        this->addEvaluatedField(element_length);
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "conduction time step size test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = thermal_conductivity.extent(1);
        const int num_dim = element_length.extent(2);
        for (int qp = 0; qp < num_point; ++qp)
        {
            thermal_conductivity(c, qp) = 1.75;
            specific_heat(c, qp) = 0.35;
            solid_density(c, qp) = 2.75;
            for (int dim = 0; dim < num_dim; ++dim)
            {
                element_length(c, qp, dim) = static_cast<double>(dim) + 1.0;
            }
        }
    }
};

template<class EvalType>
void testEval(const int num_space_dim)
{
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    auto eval = Teuchos::rcp(
        new ClosureModel::ConductionTimeStepSize<EvalType, panzer::Traits>(ir));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_local_dt);

    test_fixture.evaluate<EvalType>();

    auto fv_local_dt = test_fixture.getTestFieldData<EvalType>(eval->_local_dt);

    const int num_points = ir.num_points;

    const double exp_local_dt = num_space_dim == 2 ? 1.375 : 3.85;

    for (int qp = 0; qp < num_points; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_local_dt, fieldValue(fv_local_dt, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(ConductionTimeStepSize2d, Residual)
{
    testEval<panzer::Traits::Residual>(2);
}

//-----------------------------------------------------------------//
TEST(ConductionTimeStepSize3d, Residual)
{
    testEval<panzer::Traits::Residual>(3);
}

//-----------------------------------------------------------------//
TEST(ConductionTimeStepSize2d, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2);
}

//-----------------------------------------------------------------//
TEST(ConductionTimeStepSize3d, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(3);
}

} // namespace Test
} // namespace VertexCFD
