#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_SolidElectricPotentialDiffusionFlux.hpp"

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        grad_electric_potential;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> sigma;

    Dependencies(const panzer::IntegrationRule& ir, std::string grad_prefix)
        : grad_electric_potential(grad_prefix + "GRAD_electric_potential",
                                  ir.dl_vector)
        , sigma("solid_electric_conductivity", ir.dl_scalar)
    {
        this->addEvaluatedField(grad_electric_potential);
        this->addEvaluatedField(sigma);
        this->setName(
            "Solid Electric Potential Diffusion Flux Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "solid electric potential diffusion flux test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = grad_electric_potential.extent(1);
        const int num_grad_dim = grad_electric_potential.extent(2);
        for (int qp = 0; qp < num_point; ++qp)
        {
            sigma(c, qp) = 3.0;
            for (int dim = 0; dim < num_grad_dim; ++dim)
                grad_electric_potential(c, qp, dim) = 1.5 * (qp + 1)
                                                      * (dim + 1);
        }
    }
};

//-----------------------------------------------------------------------------//
template<class EvalType>
void testEval(const int num_grad_dim,
              const std::string flux_prefix = "",
              const std::string grad_prefix = "")
{
    // Test fixture
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_grad_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Initialize dependents
    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir, grad_prefix));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize closure model constructor
    const auto eval = Teuchos::rcp(
        new ClosureModel::SolidElectricPotentialDiffusionFlux<EvalType,
                                                              panzer::Traits>(
            ir, flux_prefix, grad_prefix));

    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_electric_potential_flux);
    test_fixture.evaluate<EvalType>();

    const auto diff_flux = test_fixture.getTestFieldData<EvalType>(
        eval->_electric_potential_flux);

    // Assert values
    const int num_point = ir.num_points;
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int dim = 0; dim < num_grad_dim; dim++)
        {
            const auto exp_value = -3.0 * 1.5 * (qp + 1) * (dim + 1);
            EXPECT_DOUBLE_EQ(exp_value, fieldValue(diff_flux, 0, qp, dim));
        }
    }
}

//-----------------------------------------------------------------//
TEST(SolidElectricPotentialDiffusion2D, Residual)
{
    testEval<panzer::Traits::Residual>(2);
}

//-----------------------------------------------------------------//
TEST(SolidElectricPotentialDiffusion2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2);
}

//-----------------------------------------------------------------//
TEST(SolidElectricPotentialDiffusion3D, Residual)
{
    testEval<panzer::Traits::Residual>(3);
}

//-----------------------------------------------------------------//
TEST(SolidElectricPotentialDiffusion3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(3);
}

//-----------------------------------------------------------------//
TEST(SolidElectricPotentialDiffusion3DPrefix, Residual)
{
    testEval<panzer::Traits::Residual>(3, "BAR_", "BOO_");
}

//-----------------------------------------------------------------//
TEST(SolidElectricPotentialDiffusion3DPrefix, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(3, "BAR_", "BOO_");
}

} // namespace Test
} // namespace VertexCFD
