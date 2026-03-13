#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFScalarViscousFlux.hpp"

#include "utils/VertexCFD_Utils_PhaseIndex.hpp"
#include "utils/VertexCFD_Utils_PhaseLayout.hpp"
#include "utils/VertexCFD_Utils_ScalarToVector.hpp"

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> artificial_viscosity;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_scalar;

    Dependencies(const panzer::IntegrationRule& ir)
        : artificial_viscosity("artificial_viscosity", ir.dl_scalar)
        , grad_scalar("GRAD_alpha_air", ir.dl_vector)

    {
        this->addEvaluatedField(artificial_viscosity);
        this->addEvaluatedField(grad_scalar);

        this->setName(
            "Incompressible LSVOF Viscous Flux for Scalar Equation Unit "
            "Test "
            "Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "LSVOF viscous flux test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = grad_scalar.extent(1);
        const int num_space_dim = grad_scalar.extent(2);

        for (int qp = 0; qp < num_point; ++qp)
        {
            artificial_viscosity(c, qp) = 0.1;

            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                const int sign = pow(-1, dim + 1);
                const int dimqp = (dim + 1) * sign;
                grad_scalar(c, qp, dim) = abs((0.25 + 0.5) * dimqp);
            }
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval()
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Initialize dependents
    auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize class object to test
    const auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleLSVOFScalarViscousFlux<EvalType,
                                                               panzer::Traits,
                                                               num_space_dim>(
            ir, "alpha_air", "alpha_air_equation"));

    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_scalar_flux);

    test_fixture.evaluate<EvalType>();

    const auto fc_scalar
        = test_fixture.getTestFieldData<EvalType>(eval->_scalar_flux);

    const int num_point = ir.num_points;

    const double exp_scalar_flux_3d[3] = {0.075, 0.15, 0.225};
    const double exp_scalar_flux_2d[2] = {0.075, 0.15};

    const double* exp_scalar_flux = num_space_dim == 3 ? exp_scalar_flux_3d
                                                       : exp_scalar_flux_2d;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int dim = 0; dim < num_space_dim; dim++)
        {
            EXPECT_DOUBLE_EQ(exp_scalar_flux[dim],
                             fieldValue(fc_scalar, 0, qp, dim));
        }
    }
}

//-----------------------------------------------------------------//
TEST(TestLSVOFScalarViscous2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(TestLSVOFScalarViscous2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(TestLSVOFScalarViscous3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(TestLSVOFScalarViscous3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
