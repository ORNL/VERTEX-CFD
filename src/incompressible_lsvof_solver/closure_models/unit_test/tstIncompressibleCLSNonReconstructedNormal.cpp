#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleCLSNonReconstructedNormal.hpp"

#include <Teuchos_RCP.hpp>

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
template<class EvalType>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    PHX::MDField<double, panzer::Cell, panzer::Point, panzer::Dim> grad_phi_0;

    Dependencies(const panzer::IntegrationRule& ir)
        : grad_phi_0("STAR_GRAD_level_set", ir.dl_vector)
    {
        this->addEvaluatedField(grad_phi_0);

        this->setName(
            "Incompressible CLS Non-Reconstructed Interface Normal Unit Test "
            "Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "Incompressible CLS non-reconstructed interface normal unit test "
            "dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = grad_phi_0.extent(1);
        const int num_space_dim = grad_phi_0.extent(2);

        for (int qp = 0; qp < num_point; ++qp)
        {
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                grad_phi_0(c, qp, dim) = pow(-1, dim) * (dim + 1);
            }
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval()
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize class object to test
    const auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleCLSNonReconstructedNormal<EvalType,
                                                                  panzer::Traits,
                                                                  num_space_dim>(
            ir));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_q);

    test_fixture.evaluate<EvalType>();

    const auto fc_q = test_fixture.getTestFieldData<EvalType>(eval->_q);

    const int num_point = ir.num_points;

    // Expected values
    const double exp_q_2D[2] = {0.4472135954999579, -0.8944271909999159};
    const double exp_q_3D[3]
        = {0.2672612419124244, -0.5345224838248488, 0.8017837257372732};
    const double* exp_q = num_space_dim == 2 ? exp_q_2D : exp_q_3D;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int dim = 0; dim < num_space_dim; ++dim)
        {
            EXPECT_DOUBLE_EQ(exp_q[dim], fieldValue(fc_q, 0, qp, dim));
        }
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleCLSNonReconstructedNormal2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleCLSNonReconstructedNormal2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleCLSNonReconstructedNormal3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleCLSNonReconstructedNormal3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
