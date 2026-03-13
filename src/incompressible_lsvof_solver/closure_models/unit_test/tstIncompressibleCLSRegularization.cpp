#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleCLSRegularization.hpp"

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
    using scalar_type = typename EvalType::ScalarT;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> lambda;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_phi;
    PHX::MDField<double, panzer::Cell, panzer::Point, panzer::Dim> q;

    Dependencies(const panzer::IntegrationRule& ir)
        : lambda("CLS_lambda", ir.dl_scalar)
        , grad_phi("GRAD_level_set", ir.dl_vector)
        , q("CLS_interface_normal", ir.dl_vector)
    {
        this->addEvaluatedField(lambda);
        this->addEvaluatedField(grad_phi);
        this->addEvaluatedField(q);

        this->setName(
            "Incompressible CLS Regularization Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "Incompressible CLS regularization unit test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = grad_phi.extent(1);
        const int num_space_dim = grad_phi.extent(2);

        for (int qp = 0; qp < num_point; ++qp)
        {
            lambda(c, qp) = 1.5;

            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                grad_phi(c, qp, dim) = pow(-1, dim) * (dim + 1);
                q(c, qp, dim) = pow(-1, dim + 1) * (dim + 2);
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
        new ClosureModel::IncompressibleCLSRegularization<EvalType,
                                                          panzer::Traits,
                                                          num_space_dim>(ir));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_cls_reg);

    test_fixture.evaluate<EvalType>();

    const auto fc_reg = test_fixture.getTestFieldData<EvalType>(eval->_cls_reg);

    const int num_point = ir.num_points;

    // Expected values
    const double exp_reg[3] = {-4.5, 7.5, -10.5};

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int dim = 0; dim < num_space_dim; ++dim)
        {
            EXPECT_DOUBLE_EQ(exp_reg[dim], fieldValue(fc_reg, 0, qp, dim));
        }
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleCLSRegularization2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleCLSRegularization2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleCLSRegularization3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleCLSRegularization3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
