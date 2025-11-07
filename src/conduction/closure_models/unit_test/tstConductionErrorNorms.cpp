#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "conduction/closure_models/VertexCFD_Closure_ConductionErrorNorms.hpp"

#include <gtest/gtest.h>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    // exact solution
    PHX::MDField<double, panzer::Cell, panzer::Point> exact_phi;
    Kokkos::Array<PHX::MDField<double, panzer::Cell, panzer::Point>, num_space_dim>
        exact_velocity;
    PHX::MDField<double, panzer::Cell, panzer::Point> exact_ep;

    // numerical solution
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> phi;
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        velocity;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> ep;

    Dependencies(const panzer::IntegrationRule& ir)
        : exact_phi("Exact_lagrange_pressure", ir.dl_scalar)
        , exact_ep("Exact_temperature", ir.dl_scalar)
        , phi("lagrange_pressure", ir.dl_scalar)
        , ep("temperature", ir.dl_scalar)
    {
        // exact solution
        this->addEvaluatedField(exact_phi);
        Utils::addEvaluatedVectorField(
            *this, ir.dl_scalar, exact_velocity, "Exact_velocity_");
        this->addEvaluatedField(exact_ep);

        // numerical solution
        this->addEvaluatedField(phi);
        Utils::addEvaluatedVectorField(
            *this, ir.dl_scalar, velocity, "velocity_");
        this->addEvaluatedField(ep);

        this->setName("Conduction Error Norms Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData) override
    {
        // assign prescribed values to exact and numerical solution
        exact_phi.deep_copy(0.1);
        phi.deep_copy(0.2);
        for (int i = 0; i < num_space_dim; ++i)
        {
            exact_velocity[i].deep_copy(0.3 + 0.1 * i);
            velocity[i].deep_copy(0.7 + 0.2 * i);
        }
        exact_ep.deep_copy(0.8);
        ep.deep_copy(1.2);
    }
};

template<class EvalType, int NumSpaceDim>
void testEval()
{
    // Setup test fixture.
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 0;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    const auto deps
        = Teuchos::rcp(new Dependencies<EvalType, num_space_dim>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Create evaluator.
    const auto eval = Teuchos::rcp(
        new ClosureModel::
            ConductionErrorNorms<EvalType, panzer::Traits, num_space_dim>(ir));
    test_fixture.registerEvaluator<EvalType>(eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(eval->_L1_error_continuity);
    test_fixture.registerTestField<EvalType>(eval->_L2_error_continuity);
    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        test_fixture.registerTestField<EvalType>(eval->_L1_error_momentum[dim]);
        test_fixture.registerTestField<EvalType>(eval->_L2_error_momentum[dim]);
    }

    // Evaluate
    test_fixture.evaluate<EvalType>();

    // Check the values
    const auto L1_error_continuity
        = test_fixture.getTestFieldData<EvalType>(eval->_L1_error_continuity);
    const auto L2_error_continuity
        = test_fixture.getTestFieldData<EvalType>(eval->_L2_error_continuity);
    const auto L1_error_ep_equ
        = test_fixture.getTestFieldData<EvalType>(eval->_L1_error_energy);
    const auto L2_error_ep_equ
        = test_fixture.getTestFieldData<EvalType>(eval->_L2_error_energy);

    // Check the L1/L2 error solutions
    const int num_point = ir.num_points;
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(0., fieldValue(L1_error_continuity, 0, qp));
        EXPECT_DOUBLE_EQ(0., fieldValue(L2_error_continuity, 0, qp));

        for (int dim = 0; dim < num_space_dim; ++dim)
        {
            const auto L1_error_momentum
                = test_fixture.getTestFieldData<EvalType>(
                    eval->_L1_error_momentum[dim]);
            const auto L2_error_momentum
                = test_fixture.getTestFieldData<EvalType>(
                    eval->_L2_error_momentum[dim]);

            EXPECT_DOUBLE_EQ(0., fieldValue(L1_error_momentum, 0, qp));
            EXPECT_DOUBLE_EQ(0., fieldValue(L2_error_momentum, 0, qp));
        }

        EXPECT_DOUBLE_EQ(0.4, fieldValue(L1_error_ep_equ, 0, qp));
        EXPECT_DOUBLE_EQ(0.15999999999999992,
                         fieldValue(L2_error_ep_equ, 0, qp));
    }
}

//---------------------------------------------------------------------------//
TEST(ConductionError, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

//---------------------------------------------------------------------------//
TEST(ConductionError, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//---------------------------------------------------------------------------//

} // end namespace Test
} // end namespace VertexCFD
