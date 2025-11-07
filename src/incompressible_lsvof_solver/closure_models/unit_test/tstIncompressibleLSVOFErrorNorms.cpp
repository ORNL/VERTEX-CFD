#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFErrorNorms.hpp"

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
    PHX::MDField<double, panzer::Cell, panzer::Point> exact_dof;

    // numerical solution
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> phi;
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        velocity;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> dof;

    Dependencies(const panzer::IntegrationRule& ir)
        : exact_phi("Exact_lagrange_pressure", ir.dl_scalar)
        , exact_dof("Exact_dof", ir.dl_scalar)
        , phi("lagrange_pressure", ir.dl_scalar)
        , dof("dof", ir.dl_scalar)
    {
        // exact solution
        this->addEvaluatedField(exact_phi);
        Utils::addEvaluatedVectorField(
            *this, ir.dl_scalar, exact_velocity, "Exact_velocity_");
        this->addEvaluatedField(exact_dof);

        // numerical solution
        this->addEvaluatedField(phi);
        Utils::addEvaluatedVectorField(
            *this, ir.dl_scalar, velocity, "velocity_");
        this->addEvaluatedField(dof);

        this->setName(
            "Incompressible LSVOF Error Norms Unit Test Dependencies");
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
        exact_dof.deep_copy(0.8);
        dof.deep_copy(1.2);
    }
};

template<class EvalType, int NumSpaceDim>
void testEval(const bool solve_ns)
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

    // Create the param list to initialize the evaluator
    Teuchos::ParameterList closure_params;
    closure_params.set("DOF Name", "dof");
    closure_params.set("Compute NS Error Norms", solve_ns);

    // Create evaluator.
    const auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleLSVOFErrorNorms<EvalType,
                                                        panzer::Traits,
                                                        num_space_dim>(
            ir, closure_params));
    test_fixture.registerEvaluator<EvalType>(eval);

    // Add required test fields.
    if (solve_ns)
    {
        test_fixture.registerTestField<EvalType>(eval->_L1_error_continuity);
        test_fixture.registerTestField<EvalType>(eval->_L2_error_continuity);
        for (int dim = 0; dim < num_space_dim; ++dim)
        {
            test_fixture.registerTestField<EvalType>(
                eval->_L1_error_momentum[dim]);
            test_fixture.registerTestField<EvalType>(
                eval->_L2_error_momentum[dim]);
        }
    }

    test_fixture.registerTestField<EvalType>(eval->_L1_error_dof);
    test_fixture.registerTestField<EvalType>(eval->_L2_error_dof);

    // Evaluate
    test_fixture.evaluate<EvalType>();

    // Check the values
    const auto L1_error_dof
        = test_fixture.getTestFieldData<EvalType>(eval->_L1_error_dof);
    const auto L2_error_dof
        = test_fixture.getTestFieldData<EvalType>(eval->_L2_error_dof);

    // Reference values
    const double L1_ref[5] = {
        0.1, 0.39999999999999997, 0.5, 0.6000000000000001, 0.3999999999999999};
    const double L2_ref[5] = {0.010000000000000002,
                              0.15999999999999998,
                              0.25,
                              0.3600000000000001,
                              0.15999999999999992};

    // Check the L1/L2 error solutions
    const int num_point = ir.num_points;
    for (int qp = 0; qp < num_point; ++qp)
    {
        if (solve_ns)
        {
            const auto L1_error_continuity
                = test_fixture.getTestFieldData<EvalType>(
                    eval->_L1_error_continuity);
            const auto L1_error_momentum_0
                = test_fixture.getTestFieldData<EvalType>(
                    eval->_L1_error_momentum[0]);
            const auto L1_error_momentum_1
                = test_fixture.getTestFieldData<EvalType>(
                    eval->_L1_error_momentum[1]);
            const auto L2_error_continuity
                = test_fixture.getTestFieldData<EvalType>(
                    eval->_L2_error_continuity);
            const auto L2_error_momentum_0
                = test_fixture.getTestFieldData<EvalType>(
                    eval->_L2_error_momentum[0]);
            const auto L2_error_momentum_1
                = test_fixture.getTestFieldData<EvalType>(
                    eval->_L2_error_momentum[1]);

            EXPECT_DOUBLE_EQ(L1_ref[0], fieldValue(L1_error_continuity, 0, qp));
            EXPECT_DOUBLE_EQ(L1_ref[1], fieldValue(L1_error_momentum_0, 0, qp));
            EXPECT_DOUBLE_EQ(L1_ref[2], fieldValue(L1_error_momentum_1, 0, qp));

            EXPECT_DOUBLE_EQ(L2_ref[0], fieldValue(L2_error_continuity, 0, qp));
            EXPECT_DOUBLE_EQ(L2_ref[1], fieldValue(L2_error_momentum_0, 0, qp));
            EXPECT_DOUBLE_EQ(L2_ref[2], fieldValue(L2_error_momentum_1, 0, qp));

            if (num_space_dim == 3)
            {
                const auto L1_error_momentum_2
                    = test_fixture.getTestFieldData<EvalType>(
                        eval->_L1_error_momentum[2]);
                const auto L2_error_momentum_2
                    = test_fixture.getTestFieldData<EvalType>(
                        eval->_L2_error_momentum[2]);

                EXPECT_DOUBLE_EQ(L1_ref[num_space_dim],
                                 fieldValue(L1_error_momentum_2, 0, qp));
                EXPECT_DOUBLE_EQ(L2_ref[num_space_dim],
                                 fieldValue(L2_error_momentum_2, 0, qp));
            }
        }

        EXPECT_DOUBLE_EQ(L1_ref[4], fieldValue(L1_error_dof, 0, qp));
        EXPECT_DOUBLE_EQ(L2_ref[4], fieldValue(L2_error_dof, 0, qp));
    }
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFErrorNorms_NoNS2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(false);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFErrorNorms_NoNS2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(false);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFErrorNorms_NoNS3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(false);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFErrorNorms_NoNS3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(false);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFErrorNorms_2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(true);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFErrorNorms_2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(true);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFErrorNorms_3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(true);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLSVOFErrorNorms_3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(true);
}

//---------------------------------------------------------------------------//

} // end namespace Test
} // end namespace VertexCFD
