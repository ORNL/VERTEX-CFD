#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include <full_induction_mhd_solver/closure_models/VertexCFD_Closure_FullInductionModelErrorNorms.hpp>

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
    Kokkos::Array<PHX::MDField<double, panzer::Cell, panzer::Point>, num_space_dim>
        exact_induced_magn_field;

    // numerical solution
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        induced_magn_field;

    Dependencies(const panzer::IntegrationRule& ir)
    {
        // exact solution
        Utils::addEvaluatedVectorField(*this,
                                       ir.dl_scalar,
                                       exact_induced_magn_field,
                                       "Exact_induced_magnetic_field_");

        // numerical solution
        Utils::addEvaluatedVectorField(
            *this, ir.dl_scalar, induced_magn_field, "induced_magnetic_field_");

        this->setName("FullInduction Error Norms Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData) override
    {
        // assign prescribed values to exact and numerical solution
        for (int i = 0; i < num_space_dim; ++i)
        {
            exact_induced_magn_field[i].deep_copy(0.3 + 0.1 * i);
            induced_magn_field[i].deep_copy(0.7 + 0.2 * i);
        }
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
        new ClosureModel::FullInductionModelErrorNorms<EvalType,
                                                       panzer::Traits,
                                                       num_space_dim>(ir));
    test_fixture.registerEvaluator<EvalType>(eval);

    // Add required test fields.
    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        test_fixture.registerTestField<EvalType>(eval->_L1_error_induced[dim]);
        test_fixture.registerTestField<EvalType>(eval->_L2_error_induced[dim]);
    }

    // Evaluate
    test_fixture.evaluate<EvalType>();

    // Check the values
    const auto L1_error_induced_0
        = test_fixture.getTestFieldData<EvalType>(eval->_L1_error_induced[0]);
    const auto L1_error_induced_1
        = test_fixture.getTestFieldData<EvalType>(eval->_L1_error_induced[1]);
    const auto L2_error_induced_0
        = test_fixture.getTestFieldData<EvalType>(eval->_L2_error_induced[0]);
    const auto L2_error_induced_1
        = test_fixture.getTestFieldData<EvalType>(eval->_L2_error_induced[1]);

    // Reference values
    const double L1_ref[3] = {0.39999999999999997, 0.5, 0.6000000000000001};
    const double L2_ref[3] = {0.15999999999999998, 0.25, 0.3600000000000001};

    // Check the L1/L2 error solutions
    const int num_point = ir.num_points;
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(L1_ref[0], fieldValue(L1_error_induced_0, 0, qp));
        EXPECT_DOUBLE_EQ(L1_ref[1], fieldValue(L1_error_induced_1, 0, qp));

        EXPECT_DOUBLE_EQ(L2_ref[0], fieldValue(L2_error_induced_0, 0, qp));
        EXPECT_DOUBLE_EQ(L2_ref[1], fieldValue(L2_error_induced_1, 0, qp));

        if (num_space_dim == 3)
        {
            const auto L1_error_induced_2
                = test_fixture.getTestFieldData<EvalType>(
                    eval->_L1_error_induced[2]);
            const auto L2_error_induced_2
                = test_fixture.getTestFieldData<EvalType>(
                    eval->_L2_error_induced[2]);

            EXPECT_DOUBLE_EQ(L1_ref[2], fieldValue(L1_error_induced_2, 0, qp));
            EXPECT_DOUBLE_EQ(L2_ref[2], fieldValue(L2_error_induced_2, 0, qp));
        }
    }
}

//---------------------------------------------------------------------------//
TEST(FullInductionL1L2Isothermal_Error2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>();
}

//---------------------------------------------------------------------------//
TEST(FullInductionL1L2Isothermal_Error2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//---------------------------------------------------------------------------//
TEST(FullInductionL1L2Isothermal_Error3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>();
}

//---------------------------------------------------------------------------//
TEST(FullInductionL1L2Isothermal_Error3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//---------------------------------------------------------------------------//

template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.type_name = "FullInductionModelErrorNorm";
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Full Induction MHD Properties");
    test_fixture.factory_type = "Full Induction MHD";
    if (num_space_dim == 2)
        test_fixture.eval_name = "Full Induction Model Error Norms 2D";
    else if (num_space_dim == 3)
        test_fixture.eval_name = "Full Induction Model Error Norms 3D";
    test_fixture.template buildAndTest<
        ClosureModel::FullInductionModelErrorNorms<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(FullInductionErrorNorms_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(FullInductionErrorNorms_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(FullInductionErrorNorms_Factory3D, residual_test)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(FullInductionErrorNorms_Factory3D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

} // end namespace Test
} // end namespace VertexCFD
