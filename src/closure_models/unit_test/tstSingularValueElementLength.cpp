#include <VertexCFD_ClosureModelFactoryTestHarness.hpp>
#include <VertexCFD_EvaluatorTestHarness.hpp>

#include <closure_models/VertexCFD_Closure_SingularValueElementLength.hpp>

#include <Panzer_Dimension.hpp>
#include <Panzer_Traits.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_MDField.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <stdexcept>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
template<class EvalType>
void testEval(const std::string method)
{
    // Setup test fixture.
    const int num_space_dim = 2;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Overwrite Jacobian values with trivial values for each quadrature points
    // test_fixture.int_values->jac(cell, qp, i, j)
    auto jac_view = test_fixture.int_values->jac.get_static_view();
    auto jac_mirror = Kokkos::create_mirror(jac_view);

    // qp = 0
    jac_mirror(0, 0, 0, 0) = 4.0;
    jac_mirror(0, 0, 0, 1) = 0.0;
    jac_mirror(0, 0, 1, 0) = 0.0;
    jac_mirror(0, 0, 1, 1) = 9.0;

    // qp = 1
    jac_mirror(0, 1, 0, 0) = 2.0;
    jac_mirror(0, 1, 0, 1) = 6.0;
    jac_mirror(0, 1, 1, 0) = 6.0;
    jac_mirror(0, 1, 1, 1) = 2.0;

    // qp = 2
    jac_mirror(0, 2, 0, 0) = -0.2;
    jac_mirror(0, 2, 0, 1) = 0.3;
    jac_mirror(0, 2, 1, 0) = 0.4;
    jac_mirror(0, 2, 1, 1) = 0.5;

    // qp = 3
    jac_mirror(0, 3, 0, 0) = 2.0;
    jac_mirror(0, 3, 0, 1) = -2.0;
    jac_mirror(0, 3, 1, 0) = -2.0;
    jac_mirror(0, 3, 1, 1) = 2.0;

    Kokkos::deep_copy(jac_view, jac_mirror);

    // Create evaluator.
    auto singular_value_element_length_eval = Teuchos::rcp(
        new ClosureModel::SingularValueElementLength<EvalType, panzer::Traits>(
            *test_fixture.ir, method));
    test_fixture.registerEvaluator<EvalType>(
        singular_value_element_length_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(
        singular_value_element_length_eval->_element_length);

    // Evaluate MFEM element length.
    test_fixture.evaluate<EvalType>();

    // Check the MFEM element length.
    auto element_length_result = test_fixture.getTestFieldData<EvalType>(
        singular_value_element_length_eval->_element_length);
    const int num_point = element_length_result.extent(1);
    // Reference values 'sigma'
    Kokkos::Array<double, 4> sigma;
    sigma[0] = method == "singular_value_max" ? 9.0 : 4.0;
    sigma[1] = method == "singular_value_max" ? 8.0 : 4.0;
    sigma[2] = method == "singular_value_max" ? 0.65308862983900229
                                              : 0.33686086382216435;
    sigma[3] = method == "singular_value_max" ? 4.0 : 0.0;
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(sigma[qp],
                         fieldValue(element_length_result, 0, qp, 0));
        EXPECT_DOUBLE_EQ(sigma[qp],
                         fieldValue(element_length_result, 0, qp, 1));
    }
}

template<class EvalType>
void testCatchException(const std::string method)
{
    const std::string msg
        = "Element Length Method 'None' is not a correct input.\n"
          "Choose between 'singular_value_min' or 'singular_value_max'";

    EXPECT_THROW(
        try { testEval<EvalType>(method); } catch (const std::runtime_error& e) {
            EXPECT_EQ(msg, e.what());
            throw;
        },
        std::runtime_error);
}

//---------------------------------------------------------------------------//
TEST(SingularValueElementLengthMax, residual_test)
{
    testEval<panzer::Traits::Residual>("singular_value_max");
}

//---------------------------------------------------------------------------//
TEST(SingularValueElementLengthMin, residual_test)
{
    testEval<panzer::Traits::Residual>("singular_value_min");
}

//---------------------------------------------------------------------------//
TEST(SingularValueElementLengthNone, residual_test)
{
    testCatchException<panzer::Traits::Residual>("None");
}

//---------------------------------------------------------------------------//
TEST(SingularValueElementLengthMax, jacobian_test)
{
    testEval<panzer::Traits::Jacobian>("singular_value_max");
}

//---------------------------------------------------------------------------//
TEST(SingularValueElementLengthMin, jacobian_test)
{
    testEval<panzer::Traits::Jacobian>("singular_value_min");
}

//---------------------------------------------------------------------------//
TEST(SingularValueElementLengthNone, jacobian_test)
{
    testCatchException<panzer::Traits::Jacobian>("None");
}

//---------------------------------------------------------------------------//

template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.type_name = "SingularValueElementLength";
    test_fixture.eval_name = "Singular Value Element Length";
    test_fixture.model_params.set("Element Length Method",
                                  "singular_value_min");
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.template buildAndTest<
        ClosureModel::SingularValueElementLength<EvalType, panzer::Traits>,
        num_space_dim>();
}

TEST(SingularValueElementLength_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(SingularValueElementLength_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

} // end namespace Test
} // end namespace VertexCFD
