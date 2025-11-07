#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "closure_models/VertexCFD_Closure_ConstantVectorField.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{

template<class EvalType, int NumSpaceDim>
void testEval()
{
    // Test fixture
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Initialize class object to test and expected values
    Teuchos::ParameterList closure_params;
    closure_params.set("Vector Field Name", "velocity");
    Teuchos::Array<double> vec_field_vct(num_space_dim);

    vec_field_vct[0] = 1.3;
    vec_field_vct[1] = 3.5;

    if (num_space_dim == 3)
    {
        vec_field_vct[2] = 2.4;
    }
    closure_params.set("Vector Field Value", vec_field_vct);

    const auto eval = Teuchos::rcp(
        new ClosureModel::ConstantVectorField<EvalType, panzer::Traits, num_space_dim>(
            ir, closure_params));

    // Register
    test_fixture.registerEvaluator<EvalType>(eval);
    for (int dim = 0; dim < num_space_dim; ++dim)
        test_fixture.registerTestField<EvalType>(eval->_vector_field[dim]);
    test_fixture.evaluate<EvalType>();

    // Expected vector field
    const double exp_vec_field_3d[3] = {1.3, 3.5, 2.4};
    const double exp_vec_field_2d[2] = {1.3, 3.5};

    const double* exp_vec_field = num_space_dim == 3 ? exp_vec_field_3d
                                                     : exp_vec_field_2d;

    const int num_point = ir.num_points;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int dim = 0; dim < num_space_dim; ++dim)
        {
            const auto calc_vec_field = test_fixture.getTestFieldData<EvalType>(
                eval->_vector_field[dim]);
            EXPECT_EQ(exp_vec_field[dim], fieldValue(calc_vec_field, 0, qp));
        }
    }
}

//-----------------------------------------------------------------//
TEST(ConstantVectorField2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(ConstantVectorField2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(ConstantVectorField3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(ConstantVectorField3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
