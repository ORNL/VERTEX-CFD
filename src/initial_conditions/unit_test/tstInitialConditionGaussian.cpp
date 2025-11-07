#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "initial_conditions/VertexCFD_InitialCondition_Gaussian.hpp"
#include "initial_conditions/VertexCFD_InitialCondition_InverseGaussian.hpp"

#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testGaussian(const bool scaling = false)
{
    const int integration_order = 1;
    const int basis_order = 1;
    constexpr int num_space_dim = NumSpaceDim;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    Teuchos::ParameterList params;
    Teuchos::Array<double> center(num_space_dim);
    center[0] = 1.0 / 3.0;
    center[1] = 0.5;
    if (num_space_dim > 2)
        center[2] = 2.0 / 3.0;
    params.set<Teuchos::Array<double>>("Center", center);
    if (scaling)
    {
        params.set<std::string>("Scaling", "scaled");
        params.set<double>("Scale", 1.0);
    }

    Teuchos::Array<double> sigma(num_space_dim);
    sigma[0] = 0.5;
    sigma[1] = 2.0;
    if (num_space_dim > 2)
        sigma[2] = 3.0;
    params.set<Teuchos::Array<double>>("Sigma", sigma);
    const double base = 1.0 / 3.0;
    params.set<double>("Base", base);
    params.set<std::string>("Equation Set Name", "dof");
    auto eval = Teuchos::rcp(
        new InitialCondition::Gaussian<EvalType, panzer::Traits, num_space_dim>(
            params, *test_fixture.basis_ir_layout->getBasis()));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_ic);

    test_fixture.evaluate<EvalType>();

    auto ic = test_fixture.getTestFieldData<EvalType>(eval->_ic);

    // Number of degree of freedom based on `num_space_dim` value
    int num_dofs = 4;
    if (num_space_dim == 3)
        num_dofs = 8;
    EXPECT_EQ(num_dofs, ic.extent(1));

    if (scaling)
    {
        if (num_space_dim == 2)
        {
            EXPECT_DOUBLE_EQ(1.1094346363285787, fieldValue(ic, 0, 0));
            EXPECT_DOUBLE_EQ(0.73179702839459293, fieldValue(ic, 0, 1));
            EXPECT_DOUBLE_EQ(0.73179702839459293, fieldValue(ic, 0, 2));
            EXPECT_DOUBLE_EQ(1.1094346363285787, fieldValue(ic, 0, 3));
        }
        else if (num_space_dim == 3)
        {
            EXPECT_DOUBLE_EQ(1.0905062861781287, fieldValue(ic, 0, 0));
            EXPECT_DOUBLE_EQ(0.72207888939230869, fieldValue(ic, 0, 1));
            EXPECT_DOUBLE_EQ(0.72207888939230869, fieldValue(ic, 0, 2));
            EXPECT_DOUBLE_EQ(1.0905062861781287, fieldValue(ic, 0, 3));
            EXPECT_DOUBLE_EQ(1.104658643428823, fieldValue(ic, 0, 4));
            EXPECT_DOUBLE_EQ(0.72934495187948012, fieldValue(ic, 0, 5));
            EXPECT_DOUBLE_EQ(0.72934495187948012, fieldValue(ic, 0, 6));
            EXPECT_DOUBLE_EQ(1.104658643428823, fieldValue(ic, 0, 7));
        }
    }
    else
    {
        if (num_space_dim == 2)
        {
            EXPECT_DOUBLE_EQ(0.4568536920450875, fieldValue(ic, 0, 0));
            EXPECT_DOUBLE_EQ(0.3967508000449945, fieldValue(ic, 0, 1));
            EXPECT_DOUBLE_EQ(0.3967508000449945, fieldValue(ic, 0, 2));
            EXPECT_DOUBLE_EQ(0.4568536920450875, fieldValue(ic, 0, 3));
        }
        else if (num_space_dim == 3)
        {
            EXPECT_DOUBLE_EQ(0.3493585546023939, fieldValue(ic, 0, 0));
            EXPECT_DOUBLE_EQ(0.3415609562691542, fieldValue(ic, 0, 1));
            EXPECT_DOUBLE_EQ(0.3415609562691542, fieldValue(ic, 0, 2));
            EXPECT_DOUBLE_EQ(0.3493585546023939, fieldValue(ic, 0, 3));
            EXPECT_DOUBLE_EQ(0.3496580828086895, fieldValue(ic, 0, 4));
            EXPECT_DOUBLE_EQ(0.3417147391778995, fieldValue(ic, 0, 5));
            EXPECT_DOUBLE_EQ(0.3417147391778995, fieldValue(ic, 0, 6));
            EXPECT_DOUBLE_EQ(0.3496580828086895, fieldValue(ic, 0, 7));
        }
    }
}

//---------------------------------------------------------------------------//
TEST(Gaussian2D, residual)
{
    testGaussian<panzer::Traits::Residual, 2>();
}

//---------------------------------------------------------------------------//
TEST(Gaussian2D, jacobian)
{
    testGaussian<panzer::Traits::Jacobian, 2>();
}

//---------------------------------------------------------------------------//
TEST(Gaussian3D, residual)
{
    testGaussian<panzer::Traits::Residual, 3>();
}

//---------------------------------------------------------------------------//
TEST(Gaussian3D, jacobian)
{
    testGaussian<panzer::Traits::Jacobian, 3>();
}

//---------------------------------------------------------------------------//
TEST(ScaledGaussian2D, residual)
{
    testGaussian<panzer::Traits::Residual, 2>(true);
}

//---------------------------------------------------------------------------//
TEST(ScaledGaussian2D, jacobian)
{
    testGaussian<panzer::Traits::Jacobian, 2>(true);
}

//---------------------------------------------------------------------------//
TEST(ScaledGaussian3D, residual)
{
    testGaussian<panzer::Traits::Residual, 3>(true);
}

//---------------------------------------------------------------------------//
TEST(ScaledGaussian3D, jacobian)
{
    testGaussian<panzer::Traits::Jacobian, 3>(true);
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testInverseGaussian()
{
    const int integration_order = 1;
    const int basis_order = 1;
    constexpr int num_space_dim = NumSpaceDim;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    Teuchos::ParameterList params;
    Teuchos::Array<double> center(num_space_dim);
    center[0] = 1.0 / 3.0;
    center[1] = 0.5;
    if (num_space_dim > 2)
        center[2] = 2.0 / 3.0;
    params.set<Teuchos::Array<double>>("Center", center);
    Teuchos::Array<double> sigma(num_space_dim);
    sigma[0] = 0.5;
    sigma[1] = 2.0;
    if (num_space_dim > 2)
        sigma[2] = 3.0;
    params.set<Teuchos::Array<double>>("Sigma", sigma);
    const double base = 1.0 / 3.0;
    params.set<double>("Base", base);
    params.set<std::string>("Equation Set Name", "dof");
    auto eval = Teuchos::rcp(
        new InitialCondition::InverseGaussian<EvalType, panzer::Traits, num_space_dim>(
            params, *test_fixture.basis_ir_layout->getBasis()));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_ic);

    test_fixture.evaluate<EvalType>();

    auto ic = test_fixture.getTestFieldData<EvalType>(eval->_ic);

    // Number of degree of freedom based on `num_space_dim` value
    int num_dofs = 4;
    if (num_space_dim == 3)
        num_dofs = 8;
    EXPECT_EQ(num_dofs, ic.extent(1));

    if (num_space_dim == 2)
    {
        EXPECT_DOUBLE_EQ(2.1888845759865475, fieldValue(ic, 0, 0));
        EXPECT_DOUBLE_EQ(2.5204738084626235, fieldValue(ic, 0, 1));
        EXPECT_DOUBLE_EQ(2.5204738084626235, fieldValue(ic, 0, 2));
        EXPECT_DOUBLE_EQ(2.1888845759865475, fieldValue(ic, 0, 3));
    }
    else if (num_space_dim == 3)
    {
        EXPECT_DOUBLE_EQ(2.8623887602755378, fieldValue(ic, 0, 0));
        EXPECT_DOUBLE_EQ(2.9277350986568493, fieldValue(ic, 0, 1));
        EXPECT_DOUBLE_EQ(2.9277350986568493, fieldValue(ic, 0, 2));
        EXPECT_DOUBLE_EQ(2.8623887602755378, fieldValue(ic, 0, 3));
        EXPECT_DOUBLE_EQ(2.8599367472569936, fieldValue(ic, 0, 4));
        EXPECT_DOUBLE_EQ(2.9264175212512321, fieldValue(ic, 0, 5));
        EXPECT_DOUBLE_EQ(2.9264175212512321, fieldValue(ic, 0, 6));
        EXPECT_DOUBLE_EQ(2.8599367472569936, fieldValue(ic, 0, 7));
    }
}

//---------------------------------------------------------------------------//
TEST(InverseGaussian2D, residual)
{
    testInverseGaussian<panzer::Traits::Residual, 2>();
}

//---------------------------------------------------------------------------//
TEST(InverseGaussian2D, jacobian)
{
    testInverseGaussian<panzer::Traits::Jacobian, 2>();
}

//---------------------------------------------------------------------------//
TEST(InverseGaussian3D, residual)
{
    testInverseGaussian<panzer::Traits::Residual, 3>();
}

//---------------------------------------------------------------------------//
TEST(InverseGaussian3D, jacobian)
{
    testInverseGaussian<panzer::Traits::Jacobian, 3>();
}

//---------------------------------------------------------------------------//

} // namespace Test
} // namespace VertexCFD
