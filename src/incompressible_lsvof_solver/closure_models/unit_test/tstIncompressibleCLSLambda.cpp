#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleCLSLambda.hpp"

#include <Panzer_GlobalData.hpp>
#include <Panzer_ParameterLibrary.hpp>
#include <Panzer_ScalarParameterEntry.hpp>
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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> phi;
    PHX::MDField<double, panzer::Cell, panzer::Point> epsilon;

    Dependencies(const panzer::IntegrationRule& ir)
        : phi("level_set", ir.dl_scalar)
        , epsilon("CLS_epsilon", ir.dl_scalar)
    {
        this->addEvaluatedField(phi);
        this->addEvaluatedField(epsilon);

        this->setName("Incompressible CLS Lambda Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData) override
    {
        phi.deep_copy(0.75);
        epsilon.deep_copy(1.5);
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

    // Set up required responses
    auto global_data = panzer::createGlobalData();
    auto& pl = *global_data->pl;

    pl.addParameterFamily(
        "Level Set Magnitude - level_set_magnitude", true, false);
    Teuchos::RCP<panzer::ScalarParameterEntry<EvalType>> level_set_mag;
    level_set_mag = Teuchos::rcp(new panzer::ScalarParameterEntry<EvalType>);
    level_set_mag->setValue(2.5);
    pl.addEntry<EvalType>("Level Set Magnitude - level_set_magnitude",
                          level_set_mag);

    pl.addParameterFamily("Max Dphi - d_level_set", true, false);
    Teuchos::RCP<panzer::ScalarParameterEntry<EvalType>> max_d_phi;
    max_d_phi = Teuchos::rcp(new panzer::ScalarParameterEntry<EvalType>);
    max_d_phi->setValue(1.25);
    pl.addEntry<EvalType>("Max Dphi - d_level_set", max_d_phi);

    // Set time step size
    test_fixture.setStepSize(1.8);

    // Parameter list
    Teuchos::ParameterList params;
    params.set("Domain Volume", 1.2);
    params.set("Regularization Coefficient", 5.0);
    params.set("Interface Thickness Coefficient", 1.2);

    // Initialize class object to test
    const auto eval = Teuchos::rcp(
        new ClosureModel::
            IncompressibleCLSLambda<EvalType, panzer::Traits, num_space_dim>(
                ir, params, global_data));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_lambda);
    test_fixture.registerTestField<EvalType>(eval->_mag_phi);
    test_fixture.registerTestField<EvalType>(eval->_d_phi);

    test_fixture.evaluate<EvalType>();

    const auto fc_lambda
        = test_fixture.getTestFieldData<EvalType>(eval->_lambda);
    const auto fc_mag_phi
        = test_fixture.getTestFieldData<EvalType>(eval->_mag_phi);
    const auto fc_d_phi = test_fixture.getTestFieldData<EvalType>(eval->_d_phi);

    const int num_point = ir.num_points;

    // Expected values
    const double exp_mag_phi = 0.75;
    const double exp_d_phi = 2.5 / 1.2 - 0.75;
    double exp_lambda = (1.5 / (1.2 * 1.8)) * (5.0 * 1.5 / (1.2 * 1.25));

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_lambda, fieldValue(fc_lambda, 0, qp));
        EXPECT_DOUBLE_EQ(exp_mag_phi, fieldValue(fc_mag_phi, 0, qp));
        EXPECT_DOUBLE_EQ(exp_d_phi, fieldValue(fc_d_phi, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleCLSLambda2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleCLSLambda2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleCLSLambda3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleCLSLambda3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//-----------------------------------------------------------------//

} // namespace Test
} // namespace VertexCFD
