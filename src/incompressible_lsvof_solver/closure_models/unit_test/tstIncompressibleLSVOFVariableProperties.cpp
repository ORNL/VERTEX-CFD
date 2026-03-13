#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "utils/VertexCFD_Utils_PhaseIndex.hpp"
#include "utils/VertexCFD_Utils_ScalarToVector.hpp"

#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFVariableProperties.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
// LSVOF model types
enum class LSVOFModel
{
    VOF,
    CLS
};
//---------------------------------------------------------------------------//
template<class EvalType>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    LSVOFModel _lsvof_model;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> alpha_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> alpha_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> alpha_2;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> dxdt_alpha_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> dxdt_alpha_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> dxdt_alpha_2;

    Dependencies(const panzer::IntegrationRule& ir,
                 const LSVOFModel& lsvof_model)
        : _lsvof_model(lsvof_model)
        , alpha_0(_lsvof_model == LSVOFModel::VOF ? "ph0" : "CLS_heaviside",
                  ir.dl_scalar)
        , alpha_1("ph1", ir.dl_scalar)
        , alpha_2("ph2", ir.dl_scalar)
        , dxdt_alpha_0("DXDT_ph0", ir.dl_scalar)
        , dxdt_alpha_1("DXDT_ph1", ir.dl_scalar)
        , dxdt_alpha_2("DXDT_ph2", ir.dl_scalar)
    {
        this->addEvaluatedField(alpha_0);
        if (lsvof_model == LSVOFModel::VOF)
        {
            this->addEvaluatedField(alpha_1);
            this->addEvaluatedField(alpha_2);
            this->addEvaluatedField(dxdt_alpha_0);
            this->addEvaluatedField(dxdt_alpha_1);
            this->addEvaluatedField(dxdt_alpha_2);
        }

        this->setName(
            "Incompressible LSVOF Variable Properties Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData) override
    {
        alpha_0.deep_copy(0.4);

        if (_lsvof_model == LSVOFModel::VOF)
        {
            alpha_1.deep_copy(0.5);
            alpha_2.deep_copy(0.1);

            dxdt_alpha_0.deep_copy(0.1);
            dxdt_alpha_1.deep_copy(-0.4);
            dxdt_alpha_2.deep_copy(0.3);
        }
    }
};

template<class EvalType>
void testEval(const LSVOFModel& lsvof_model)
{
    constexpr int num_space_dim = 3;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Initialize phase properties and void fractions
    const int num_phases = lsvof_model == LSVOFModel::VOF ? 3 : 2;
    const double ph0_rho = 1.2;
    const double ph1_rho = 231.5;
    const double ph2_rho = 4719.8;
    const double ph0_mu = 1.79E-05;
    const double ph1_mu = 3.91E-03;
    const double ph2_mu = 7.11E-04;
    const double a0 = 0.4;
    const double a1 = 0.5;
    const double a2 = 0.1;
    const double da0dt = 0.1;
    const double da1dt = -0.4;
    const double da2dt = 0.3;

    auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir, lsvof_model));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Phase list
    std::vector<std::string> phase_list;
    phase_list.push_back("ph0");
    phase_list.push_back("ph1");

    // Create combined array of phase fraction fields
    if (lsvof_model == LSVOFModel::VOF)
    {
        const auto phase_vec
            = Utils::ScalarToVector<EvalType, PhaseIndex>::createFromList(
                ir, "volume_fractions", phase_list, true, false);
        test_fixture.registerEvaluator<EvalType>(phase_vec);

        phase_list.push_back("ph2");
    }

    const std::string lsvof_model_name
        = lsvof_model == LSVOFModel::VOF ? "VOF" : "CLS";

    Teuchos::ParameterList lsvof_params;
    lsvof_params.set("LSVOF Model", lsvof_model_name);
    lsvof_params.set<int>("Number of Phases", num_phases);
    lsvof_params.sublist("Phase 1")
        .set("Phase name", "ph0")
        .set<double>("Density", ph0_rho)
        .set<double>("Dynamic viscosity", ph0_mu);
    lsvof_params.sublist("Phase 2")
        .set("Phase name", "ph1")
        .set<double>("Density", ph1_rho)
        .set<double>("Dynamic viscosity", ph1_mu);
    if (lsvof_model == LSVOFModel::VOF)
    {
        lsvof_params.sublist("Phase 3")
            .set("Phase name", "ph2")
            .set<double>("Density", ph2_rho)
            .set<double>("Dynamic viscosity", ph2_mu);
    }

    // Initialize class object to test
    const auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleLSVOFVariableProperties<EvalType,
                                                                panzer::Traits>(
            ir, lsvof_params, phase_list));

    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_rho);
    test_fixture.registerTestField<EvalType>(eval->_mu);
    if (lsvof_model == LSVOFModel::VOF)
        test_fixture.registerTestField<EvalType>(eval->_dxdt_rho);

    test_fixture.evaluate<EvalType>();

    const auto fc_rho = test_fixture.getTestFieldData<EvalType>(eval->_rho);
    const auto fc_mu = test_fixture.getTestFieldData<EvalType>(eval->_mu);

    const int num_point = ir.num_points;

    // Expected values
    double exp_rho = ph0_rho * a0 + ph1_rho * a1 + ph2_rho * a2;
    double exp_mu = ph0_mu * a0 + ph1_mu * a1 + ph2_mu * a2;

    if (lsvof_model == LSVOFModel::CLS)
    {
        exp_rho = ph0_rho * a0 + ph1_rho * (1.0 - a0);
        exp_mu = ph0_mu * a0 + ph1_mu * (1.0 - a0);
    }

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_EQ(exp_rho, fieldValue(fc_rho, 0, qp));
        EXPECT_EQ(exp_mu, fieldValue(fc_mu, 0, qp));

        if (lsvof_model == LSVOFModel::VOF)
        {
            const auto fc_dxdt_rho
                = test_fixture.getTestFieldData<EvalType>(eval->_dxdt_rho);

            const double exp_dxdt_rho = da0dt * ph0_rho + da1dt * ph1_rho
                                        + da2dt * ph2_rho;

            EXPECT_EQ(exp_dxdt_rho, fieldValue(fc_dxdt_rho, 0, qp));
        }
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleLSVOFVariableProperties, Residual)
{
    testEval<panzer::Traits::Residual>(LSVOFModel::VOF);
}

//-----------------------------------------------------------------//
TEST(IncompressibleLSVOFVariableProperties, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(LSVOFModel::VOF);
}

//-----------------------------------------------------------------//
TEST(IncompressibleCLSVariableProperties, Residual)
{
    testEval<panzer::Traits::Residual>(LSVOFModel::CLS);
}

//-----------------------------------------------------------------//
TEST(IncompressibleCLSVariableProperties, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(LSVOFModel::CLS);
}

} // namespace Test
} // namespace VertexCFD
