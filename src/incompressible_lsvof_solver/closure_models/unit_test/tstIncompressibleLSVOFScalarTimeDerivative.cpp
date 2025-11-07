#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFScalarTimeDerivative.hpp"

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

    double _dof;
    double _dxdt_dof;
    double _rho;
    double _drho_dt;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> dof;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> dxdt_dof;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> rho;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> drho_dt;

    Dependencies(const panzer::IntegrationRule& ir,
                 const double dof_val,
                 const double dxdt_dof_val,
                 const double rho_val,
                 const double drho_dt_val)
        : _dof(dof_val)
        , _dxdt_dof(dxdt_dof_val)
        , _rho(rho_val)
        , _drho_dt(drho_dt_val)
        , dof("dof", ir.dl_scalar)
        , dxdt_dof("DXDT_dof", ir.dl_scalar)
        , rho("density", ir.dl_scalar)
        , drho_dt("DXDT_density", ir.dl_scalar)
    {
        this->addEvaluatedField(dof);
        this->addEvaluatedField(dxdt_dof);
        this->addEvaluatedField(rho);
        this->addEvaluatedField(drho_dt);

        this->setName(
            "Incompressible LSVOF Scalar Time Derivative Unit "
            "Test "
            "Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData) override
    {
        dof.deep_copy(_dof);
        dxdt_dof.deep_copy(_dxdt_dof);
        rho.deep_copy(_rho);
        drho_dt.deep_copy(_drho_dt);
    }
};

template<class EvalType>
void testEval(const bool mass_weighted)
{
    const int num_space_dim = 3;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Set up dependencies
    const double rho = 0.25;
    const double dof = 0.5;
    const double dxdt_dof = 0.8;
    const double drho_dt = 1.2;

    auto deps = Teuchos::rcp(
        new Dependencies<EvalType>(ir, dof, dxdt_dof, rho, drho_dt));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize class object to test
    const auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleLSVOFScalarTimeDerivative<EvalType,
                                                                  panzer::Traits>(
            ir, "dof", "dof_eqn"));

    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_dqdt_dof);

    test_fixture.evaluate<EvalType>();

    const auto fc_dqdt_dof
        = test_fixture.getTestFieldData<EvalType>(eval->_dqdt_dof);

    const int num_point = ir.num_points;

    // Expected values
    const double exp_dqdt_dof = mass_weighted ? rho * dxdt_dof + drho_dt * dof
                                              : dxdt_dof;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_EQ(exp_dqdt_dof, fieldValue(fc_dqdt_dof, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(TestLSVOFScalarConvective, Residual)
{
    testEval<panzer::Traits::Residual>(false);
}

//-----------------------------------------------------------------//
TEST(TestLSVOFScalarConvective, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(false);
}

//-----------------------------------------------------------------//
TEST(TestLSVOFScalarConvectiveMassWeighted, Residual)
{
    testEval<panzer::Traits::Residual>(true);
}

//-----------------------------------------------------------------//
TEST(TestLSVOFScalarConvectiveMassWeighted, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(true);
}

} // namespace Test
} // namespace VertexCFD
