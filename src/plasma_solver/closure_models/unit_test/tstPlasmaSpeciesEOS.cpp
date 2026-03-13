#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "utils/VertexCFD_Utils_VelocityDim.hpp"
#include "utils/VertexCFD_Utils_VelocityLayout.hpp"

#include "plasma_solver/closure_models/VertexCFD_Closure_PlasmaSpeciesEOS.hpp"

#include <Panzer_Dimension.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>

#include <Teuchos_RCP.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
// Test data dependencies.
template<class EvalType>
struct Dependencies : public PHX::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> nd;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, VelocityDim> vel;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> T;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> dxdt_nd;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, VelocityDim> dxdt_vel;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> dxdt_T;

    Dependencies(const panzer::IntegrationRule& ir)
        : nd("fast_particle_number_density", ir.dl_scalar)
        , vel("fast_particle_velocity",
              Utils::buildVelocityLayout(ir.dl_scalar, ir.spatial_dimension))
        , T("fast_particle_temperature", ir.dl_scalar)
        , dxdt_nd("DXDT_fast_particle_number_density", ir.dl_scalar)
        , dxdt_vel(
              "DXDT_fast_particle_velocity",
              Utils::buildVelocityLayout(ir.dl_scalar, ir.spatial_dimension))
        , dxdt_T("DXDT_fast_particle_temperature", ir.dl_scalar)
    {
        this->addEvaluatedField(nd);
        this->addEvaluatedField(vel);
        this->addEvaluatedField(T);
        this->addEvaluatedField(dxdt_nd);
        this->addEvaluatedField(dxdt_vel);
        this->addEvaluatedField(dxdt_T);

        this->setName("Plasma Species EOS Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        nd.deep_copy(1.5);
        T.deep_copy(2.5);
        dxdt_nd.deep_copy(-0.5);
        dxdt_T.deep_copy(-3.0);
        Kokkos::parallel_for(
            "test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = vel.extent(1);
        const int vel_dim = vel.extent(2);
        for (int qp = 0; qp < num_point; ++qp)
        {
            for (int d = 0; d < vel_dim; ++d)
            {
                vel(c, qp, d) = 0.25 * (d + 1);
                dxdt_vel(c, qp, d) = -1.5 * (d + 1);
            }
        }
    }
};

//---------------------------------------------------------------------------//
template<class EvalType>
void testEval(const int num_space_dim)
{
    // Setup test fixture.
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Eval dependencies.
    const auto dep_eval
        = Teuchos::rcp(new Dependencies<EvalType>(*test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Species properties
    Teuchos::ParameterList closure_params;
    closure_params.set("Species Name", "fast_particle");
    closure_params.set("Species Mass", 0.2);

    // Create test evaluator.
    const auto eos_eval = Teuchos::rcp(
        new ClosureModel::PlasmaSpeciesEOS<EvalType, panzer::Traits>(
            *test_fixture.ir, closure_params));
    test_fixture.registerEvaluator<EvalType>(eos_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(eos_eval->_rho);
    test_fixture.registerTestField<EvalType>(eos_eval->_drhodt);
    test_fixture.registerTestField<EvalType>(eos_eval->_p);
    test_fixture.registerTestField<EvalType>(eos_eval->_E);
    test_fixture.registerTestField<EvalType>(eos_eval->_dEdt);

    // Evaluate test fields.
    test_fixture.evaluate<EvalType>();

    // Momentum density
    const auto rho = test_fixture.getTestFieldData<EvalType>(eos_eval->_rho);
    const double exp_rho = 0.30000000000000004;
    const auto drhodt
        = test_fixture.getTestFieldData<EvalType>(eos_eval->_drhodt);
    const double exp_drhodt = -0.1;

    // Pressure
    const auto p = test_fixture.getTestFieldData<EvalType>(eos_eval->_p);
    const double exp_p = 3.75;

    // Energy density
    const auto E = test_fixture.getTestFieldData<EvalType>(eos_eval->_E);
    const auto dEdt = test_fixture.getTestFieldData<EvalType>(eos_eval->_dEdt);
    double exp_E, exp_dEdt;
    if (num_space_dim == 3)
    {
        exp_E = 5.75625;
        exp_dEdt = -10.243749999999999;
    }
    else
    {
        exp_E = 5.671875;
        exp_dEdt = -9.203125;
    }

    // Assert values (assume one quadrature point)
    EXPECT_DOUBLE_EQ(exp_rho, fieldValue(rho, 0, 0));
    EXPECT_DOUBLE_EQ(exp_drhodt, fieldValue(drhodt, 0, 0));
    EXPECT_DOUBLE_EQ(exp_p, fieldValue(p, 0, 0));
    EXPECT_DOUBLE_EQ(exp_E, fieldValue(E, 0, 0));
    EXPECT_DOUBLE_EQ(exp_dEdt, fieldValue(dEdt, 0, 0));

} // namespace Test

//---------------------------------------------------------------------------//
TEST(PlasmaSpeciesEOS2D, Residual)
{
    testEval<panzer::Traits::Residual>(2);
}

//---------------------------------------------------------------------------//
TEST(PlasmaSpeciesEOS2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2);
}

//---------------------------------------------------------------------------//
TEST(PlasmaSpeciesEOS3D, Residual)
{
    testEval<panzer::Traits::Residual>(3);
}

//---------------------------------------------------------------------------//
TEST(PlasmaSpeciesEOS3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(3);
}

//---------------------------------------------------------------------------//

} // namespace Test
} // end namespace VertexCFD
