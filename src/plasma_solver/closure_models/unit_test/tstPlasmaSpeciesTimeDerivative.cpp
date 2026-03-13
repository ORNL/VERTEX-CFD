#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "utils/VertexCFD_Utils_VelocityDim.hpp"
#include "utils/VertexCFD_Utils_VelocityLayout.hpp"

#include "plasma_solver/closure_models/VertexCFD_Closure_PlasmaSpeciesTimeDerivative.hpp"

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> rho;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, VelocityDim> vel;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> dxdt_nd;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, VelocityDim> dxdt_vel;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> dvdt_rho;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> dvdt_E;

    Dependencies(const panzer::IntegrationRule& ir)
        : rho("fast_particle_momentum_density", ir.dl_scalar)
        , vel("fast_particle_velocity",
              Utils::buildVelocityLayout(ir.dl_scalar, ir.spatial_dimension))
        , dxdt_nd("DXDT_fast_particle_number_density", ir.dl_scalar)
        , dxdt_vel(
              "DXDT_fast_particle_velocity",
              Utils::buildVelocityLayout(ir.dl_scalar, ir.spatial_dimension))
        , dvdt_rho("DVDT_fast_particle_momentum_density", ir.dl_scalar)
        , dvdt_E("DVDT_fast_particle_energy", ir.dl_scalar)
    {
        this->addEvaluatedField(rho);
        this->addEvaluatedField(vel);
        this->addEvaluatedField(dxdt_nd);
        this->addEvaluatedField(dxdt_vel);
        this->addEvaluatedField(dvdt_rho);
        this->addEvaluatedField(dvdt_E);

        this->setName("Plasma Species Time Derivative Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        rho.deep_copy(2.0);
        dxdt_nd.deep_copy(-2.5);
        dvdt_rho.deep_copy(1.5);
        dvdt_E.deep_copy(3.5);
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
                vel(c, qp, d) = 4.0 * (d + 1);
                dxdt_vel(c, qp, d) = -1.5 * (d + 1);
            }
        }
    }
};

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval()
{
    // Setup test fixture.
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Eval dependencies.
    const auto dep_eval
        = Teuchos::rcp(new Dependencies<EvalType>(*test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Create test evaluator.
    const auto dqdt_eval = Teuchos::rcp(
        new ClosureModel::PlasmaSpeciesTimeDerivative<EvalType,
                                                      panzer::Traits,
                                                      num_space_dim>(
            *test_fixture.ir, "fast_particle"));
    test_fixture.registerEvaluator<EvalType>(dqdt_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(dqdt_eval->_dqdt_continuity);
    for (int dim = 0; dim < num_space_dim; dim++)
    {
        test_fixture.registerTestField<EvalType>(
            dqdt_eval->_dqdt_momentum[dim]);
    }
    test_fixture.registerTestField<EvalType>(dqdt_eval->_dqdt_energy);

    // Evaluate test fields.
    test_fixture.evaluate<EvalType>();

    // Number density equation
    const auto dqdt_nd
        = test_fixture.getTestFieldData<EvalType>(dqdt_eval->_dqdt_continuity);
    const double exp_dqdt_nd = -2.5;

    // Momentum equations
    const double exp_mom[3] = {-1.5 * 2.0 + 4.0 * 1.5,
                               -3.0 * 2.0 + 8.0 * 1.5,
                               num_space_dim > 2
                                   ? -4.5 * 2.0 + 12.0 * 1.5
                                   : std::numeric_limits<double>::quiet_NaN()};

    // Energy equation
    const auto dqdt_E
        = test_fixture.getTestFieldData<EvalType>(dqdt_eval->_dqdt_energy);
    const double exp_dqdt_E = 3.5;

    // Assert values (assume one quadrature point)
    EXPECT_DOUBLE_EQ(exp_dqdt_nd, fieldValue(dqdt_nd, 0, 0));
    for (int dim = 0; dim < num_space_dim; dim++)
    {
        const auto dqdt_momentum = test_fixture.getTestFieldData<EvalType>(
            dqdt_eval->_dqdt_momentum[dim]);
        EXPECT_DOUBLE_EQ(exp_mom[dim], fieldValue(dqdt_momentum, 0, 0));
    }
    EXPECT_DOUBLE_EQ(exp_dqdt_E, fieldValue(dqdt_E, 0, 0));
} // namespace Test

//---------------------------------------------------------------------------//
TEST(PlasmaSpeciesTimeDerivative2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

//---------------------------------------------------------------------------//
TEST(PlasmaSpeciesTimeDerivative2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//---------------------------------------------------------------------------//
TEST(PlasmaSpeciesTimeDerivative3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

//---------------------------------------------------------------------------//
TEST(PlasmaSpeciesTimeDerivative3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//---------------------------------------------------------------------------//

} // namespace Test
} // end namespace VertexCFD
