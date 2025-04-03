#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "turbulence_models/boundary_conditions/VertexCFD_BoundaryState_TurbulenceKOmegaWallResolved.hpp"

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_config.hpp>

#include <mpi.h>

#include <iostream>

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> _grad_k;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> _grad_w;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> _normals;

    Dependencies(const panzer::IntegrationRule& ir)
        : _k("turb_kinetic_energy", ir.dl_scalar)
        , _grad_k("GRAD_turb_kinetic_energy", ir.dl_vector)
        , _grad_w("GRAD_turb_specific_dissipation_rate", ir.dl_vector)
        , _normals("Side Normal", ir.dl_vector)
    {
        this->addEvaluatedField(_k);
        this->addEvaluatedField(_grad_k);
        this->addEvaluatedField(_grad_w);
        this->addEvaluatedField(_normals);
        this->setName(
            "Turbulence Model K Omega Wall Resolved Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "k omega wall resolved test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = _k.extent(1);
        const int num_space_dim = _normals.extent(2);

        for (int qp = 0; qp < num_point; ++qp)
        {
            _k(c, qp) = 2.0;

            for (int d = 0; d < num_space_dim; ++d)
            {
                _grad_k(c, qp, d) = 0.02 * (d + 1.0);
                _grad_w(c, qp, d) = 0.07 * (d + 1.0);
                _normals(c, qp, d) = 0.33 * (d + 1.0);
            }
        }
    }
};

//---------------------------------------------------------------------------//
template<class EvalType>
void testEval(const int num_grad_dim, const double time)
{
    // Test fixture
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_grad_dim, integration_order, basis_order);

    // Create dependencies
    const auto dep_eval
        = Teuchos::rcp(new Dependencies<EvalType>(*test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Create dirichlet evaluator.
    Teuchos::ParameterList bc_params;
    bc_params.set("Omega Wall Value", 1.0e+5);
    bc_params.set("Omega Wall Initial Value", 1.0e+2);
    bc_params.set("Omega Ramp Time", 10.0);
    const auto wall_eval = Teuchos::rcp(
        new BoundaryCondition::TurbulenceKOmegaWallResolved<EvalType,
                                                            panzer::Traits>(
            *test_fixture.ir, bc_params));
    test_fixture.registerEvaluator<EvalType>(wall_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(wall_eval->_boundary_k);
    test_fixture.registerTestField<EvalType>(wall_eval->_boundary_w);
    test_fixture.registerTestField<EvalType>(wall_eval->_boundary_grad_k);
    test_fixture.registerTestField<EvalType>(wall_eval->_boundary_grad_w);

    // Set time
    test_fixture.setTime(time);

    // Evaluate values
    test_fixture.evaluate<EvalType>();

    // Set expected values
    const double nan_val = std::numeric_limits<double>::quiet_NaN();
    const double exp_k = 2.0;
    const double exp_w = time < 10.0 ? 3162.2776601683795 : 1e+5;
    const double exp_grad_k_2D[3] = {0.00911, 0.01822, nan_val};
    const double exp_grad_k_3D[3] = {
        -0.010492000000000001, -0.020984000000000003, -0.031476000000000004};
    const auto exp_grad_k = (num_grad_dim == 3) ? exp_grad_k_3D : exp_grad_k_2D;
    const double exp_grad_w[3]
        = {0.07, 0.14, num_grad_dim == 3 ? 0.21 : nan_val};

    // Check values
    const auto boundary_k_result
        = test_fixture.getTestFieldData<EvalType>(wall_eval->_boundary_k);
    const auto boundary_w_result
        = test_fixture.getTestFieldData<EvalType>(wall_eval->_boundary_w);
    const auto boundary_grad_k_result
        = test_fixture.getTestFieldData<EvalType>(wall_eval->_boundary_grad_k);
    const auto boundary_grad_w_result
        = test_fixture.getTestFieldData<EvalType>(wall_eval->_boundary_grad_w);

    const int num_point = boundary_k_result.extent(1);
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_k, fieldValue(boundary_k_result, 0, qp));
        EXPECT_DOUBLE_EQ(exp_w, fieldValue(boundary_w_result, 0, qp));

        for (int d = 0; d < num_grad_dim; ++d)
        {
            EXPECT_DOUBLE_EQ(exp_grad_k[d],
                             fieldValue(boundary_grad_k_result, 0, qp, d));
            EXPECT_DOUBLE_EQ(exp_grad_w[d],
                             fieldValue(boundary_grad_w_result, 0, qp, d));
        }
    }
}

//---------------------------------------------------------------------------//
// 2-D case w/ ramping
TEST(TestKOmegaWallResolvedRamp2D, Residual)
{
    testEval<panzer::Traits::Residual>(2, 5.0);
}

TEST(TestKOmegaWallResolvedRamp2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2, 5.0);
}

//---------------------------------------------------------------------------//
// 3-D case w/ ramping
TEST(TestKOmegaWallResolvedRamp3D, Residual)
{
    testEval<panzer::Traits::Residual>(3, 5.0);
}

TEST(TestKOmegaWallResolvedRamp3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(3, 5.0);
}

//---------------------------------------------------------------------------//
// 2-D case after ramping
TEST(TestKOmegaWallResolvedFinal2D, Residual)
{
    testEval<panzer::Traits::Residual>(2, 15.0);
}

TEST(TestKOmegaWallResolvedFinal2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(2, 15.0);
}

//---------------------------------------------------------------------------//
// 3-D case after ramping
TEST(TestKOmegaWallResolvedFinal3D, Residual)
{
    testEval<panzer::Traits::Residual>(3, 15.0);
}

TEST(TestKOmegaWallResolvedFinal3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(3, 15.0);
}

//---------------------------------------------------------------------------//
} // end namespace Test
} // end namespace VertexCFD
