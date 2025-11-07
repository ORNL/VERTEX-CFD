#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleGradDiv.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
// Test data dependencies.
template<class EvalType>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        momentum_flux_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        momentum_flux_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        momentum_flux_2;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_2;

    Dependencies(const panzer::IntegrationRule& ir)
        : momentum_flux_0("VISCOUS_FLUX_momentum_0", ir.dl_vector)
        , momentum_flux_1("VISCOUS_FLUX_momentum_1", ir.dl_vector)
        , momentum_flux_2("VISCOUS_FLUX_momentum_2", ir.dl_vector)
        , grad_vel_0("GRAD_velocity_0", ir.dl_vector)
        , grad_vel_1("GRAD_velocity_1", ir.dl_vector)
        , grad_vel_2("GRAD_velocity_2", ir.dl_vector)
    {
        this->addEvaluatedField(momentum_flux_0);
        this->addEvaluatedField(momentum_flux_1);
        this->addEvaluatedField(momentum_flux_2);

        this->addEvaluatedField(grad_vel_0);
        this->addEvaluatedField(grad_vel_1);
        this->addEvaluatedField(grad_vel_2);

        this->setName("Incompressible Grad Div Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "grad div test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = grad_vel_0.extent(1);
        const int num_space_dim = grad_vel_0.extent(2);
        using Kokkos::pow;
        for (int qp = 0; qp < num_point; ++qp)
        {
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                const int sign = pow(-1, dim + 1);
                const int dimqp = (dim + 1) * sign;
                grad_vel_0(c, qp, dim) = 0.25 * dimqp;
                grad_vel_1(c, qp, dim) = 0.5 * dimqp;
                grad_vel_2(c, qp, dim) = 0.125 * dimqp;

                momentum_flux_0(c, qp, dim) = 0.5 * dimqp;
                momentum_flux_1(c, qp, dim) = 1.0 * dimqp;
                momentum_flux_2(c, qp, dim) = 0.25 * dimqp;
            }
        }
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

    auto& ir = *test_fixture.ir;
    const double nan_val = std::numeric_limits<double>::quiet_NaN();

    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    Teuchos::ParameterList stability_param_list;
    stability_param_list.set("GradDiv Stabilization Coefficient", 0.5);

    auto eval = Teuchos::rcp(
        new ClosureModel::
            IncompressibleGradDiv<EvalType, panzer::Traits, num_space_dim>(
                ir, stability_param_list));
    test_fixture.registerEvaluator<EvalType>(eval);
    for (int dim = 0; dim < num_space_dim; ++dim)
        test_fixture.registerTestField<EvalType>(eval->_momentum_flux[dim]);

    test_fixture.evaluate<EvalType>();

    const double exp_mom_0_flux[3] = {num_space_dim == 3 ? -0.3125 : -0.125,
                                      1.0,
                                      num_space_dim == 3 ? -1.5 : nan_val};
    const double exp_mom_1_flux[3] = {-1.0,
                                      num_space_dim == 3 ? 2.1875 : 2.375,
                                      num_space_dim == 3 ? -3.0 : nan_val};
    const double exp_mom_2_flux[3] = {-0.25, 0.5, -0.5625};

    // Assert values
    const int num_point = ir.num_points;
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int dim = 0; dim < num_space_dim; dim++)
        {
            const auto fc_mom_0 = test_fixture.getTestFieldData<EvalType>(
                eval->_momentum_flux[0]);
            EXPECT_DOUBLE_EQ(exp_mom_0_flux[dim],
                             fieldValue(fc_mom_0, 0, qp, dim));
            const auto fc_mom_1 = test_fixture.getTestFieldData<EvalType>(
                eval->_momentum_flux[1]);
            EXPECT_DOUBLE_EQ(exp_mom_1_flux[dim],
                             fieldValue(fc_mom_1, 0, qp, dim));
            if (num_space_dim > 2) // 3D mesh
            {
                const auto fc_mom_2 = test_fixture.getTestFieldData<EvalType>(
                    eval->_momentum_flux[2]);
                EXPECT_DOUBLE_EQ(exp_mom_2_flux[dim],
                                 fieldValue(fc_mom_2, 0, qp, dim));
            }
        }
    }
}

//---------------------------------------------------------------------------//
TEST(IncompressibleGradDiv2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

//---------------------------------------------------------------------------//
TEST(IncompressibleGradDiv2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//---------------------------------------------------------------------------//
TEST(IncompressibleGradDiv3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

//---------------------------------------------------------------------------//
TEST(IncompressibleGradDiv3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Stability Parameters")
        .set("GradDiv Stabilization Coefficient", 0.1);
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.type_name = "IncompressibleGradDiv";
    test_fixture.eval_name = "Incompressible Grad-Div Source "
                             + std::to_string(num_space_dim) + "D";
    test_fixture.template buildAndTest<
        ClosureModel::IncompressibleGradDiv<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(IncompressibleGradDiv_Factory2D, Residual)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(IncompressibleGradDiv_Factory2D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

} // namespace Test
} // namespace VertexCFD
