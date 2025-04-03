#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleKOmegaSource.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
template<class EvalType, int NumSpaceDim>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;
    const double _nanval = std::numeric_limits<double>::quiet_NaN();

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_2;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> turb_eddy_viscosity;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> turb_kinetic_energy;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point>
        turb_specific_dissipation_rate;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_k;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_w;

    Dependencies(const panzer::IntegrationRule& ir)
        : grad_vel_0("GRAD_velocity_0", ir.dl_vector)
        , grad_vel_1("GRAD_velocity_1", ir.dl_vector)
        , grad_vel_2("GRAD_velocity_2", ir.dl_vector)
        , turb_eddy_viscosity("turbulent_eddy_viscosity", ir.dl_scalar)
        , turb_kinetic_energy("turb_kinetic_energy", ir.dl_scalar)
        , turb_specific_dissipation_rate("turb_specific_dissipation_rate",
                                         ir.dl_scalar)
        , grad_k("GRAD_turb_kinetic_energy", ir.dl_vector)
        , grad_w("GRAD_turb_specific_dissipation_rate", ir.dl_vector)
    {
        this->addEvaluatedField(grad_vel_0);
        this->addEvaluatedField(grad_vel_1);
        this->addEvaluatedField(grad_vel_2);
        this->addEvaluatedField(turb_eddy_viscosity);
        this->addEvaluatedField(turb_kinetic_energy);
        this->addEvaluatedField(turb_specific_dissipation_rate);
        this->addEvaluatedField(grad_k);
        this->addEvaluatedField(grad_w);
        this->setName("Incompressible K-Omega Source Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "K-Omega source test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = grad_vel_0.extent(1);
        const int num_space_dim = grad_vel_0.extent(2);
        using std::pow;

        for (int qp = 0; qp < num_point; ++qp)
        {
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                const int sign = pow(-1, dim + 1);
                const int dimqp = (dim + 1) * sign;
                grad_vel_0(c, qp, dim) = 0.250 * dimqp;
                grad_vel_1(c, qp, dim) = 0.500 * dimqp;
                grad_vel_2(c, qp, dim) = num_space_dim == 3 ? 0.125 * dimqp
                                                            : _nanval;

                grad_k(c, qp, dim) = 0.750 * dimqp;
                grad_w(c, qp, dim) = 1.250 * dimqp;
            }

            turb_eddy_viscosity(c, qp) = 1.1;
            turb_kinetic_energy(c, qp) = 0.1;
            turb_specific_dissipation_rate(c, qp) = 3.3;
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval(const bool limited)
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Create parameter list for user-defined constants
    Teuchos::ParameterList user_params;
    user_params.sublist("Turbulence Parameters")
        .set<bool>("Limit Production Term", limited);

    // Eval dependencies
    const auto deps = Teuchos::rcp(new Dependencies<EvalType, NumSpaceDim>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize and register
    auto eval = Teuchos::rcp(
        new ClosureModel::
            IncompressibleKOmegaSource<EvalType, panzer::Traits, NumSpaceDim>(
                ir, user_params));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_k_source);
    test_fixture.registerTestField<EvalType>(eval->_w_source);
    test_fixture.evaluate<EvalType>();

    // Evaluate test fields
    const auto fv_k_source
        = test_fixture.getTestFieldData<EvalType>(eval->_k_source);
    const auto fv_w_source
        = test_fixture.getTestFieldData<EvalType>(eval->_w_source);

    // Expected values
    double exp_k_source = num_space_dim == 3 ? 5.1781125 : 2.3078;
    const double exp_w_source = num_space_dim == 3 ? 89.20778886166762
                                                   : 39.63353525141912;
    if (limited)
    {
        exp_k_source = 0.5643;
    }

    // Assert values
    const int num_point = ir.num_points;
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_k_source, fieldValue(fv_k_source, 0, qp));
        EXPECT_DOUBLE_EQ(exp_w_source, fieldValue(fv_w_source, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleKOmegaSource2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>(false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleKOmegaSource2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>(false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleKOmegaSource3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>(false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleKOmegaSource3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>(false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleKOmegaSourceLimited2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>(true);
}

//-----------------------------------------------------------------//
TEST(IncompressibleKOmegaSourceLimited2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>(true);
}

//-----------------------------------------------------------------//
TEST(IncompressibleKOmegaSourceLimited3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>(true);
}

//-----------------------------------------------------------------//
TEST(IncompressibleKOmegaSourceLimited3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>(true);
}
//-----------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD
