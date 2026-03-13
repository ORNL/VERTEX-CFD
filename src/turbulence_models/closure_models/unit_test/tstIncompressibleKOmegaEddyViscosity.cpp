#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleKOmegaEddyViscosity.hpp"

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
    const double _nanval = std::numeric_limits<double>::quiet_NaN();

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_2;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> turb_kinetic_energy;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point>
        turb_specific_dissipation_rate;

    const bool _limited;

    Dependencies(const panzer::IntegrationRule& ir, const bool limited)
        : grad_vel_0("GRAD_velocity_0", ir.dl_vector)
        , grad_vel_1("GRAD_velocity_1", ir.dl_vector)
        , grad_vel_2("GRAD_velocity_2", ir.dl_vector)
        , turb_kinetic_energy("turb_kinetic_energy", ir.dl_scalar)
        , turb_specific_dissipation_rate("turb_specific_dissipation_rate",
                                         ir.dl_scalar)
        , _limited(limited)
    {
        this->addEvaluatedField(grad_vel_0);
        this->addEvaluatedField(grad_vel_1);
        this->addEvaluatedField(grad_vel_2);
        this->addEvaluatedField(turb_kinetic_energy);
        this->addEvaluatedField(turb_specific_dissipation_rate);
        this->setName(
            "Incompressible K-Omega Eddy Viscosity Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "K-Omega eddy viscosity test dependencies",
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
            }

            turb_kinetic_energy(c, qp) = 2.5;
            turb_specific_dissipation_rate(c, qp) = _limited ? 1.0 : 10.0;
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

    // Create parameter list for user-defined constants
    Teuchos::ParameterList turb_params;
    turb_params.set<double>("C_lim", 0.88).set<double>("beta_star", 0.1);

    const auto& ir = *test_fixture.ir;

    // Eval dependencies
    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir, limited));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize and register
    auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleKOmegaEddyViscosity<EvalType,
                                                            panzer::Traits,
                                                            NumSpaceDim>(
            ir, turb_params));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_nu_t);
    test_fixture.evaluate<EvalType>();

    // Evaluate test fields
    const auto fv_nu_t = test_fixture.getTestFieldData<EvalType>(eval->_nu_t);

    // Expected values
    const int num_point = ir.num_points;
    double exp_var = 0.25;
    if (limited)
    {
        exp_var = num_space_dim == 3 ? 0.4128820565413407 : 0.6162797097120913;
    }

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_EQ(exp_var, fieldValue(fv_nu_t, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleKOmegaEddyViscosity2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>(false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleKOmegaEddyViscosity2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>(false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleKOmegaEddyViscosity3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>(false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleKOmegaEddyViscosity3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>(false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleKOmegaEddyViscosityLimited2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>(true);
}

//-----------------------------------------------------------------//
TEST(IncompressibleKOmegaEddyViscosityLimited2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>(true);
}

//-----------------------------------------------------------------//
TEST(IncompressibleKOmegaEddyViscosityLimited3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>(true);
}

//-----------------------------------------------------------------//
TEST(IncompressibleKOmegaEddyViscosityLimited3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>(true);
}
//-----------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD
