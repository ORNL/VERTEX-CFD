#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleViscousHeat.hpp"

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

    // quiet_NaN is a host-side function so we store the value
    const double _nanval = std::numeric_limits<double>::quiet_NaN();

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> rho;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> nu;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_2;

    Dependencies(const panzer::IntegrationRule& ir)
        : rho("density", ir.dl_scalar)
        , nu("kinematic_viscosity", ir.dl_scalar)
        , grad_vel_0("GRAD_velocity_0", ir.dl_vector)
        , grad_vel_1("GRAD_velocity_1", ir.dl_vector)
        , grad_vel_2("GRAD_velocity_2", ir.dl_vector)

    {
        this->addEvaluatedField(rho);
        this->addEvaluatedField(nu);
        this->addEvaluatedField(grad_vel_0);
        this->addEvaluatedField(grad_vel_1);
        this->addEvaluatedField(grad_vel_2);

        this->setName("Incompressible Viscous Heat Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "viscous heat test dependencies",
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
            rho(c, qp) = 1.0;
            nu(c, qp) = 0.375;
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                const int sign = pow(-1, dim + 1);
                const int dimqp = (dim + 1) * sign;
                grad_vel_0(c, qp, dim) = 0.250 * dimqp;
                grad_vel_1(c, qp, dim) = 0.500 * dimqp;
                grad_vel_2(c, qp, dim) = num_space_dim == 3 ? 0.125 * dimqp
                                                            : _nanval;
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

    const auto& ir = *test_fixture.ir;

    // Initialize velocity gradient and dependents
    auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize class object to test
    auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleViscousHeat<EvalType,
                                                    panzer::Traits,
                                                    num_space_dim>(ir));

    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(
        eval->_viscous_heat_continuity_source);
    test_fixture.registerTestField<EvalType>(eval->_viscous_heat_energy_source);
    for (int dim = 0; dim < num_space_dim; ++dim)
        test_fixture.registerTestField<EvalType>(
            eval->_viscous_heat_momentum_source[dim]);

    test_fixture.evaluate<EvalType>();

    const auto fc_cont = test_fixture.getTestFieldData<EvalType>(
        eval->_viscous_heat_continuity_source);
    const auto fc_mom_0 = test_fixture.getTestFieldData<EvalType>(
        eval->_viscous_heat_momentum_source[0]);
    const auto fc_mom_1 = test_fixture.getTestFieldData<EvalType>(
        eval->_viscous_heat_momentum_source[1]);
    const auto fc_energy = test_fixture.getTestFieldData<EvalType>(
        eval->_viscous_heat_energy_source);

    const int num_point = ir.num_points;

    // Expected energy - see ./doc/viscous_heat_reference.py
    const double exp_energy = num_space_dim == 3 ? 1.775390625 : 0.796875;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_EQ(0.0, fieldValue(fc_cont, 0, qp));
        EXPECT_EQ(0.0, fieldValue(fc_mom_0, 0, qp));
        EXPECT_EQ(0.0, fieldValue(fc_mom_1, 0, qp));
        if (num_space_dim == 3)
        {
            const auto fc_mom_2 = test_fixture.getTestFieldData<EvalType>(
                eval->_viscous_heat_momentum_source[2]);
            EXPECT_EQ(0.0, fieldValue(fc_mom_2, 0, qp));
        }
        EXPECT_EQ(exp_energy, fieldValue(fc_energy, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleViscousHeat2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleViscousHeat2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleViscousHeat3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleViscousHeat3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.type_name = "IncompressibleViscousHeat";
    test_fixture.eval_name = "Incompressible Viscous Heat "
                             + std::to_string(num_space_dim) + "D";
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.template buildAndTest<
        ClosureModel::IncompressibleViscousHeat<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(IncompressibleViscousHeat_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(IncompressibleViscousHeat_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(IncompressibleViscousHeat_Factory3D, residual_test)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(IncompressibleViscousHeat_Factory3D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
