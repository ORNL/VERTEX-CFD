#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleShearVariables.hpp"

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_2;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> rho;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> nu;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> normals;

    Dependencies(const panzer::IntegrationRule& ir)
        : grad_vel_0("GRAD_velocity_0", ir.dl_vector)
        , grad_vel_1("GRAD_velocity_1", ir.dl_vector)
        , grad_vel_2("GRAD_velocity_2", ir.dl_vector)
        , rho("density", ir.dl_scalar)
        , nu("kinematic_viscosity", ir.dl_scalar)
        , normals("Side Normal", ir.dl_vector)
    {
        this->addEvaluatedField(grad_vel_0);
        this->addEvaluatedField(grad_vel_1);
        this->addEvaluatedField(grad_vel_2);
        this->addEvaluatedField(rho);
        this->addEvaluatedField(nu);
        this->addEvaluatedField(normals);

        this->setName("Incompressible Shear Variables Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "shear variables test dependencies",
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
            rho(c, qp) = 1.5;
            nu(c, qp) = 0.375;
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                const int sign = pow(-1, dim + 1);
                const int dimqp = (dim + 1) * sign;
                grad_vel_0(c, qp, dim) = 0.250 * dimqp;
                grad_vel_1(c, qp, dim) = 0.500 * dimqp;
                grad_vel_2(c, qp, dim) = num_space_dim == 3 ? 0.725 * dimqp
                                                            : _nanval;
                normals(c, qp, dim) = 0.125 * dimqp;
            }
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval()
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Initialize velocity gradient and dependents
    auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize class object to test
    auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleShearVariables<EvalType,
                                                       panzer::Traits,
                                                       num_space_dim>(ir));

    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_tau_w);
    test_fixture.registerTestField<EvalType>(eval->_u_tau);

    test_fixture.evaluate<EvalType>();

    const auto tau_w = test_fixture.getTestFieldData<EvalType>(eval->_tau_w);
    const auto u_tau = test_fixture.getTestFieldData<EvalType>(eval->_u_tau);

    const int num_point = ir.num_points;

    // Expected values
    const double exp_tau_w = num_space_dim == 3 ? 0.9011871138178397
                                                : 0.19652941208494246;
    const double exp_u_tau = num_space_dim == 3 ? 0.7751073533465498
                                                : 0.3619663079025841;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_tau_w, fieldValue(tau_w, 0, qp));
        EXPECT_DOUBLE_EQ(exp_u_tau, fieldValue(u_tau, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleShearVariables2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleShearVariables2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleShearVariables3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleShearVariables3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.type_name = "IncompressibleShearVariables";
    test_fixture.eval_name = "Incompressible Shear/Friction Variables "
                             + std::to_string(num_space_dim) + "D";
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.template buildAndTest<
        ClosureModel::IncompressibleShearVariables<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(IncompressibleShearVariables_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(IncompressibleShearVariables_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(IncompressibleShearVariables_Factory3D, residual_test)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(IncompressibleShearVariables_Factory3D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
