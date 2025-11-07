#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleEnstrophy.hpp"

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> nu;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_vel_2;

    Dependencies(const panzer::IntegrationRule& ir)
        : nu("kinematic_viscosity", ir.dl_scalar)
        , grad_vel_0("GRAD_velocity_0", ir.dl_vector)
        , grad_vel_1("GRAD_velocity_1", ir.dl_vector)
        , grad_vel_2("GRAD_velocity_2", ir.dl_vector)
    {
        this->addEvaluatedField(nu);
        this->addEvaluatedField(grad_vel_0);
        this->addEvaluatedField(grad_vel_1);
        this->addEvaluatedField(grad_vel_2);

        this->setName("Incompressible Enstrophy Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "enstrophy test dependencies",
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
            nu(c, qp) = 0.375;
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                const int sign = pow(-1, dim + 1);
                const int dimqp = (dim + 1) * sign;
                grad_vel_0(c, qp, dim) = 0.250 * dimqp;
                grad_vel_1(c, qp, dim) = 0.500 * dimqp;
                grad_vel_2(c, qp, dim) = num_space_dim == 3 ? 0.725 * dimqp
                                                            : _nanval;
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
        new ClosureModel::IncompressibleEnstrophy<EvalType,
                                                  panzer::Traits,
                                                  num_space_dim>(ir));

    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_vorticity);
    test_fixture.registerTestField<EvalType>(eval->_enstrophy);
    test_fixture.registerTestField<EvalType>(eval->_divergence);
    test_fixture.registerTestField<EvalType>(eval->_mag_divergence);

    test_fixture.evaluate<EvalType>();

    const auto vorticity
        = test_fixture.getTestFieldData<EvalType>(eval->_vorticity);
    const auto enstrophy
        = test_fixture.getTestFieldData<EvalType>(eval->_enstrophy);
    const auto divergence
        = test_fixture.getTestFieldData<EvalType>(eval->_divergence);
    const auto mag_divergence
        = test_fixture.getTestFieldData<EvalType>(eval->_mag_divergence);

    const int num_point = ir.num_points;

    // Expected values
    const double exp_vorticity_2d[2] = {-1.0, 0.0};
    const double exp_vorticity_3d[3] = {2.95, -0.025000000000000022, -1.0};
    const double* exp_vorticity = num_space_dim == 3 ? exp_vorticity_3d
                                                     : exp_vorticity_2d;
    const double exp_enstrophy = num_space_dim == 3 ? 3.638671875 : 0.375;
    const double exp_divergence = num_space_dim == 3 ? -1.425 : 0.75;
    const double exp_mag_divergence = num_space_dim == 3 ? 1.425 : 0.75;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int dim = 0; dim < num_space_dim; ++dim)
        {
            EXPECT_DOUBLE_EQ(exp_vorticity[dim],
                             fieldValue(vorticity, 0, qp, dim));
        }
        EXPECT_DOUBLE_EQ(exp_enstrophy, fieldValue(enstrophy, 0, qp));
        EXPECT_DOUBLE_EQ(exp_divergence, fieldValue(divergence, 0, qp));
        EXPECT_DOUBLE_EQ(exp_mag_divergence, fieldValue(mag_divergence, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleEnstrophy2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleEnstrophy2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleEnstrophy3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(IncompressibleEnstrophy3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.type_name = "IncompressibleEnstrophy";
    test_fixture.eval_name = "Incompressible Enstrophy "
                             + std::to_string(num_space_dim) + "D";
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.template buildAndTest<
        ClosureModel::IncompressibleEnstrophy<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(IncompressibleEnstrophy_Factory2D, Residual)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(IncompressibleEnstrophy_Factory2D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(IncompressibleEnstrophy_Factory3D, Residual)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(IncompressibleEnstrophy_Factory3D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
