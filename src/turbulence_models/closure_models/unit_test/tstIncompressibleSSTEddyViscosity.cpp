#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleSSTEddyViscosity.hpp"

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_k;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_w;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> wall_dist;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> rho;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> nu;

    const double _omega;

    Dependencies(const panzer::IntegrationRule& ir, double omega)
        : grad_vel_0("GRAD_velocity_0", ir.dl_vector)
        , grad_vel_1("GRAD_velocity_1", ir.dl_vector)
        , grad_vel_2("GRAD_velocity_2", ir.dl_vector)
        , turb_kinetic_energy("turb_kinetic_energy", ir.dl_scalar)
        , turb_specific_dissipation_rate("turb_specific_dissipation_rate",
                                         ir.dl_scalar)
        , grad_k("GRAD_turb_kinetic_energy", ir.dl_vector)
        , grad_w("GRAD_turb_specific_dissipation_rate", ir.dl_vector)
        , wall_dist("distance", ir.dl_scalar)
        , rho("density", ir.dl_scalar)
        , nu("kinematic_viscosity", ir.dl_scalar)
        , _omega(omega)
    {
        this->addEvaluatedField(grad_vel_0);
        this->addEvaluatedField(grad_vel_1);
        this->addEvaluatedField(grad_vel_2);
        this->addEvaluatedField(turb_kinetic_energy);
        this->addEvaluatedField(turb_specific_dissipation_rate);
        this->addEvaluatedField(grad_k);
        this->addEvaluatedField(grad_w);
        this->addEvaluatedField(wall_dist);
        this->addEvaluatedField(rho);
        this->addEvaluatedField(nu);
        this->setName(
            "Incompressible SST K-Omega Eddy Viscosity Unit Test "
            "Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "SST K-Omega eddy viscosity test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = grad_vel_0.extent(1);
        const int num_space_dim = grad_vel_0.extent(2);

        for (int qp = 0; qp < num_point; ++qp)
        {
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                const int sign = (dim % 2 == 0) ? -1 : 1;
                const int dimqp = (dim + 1) * sign;
                grad_vel_0(c, qp, dim) = 0.250 * dimqp;
                grad_vel_1(c, qp, dim) = 0.500 * dimqp;
                grad_vel_2(c, qp, dim) = num_space_dim == 3 ? 0.125 * dimqp
                                                            : _nanval;

                grad_k(c, qp, dim) = 0.750 * dimqp;
                grad_w(c, qp, dim) = 1.250 * dimqp;
            }
            turb_kinetic_energy(c, qp) = 2.5;
            turb_specific_dissipation_rate(c, qp) = _omega;
            wall_dist(c, qp) = 3.0;
            rho(c, qp) = 1.0;
            nu(c, qp) = 1.0e-5;
        }
    }
};

template<class EvalType, int NumSpaceDim, bool OnBoundary = false>
void testEval(const double omega,
              const double expected_nu_t,
              const double expected_blending_function)
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Eval dependencies
    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir, omega));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Create parameter list for user-defined constants
    Teuchos::ParameterList turb_params;
    turb_params.set("a_1", 0.41);

    // Initialize and register
    auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleSSTEddyViscosity<EvalType,
                                                         panzer::Traits,
                                                         NumSpaceDim>(
            ir, turb_params, OnBoundary));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_nu_t);
    test_fixture.registerTestField<EvalType>(eval->_sst_blending_function);
    test_fixture.evaluate<EvalType>();

    // Evaluate test fields
    const auto fv_nu_t = test_fixture.getTestFieldData<EvalType>(eval->_nu_t);
    const auto fv_blending = test_fixture.getTestFieldData<EvalType>(
        eval->_sst_blending_function);

    // Expected values
    const int num_point = ir.num_points;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(expected_nu_t, fieldValue(fv_nu_t, 0, qp));
        EXPECT_DOUBLE_EQ(expected_blending_function,
                         fieldValue(fv_blending, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTEddyViscosity2DBoundary, Residual)
{
    testEval<panzer::Traits::Residual, 2, true>(10.0, 0.25, 1.0);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTEddyViscosity2DBoundary, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2, true>(10.0, 0.25, 1.0);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTEddyViscosity3DBoundary, Residual)
{
    testEval<panzer::Traits::Residual, 3, true>(10.0, 0.25, 1.0);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTEddyViscosity3DBoundary, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3, true>(10.0, 0.25, 1.0);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTEddyViscosityLowerOmega2DBoundary, Residual)
{
    testEval<panzer::Traits::Residual, 2, true>(0.1, 1.025, 1.0);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTEddyViscosityLowerOmega2DBoundary, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2, true>(0.1, 1.025, 1.0);
}
//-----------------------------------------------------------------//
TEST(IncompressibleSSTEddyViscosityLowerOmega3DBoundary, Residual)
{
    testEval<panzer::Traits::Residual, 3, true>(0.1, 0.48572607976245563, 1.0);
}
//-----------------------------------------------------------------//
TEST(IncompressibleSSTEddyViscosityLowerOmega3DBoundary, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3, true>(0.1, 0.48572607976245563, 1.0);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTEddyViscosity2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(10.0, 0.25, 0.11706556669815522);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTEddyViscosity2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(10.0, 0.25, 0.11706556669815522);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTEddyViscosity3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(10.0, 0.25, 0.03208954762088309);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTEddyViscosity3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(10.0, 0.25, 0.03208954762088309);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTEddyViscosityLowerOmega2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(0.1, 1.025, 1.9730807370902888e-08);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTEddyViscosityLowerOmega2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(0.1, 1.025, 1.9730807370902888e-08);
}
//-----------------------------------------------------------------//
TEST(IncompressibleSSTEddyViscosityLowerOmega3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(
        0.1, 0.48572607976245563, 3.2100569051474141e-10);
}
//-----------------------------------------------------------------//
TEST(IncompressibleSSTEddyViscosityLowerOmega3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(
        0.1, 0.48572607976245563, 3.2100569051474141e-10);
}
//-----------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD
