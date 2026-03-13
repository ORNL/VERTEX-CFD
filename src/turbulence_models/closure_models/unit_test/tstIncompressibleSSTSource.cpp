#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleSSTSource.hpp"

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
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> blending_function;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> wall_dist;
    const double _omega;
    const double _f1;

    Dependencies(const panzer::IntegrationRule& ir,
                 const double omega,
                 const double blend)
        : grad_vel_0("GRAD_velocity_0", ir.dl_vector)
        , grad_vel_1("GRAD_velocity_1", ir.dl_vector)
        , grad_vel_2("GRAD_velocity_2", ir.dl_vector)
        , turb_eddy_viscosity("turbulent_eddy_viscosity", ir.dl_scalar)
        , turb_kinetic_energy("turb_kinetic_energy", ir.dl_scalar)
        , turb_specific_dissipation_rate("turb_specific_dissipation_rate",
                                         ir.dl_scalar)
        , grad_k("GRAD_turb_kinetic_energy", ir.dl_vector)
        , grad_w("GRAD_turb_specific_dissipation_rate", ir.dl_vector)
        , blending_function("sst_blending_function", ir.dl_scalar)
        , wall_dist("distance", ir.dl_scalar)
        , _omega(omega)
        , _f1(blend)
    {
        this->addEvaluatedField(grad_vel_0);
        this->addEvaluatedField(grad_vel_1);
        this->addEvaluatedField(grad_vel_2);
        this->addEvaluatedField(turb_eddy_viscosity);
        this->addEvaluatedField(turb_kinetic_energy);
        this->addEvaluatedField(turb_specific_dissipation_rate);
        this->addEvaluatedField(grad_k);
        this->addEvaluatedField(grad_w);
        this->addEvaluatedField(wall_dist);
        this->addEvaluatedField(blending_function);
        this->setName(
            "Incompressible SST K-Omega Source Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "SST K-Omega source test dependencies",
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
                grad_vel_0(c, qp, dim) = 0.250 * dimqp;
                grad_vel_1(c, qp, dim) = 0.500 * dimqp;
                grad_vel_2(c, qp, dim) = num_space_dim == 3 ? 0.125 * dimqp
                                                            : _nanval;

                grad_k(c, qp, dim) = 0.750 * dimqp;
                grad_w(c, qp, dim) = 1.250 * dimqp;
            }

            turb_eddy_viscosity(c, qp) = 0.25;
            turb_kinetic_energy(c, qp) = 2.5;
            turb_specific_dissipation_rate(c, qp) = _omega;
            wall_dist(c, qp) = 3.0;
            blending_function(c, qp) = _f1;
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval(const bool limit_production,
              const double omega,
              const double blending_function,
              const double expected_k_source,
              const double expected_w_source)
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Create parameter list for user-defined constants
    Teuchos::ParameterList turb_params;
    turb_params.set<bool>("Limit Production Term", limit_production)
        .set<double>("sigma_w1", 0.4)
        .set<double>("sigma_w2", 0.9)
        .set<double>("beta_1", 0.07)
        .set<double>("beta_2", 0.08)
        .set<double>("beta_star", 0.1)
        .set<double>("kappa", 0.4);

    // Eval dependencies
    const auto deps = Teuchos::rcp(
        new Dependencies<EvalType, NumSpaceDim>(ir, omega, blending_function));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize and register
    auto eval = Teuchos::rcp(
        new ClosureModel::
            IncompressibleSSTSource<EvalType, panzer::Traits, NumSpaceDim>(
                ir, turb_params));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_k_source);
    test_fixture.registerTestField<EvalType>(eval->_w_source);
    test_fixture.evaluate<EvalType>();

    // Evaluate test fields
    const auto fv_k_source
        = test_fixture.getTestFieldData<EvalType>(eval->_k_source);
    const auto fv_w_source
        = test_fixture.getTestFieldData<EvalType>(eval->_w_source);

    // Assert values
    const int num_point = ir.num_points;
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(expected_k_source, fieldValue(fv_k_source, 0, qp));
        EXPECT_DOUBLE_EQ(expected_w_source, fieldValue(fv_w_source, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTSource2DLowerOmega, Residual)
{
    testEval<panzer::Traits::Residual, 2>(
        false, 0.1, 1.973080737090289e-08, 0.50625, 85.1065413776178);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTSource2DLowerOmega, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(
        false, 0.1, 1.973080737090289e-08, 0.50625, 85.1065413776178);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTSource2DLowerOmegaLimited, Residual)
{
    testEval<panzer::Traits::Residual, 2>(
        true, 0.1, 1.973080737090289e-08, 0.475, 85.1065413776178);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTSource2DLowerOmegaLimited, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(
        true, 0.1, 1.973080737090289e-08, 0.475, 85.1065413776178);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTSource3DLowerOmega, Residual)
{
    testEval<panzer::Traits::Residual, 3>(
        false, 0.1, 3.210056905147414e-10, 1.15859375, 237.88081712957515);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTSource3DLowerOmega, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(
        false, 0.1, 3.210056905147414e-10, 1.15859375, 237.88081712957515);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTSource3DLowerOmegaLimited, Residual)
{
    testEval<panzer::Traits::Residual, 3>(
        true, 0.1, 3.210056905147414e-10, 0.475, 237.88081712957515);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTSource3DLowerOmegaLimited, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(
        true, 0.1, 3.210056905147414e-10, 0.475, 237.88081712957515);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTSource2DHigherOmega, Residual)
{
    testEval<panzer::Traits::Residual, 2>(
        false, 10.0, 0.0770077264126132, -1.96875, -6.38684028043009);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTSource2DHigherOmega, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(
        false, 10.0, 0.0770077264126132, -1.96875, -6.38684028043009);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTSource2DHigherOmegaLimited, Residual)
{
    testEval<panzer::Traits::Residual, 2>(
        true, 10.0, 0.0770077264126132, -1.96875, -6.38684028043009);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTSource2DHigherOmegaLimited, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(
        true, 10.0, 0.0770077264126132, -1.96875, -6.38684028043009);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTSource3DHigherOmega, Residual)
{
    testEval<panzer::Traits::Residual, 3>(
        false, 10.0, 0.032089547620883105, -1.31640625, -4.026363141094224);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTSource3DHigherOmega, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(
        false, 10.0, 0.032089547620883105, -1.31640625, -4.026363141094224);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTSource3DHigherOmegaLimited, Residual)
{
    testEval<panzer::Traits::Residual, 3>(
        true, 10.0, 0.032089547620883105, -1.31640625, -4.026363141094224);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSSTSource3DHigherOmegaLimited, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(
        true, 10.0, 0.032089547620883105, -1.31640625, -4.026363141094224);
}

//-----------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD
