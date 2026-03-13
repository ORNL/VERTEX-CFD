#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "utils/VertexCFD_Utils_VelocityDim.hpp"
#include "utils/VertexCFD_Utils_VelocityLayout.hpp"

#include "plasma_solver/closure_models/VertexCFD_Closure_PlasmaSpeciesConvectiveFlux.hpp"

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> nd;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, VelocityDim> vel;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> rho;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> p;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> E;

    Dependencies(const panzer::IntegrationRule& ir,
                 const std::string field_prefix)
        : nd(field_prefix + "fast_particle_number_density", ir.dl_scalar)
        , vel(field_prefix + "fast_particle_velocity",
              Utils::buildVelocityLayout(ir.dl_scalar, ir.spatial_dimension))
        , rho(field_prefix + "fast_particle_momentum_density", ir.dl_scalar)
        , p(field_prefix + "fast_particle_pressure", ir.dl_scalar)
        , E(field_prefix + "fast_particle_energy", ir.dl_scalar)
    {
        this->addEvaluatedField(nd);
        this->addEvaluatedField(vel);
        this->addEvaluatedField(rho);
        this->addEvaluatedField(p);
        this->addEvaluatedField(E);

        this->setName("Plasma Species Convective Flux Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        nd.deep_copy(2.0);
        rho.deep_copy(1.5);
        p.deep_copy(2.5);
        E.deep_copy(3.5);
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
                vel(c, qp, d) = 4.0 * (d + 1);
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval(const std::string field_prefix = "",
              const std::string flux_prefix = "")
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Closure parameters
    Teuchos::ParameterList closure_params;
    closure_params.set("Species Name", "fast_particle");
    closure_params.set("Residual Name",
                       "VERY_IMPORTANT_RESIDUAL_IN_PLASMA_PHYSICS");

    // Eval dependencies
    auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir, field_prefix));
    test_fixture.registerEvaluator<EvalType>(deps);

    auto eval = Teuchos::rcp(
        new ClosureModel::PlasmaSpeciesConvectiveFlux<EvalType,
                                                      panzer::Traits,
                                                      num_space_dim>(
            ir, closure_params, flux_prefix, field_prefix));

    // Register and evaluate
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_nd_flux);
    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        test_fixture.registerTestField<EvalType>(eval->_momentum_flux[dim]);
    }
    test_fixture.registerTestField<EvalType>(eval->_energy_flux);
    test_fixture.evaluate<EvalType>();

    // Number density flux
    const auto nd_flux
        = test_fixture.getTestFieldData<EvalType>(eval->_nd_flux);
    const double exp_nd_flux[3] = {
        8.0,
        16.0,
        num_space_dim == 3 ? 24.0 : std::numeric_limits<double>::quiet_NaN()};

    // Momentum density flux
    const auto mom_0_flux
        = test_fixture.getTestFieldData<EvalType>(eval->_momentum_flux[0]);
    const auto mom_1_flux
        = test_fixture.getTestFieldData<EvalType>(eval->_momentum_flux[1]);
    const double exp_mom_0_flux[3] = {
        26.5,
        48.0,
        num_space_dim == 3 ? 72.0 : std::numeric_limits<double>::quiet_NaN()};
    const double exp_mom_1_flux[3] = {
        48.0,
        98.5,
        num_space_dim == 3 ? 144.0 : std::numeric_limits<double>::quiet_NaN()};

    // Enegy density flux
    const auto energy_flux
        = test_fixture.getTestFieldData<EvalType>(eval->_energy_flux);
    const double exp_energy_flux[3] = {
        24.0,
        48.0,
        num_space_dim == 3 ? 72.0 : std::numeric_limits<double>::quiet_NaN()};

    // Assert values
    const int num_point = ir.num_points;
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int dim = 0; dim < num_space_dim; dim++)
        {
            EXPECT_EQ(exp_nd_flux[dim], fieldValue(nd_flux, 0, qp, dim));
            EXPECT_EQ(exp_mom_0_flux[dim], fieldValue(mom_0_flux, 0, qp, dim));
            EXPECT_EQ(exp_mom_1_flux[dim], fieldValue(mom_1_flux, 0, qp, dim));
            if (num_space_dim > 2)
            {
                const double exp_mom_2_flux[3] = {72.0, 144.0, 218.5};
                const auto mom_2_flux = test_fixture.getTestFieldData<EvalType>(
                    eval->_momentum_flux[2]);
                EXPECT_EQ(exp_mom_2_flux[dim],
                          fieldValue(mom_2_flux, 0, qp, dim));
            }
            EXPECT_EQ(exp_energy_flux[dim],
                      fieldValue(energy_flux, 0, qp, dim));
        }
    }
}

//-----------------------------------------------------------------//
TEST(PlasmaSpeciesConvectiveFlux2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(PlasmaSpeciesConvectiveFlux2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(PlasmaSpeciesConvectiveFlux3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>("FIELD_", "FLUX_");
}

//-----------------------------------------------------------------//
TEST(PlasmaSpeciesConvectiveFlux3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>("FIELD_", "FLUX_");
}

} // namespace Test
} // namespace VertexCFD
