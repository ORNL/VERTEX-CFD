#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "rad_solver/closure_models/VertexCFD_Closure_RADBADExactSolution.hpp"
#include "rad_solver/species_properties/VertexCFD_ConstantSpeciesProperties.hpp"

#include <gtest/gtest.h>

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

    double _u;
    double _v;
    double _w;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> velocity_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> velocity_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> velocity_2;

    Dependencies(const panzer::IntegrationRule& ir,
                 const double u,
                 const double v,
                 const double w)
        : _u(u)
        , _v(v)
        , _w(w)
        , velocity_0("velocity_0", ir.dl_scalar)
        , velocity_1("velocity_1", ir.dl_scalar)
        , velocity_2("velocity_2", ir.dl_scalar)
    {
        this->addEvaluatedField(velocity_0);
        this->addEvaluatedField(velocity_1);
        this->addEvaluatedField(velocity_2);

        this->setName(
            "RAD BAD Equation Exact Solution Unit Test "
            "Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "RAD BAD equation test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = velocity_0.extent(1);
        for (int qp = 0; qp < num_point; ++qp)
        {
            velocity_0(c, qp) = _u;
            velocity_1(c, qp) = _v;
            velocity_2(c, qp) = _w;
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval(const bool build_bateman = false,
              const bool build_advection = false,
              const bool build_diffusion = false)
{
    // Set up test fixture
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Get test fixture integrator rule
    auto& ir = *test_fixture.ir;

    const double nan_val = std::numeric_limits<double>::quiet_NaN();

    // Initialize velocity components and dependents (Only x-direction
    // supported)
    const double u = 0.5;
    const double v = 0.0;
    const double w = num_space_dim == 3 ? 0.0 : nan_val;

    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir, u, v, w));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Set list of parameters to pass to the test evaluator
    Teuchos::Array<double> species_amp_vct(3);
    Teuchos::Array<double> species_decay(4);

    species_amp_vct[0] = 1.0;
    species_amp_vct[1] = 0.0;
    species_amp_vct[2] = num_space_dim == 3 ? 0.0 : nan_val;

    species_decay[0] = -1.0;
    species_decay[1] = 0.0;
    species_decay[2] = 1.0;
    species_decay[3] = -0.1;

    Teuchos::ParameterList user_params;
    user_params.set("Initial Amplitude", species_amp_vct);

    if (build_advection || build_diffusion)
    {
        user_params.set("Initial Location", 0.5);
        user_params.set("Sigma", 0.01);
    }

    Teuchos::ParameterList rad_params;
    rad_params.set("Build Reaction Bateman Source", build_bateman);
    rad_params.set("Build Advection", build_advection);
    rad_params.set("Build Diffusion", build_diffusion);
    rad_params.set("Diffusion Coefficient", 1e-3);
    rad_params.set("Number of Species", 2);
    Teuchos::ParameterList reaction_params;
    reaction_params.set("Species Decay", species_decay);
    const SpeciesProperties::ConstantSpeciesProperties species_prop(
        rad_params, reaction_params);

    // Create test evaluator
    auto eval = Teuchos::rcp(
        new ClosureModel::RADBADExactSolution<EvalType, panzer::Traits, num_space_dim>(
            ir, species_prop, user_params));

    test_fixture.registerEvaluator<EvalType>(eval);
    for (int num = 0; num < 2; ++num)
        test_fixture.registerTestField<EvalType>(eval->_exact_species[num]);

    // Set time
    test_fixture.setTime(2.0);

    test_fixture.evaluate<EvalType>();

    double exp_species_reaction[2] = {1.0, 1.0};
    double exp_species_advection[2] = {1.0, 1.0};
    double exp_species_diffusion[2] = {1.0, 1.0};
    if (build_bateman)
    {
        exp_species_reaction[0] = 0.1353352832366127;
        exp_species_reaction[1] = 0.7593282998237435;
    }

    if (build_advection)
    {
        exp_species_advection[0] = 1.0;
        exp_species_advection[1] = 0.0;
        if (build_bateman)
            exp_species_advection[1] = 1.0;
    }

    if (build_diffusion)
    {
        exp_species_diffusion[0] = 0.15617376188860607;
        exp_species_diffusion[1] = 0.0;
        if (build_bateman)
            exp_species_diffusion[1] = 0.15617376188860607;
    }

    double exp_species[2];

    const int num_point = ir.num_points;
    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int num = 0; num < 2; ++num)
        {
            exp_species[num] = exp_species_reaction[num]
                               * exp_species_advection[num]
                               * exp_species_diffusion[num];
            const auto calc_species = test_fixture.getTestFieldData<EvalType>(
                eval->_exact_species[num]);
            EXPECT_DOUBLE_EQ(exp_species[num], fieldValue(calc_species, 0, qp));
        }
    }
}

//-----------------------------------------------------------------//
TEST(ReactionExactSolution2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(true, false, false);
}

//-----------------------------------------------------------------//
TEST(ReactionExactSolution2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(true, false, false);
}

//-----------------------------------------------------------------//
TEST(AdvectionExactSolution2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(false, true, false);
}

//-----------------------------------------------------------------//
TEST(AdvectionExactSolution2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(false, true, false);
}

//-----------------------------------------------------------------//
TEST(DiffusionExactSolution2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(false, false, true);
}

//-----------------------------------------------------------------//
TEST(DiffusionExactSolution2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(false, false, true);
}

//-----------------------------------------------------------------//
TEST(ADExactSolution2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(false, true, true);
}

//-----------------------------------------------------------------//
TEST(ADExactSolution2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(false, true, true);
}

//-----------------------------------------------------------------//
TEST(RDExactSolution2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(true, false, true);
}

//-----------------------------------------------------------------//
TEST(RDExactSolution2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(true, false, true);
}

//-----------------------------------------------------------------//
TEST(RAExactSolution2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(true, true, false);
}

//-----------------------------------------------------------------//
TEST(RAExactSolution2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(true, true, false);
}

//-----------------------------------------------------------------//
TEST(RADBADExactSolution2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(true, true, true);
}

//-----------------------------------------------------------------//
TEST(RADBADExactSolution2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(true, true, true);
}

//-----------------------------------------------------------------//
TEST(RADBADExactSolution3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(true, true, true);
}

//-----------------------------------------------------------------//
TEST(RADBADExactSolution3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(true, true, true);
}

} // namespace Test
} // namespace VertexCFD
