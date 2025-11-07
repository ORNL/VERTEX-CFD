#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "rad_solver/closure_models/VertexCFD_Closure_RADLocalTimeStepSize.hpp"
#include "rad_solver/species_properties/VertexCFD_ConstantSpeciesProperties.hpp"

#include <Panzer_Dimension.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>

#include <Teuchos_RCP.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
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
    Kokkos::Array<double, 3> _h;

    PHX::MDField<double, panzer::Cell, panzer::Point, panzer::Dim> _element_length;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _velocity_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _velocity_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _velocity_2;

    Dependencies(const panzer::IntegrationRule& ir,
                 const double u,
                 const double v,
                 const double w,
                 const Kokkos::Array<double, 3> h)
        : _u(u)
        , _v(v)
        , _w(w)
        , _h(h)
        , _element_length("element_length", ir.dl_vector)
        , _velocity_0("velocity_0", ir.dl_scalar)
        , _velocity_1("velocity_1", ir.dl_scalar)
        , _velocity_2("velocity_2", ir.dl_scalar)
    {
        this->addEvaluatedField(_element_length);
        this->addEvaluatedField(_velocity_0);
        this->addEvaluatedField(_velocity_1);
        this->addEvaluatedField(_velocity_2);

        this->setName("RAD LocalTimeStepSize Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = _element_length.extent(1);
        for (int qp = 0; qp < num_point; ++qp)
        {
            _element_length(c, qp, 0) = _h[0];
            _element_length(c, qp, 1) = _h[1];
            _element_length(c, qp, 2) = _h[2];
            _velocity_0(c, qp) = _u;
            _velocity_1(c, qp) = _v;
            _velocity_2(c, qp) = _w;
        }
    }
};

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval(const bool build_advection = false,
              const bool build_reaction = false,
              const bool build_diffusion = false)
{
    // Setup test fixture.
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);
    const auto& ir = *test_fixture.ir;

    const double nan_val = std::numeric_limits<double>::quiet_NaN();

    // Eval dependencies.
    const double u = -1.5;
    const double v = 2.0;
    const double w = num_space_dim == 3 ? -0.5 : nan_val;
    const Kokkos::Array<double, 3> h
        = {0.25, 0.50, num_space_dim == 3 ? 0.75 : nan_val};

    Teuchos::Array<double> species_decay(4);
    species_decay[0] = -1.0;
    species_decay[1] = 0.0;
    species_decay[2] = 1.0;
    species_decay[3] = -0.1;

    auto dep_eval = Teuchos::rcp(new Dependencies<EvalType>(ir, u, v, w, h));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    Teuchos::ParameterList rad_params;
    rad_params.set("Build Reaction", build_reaction);
    rad_params.set("Build Advection", build_advection);
    rad_params.set("Build Diffusion", build_diffusion);
    rad_params.set("Diffusion Coefficient", 0.0576);
    rad_params.set("Number of Species", 2);
    Teuchos::ParameterList reaction_params;
    reaction_params.set("Species Decay", species_decay);
    const SpeciesProperties::ConstantSpeciesProperties species_prop(
        rad_params, reaction_params);

    // Create test evaluator.
    auto dt_eval = Teuchos::rcp(
        new ClosureModel::RADLocalTimeStepSize<EvalType, panzer::Traits, NumSpaceDim>(
            ir, species_prop));
    test_fixture.registerEvaluator<EvalType>(dt_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(dt_eval->_local_dt);

    // Evaluate test fields.
    test_fixture.evaluate<EvalType>();

    // Check the test fields.
    const auto local_dt_result
        = test_fixture.getTestFieldData<EvalType>(dt_eval->_local_dt);

    const double exp_adv = num_space_dim == 3 ? 0.09375 : 0.1;
    const double exp_reac = 1.0;
    const double exp_diff = num_space_dim == 3 ? 0.7971938775510204
                                               : 0.8680555555555556;

    double exp_val;
    // The logic in the if conditions are based on the expected values.
    // Reaction has the largest time step. If it is the only term, expected
    // value should be reaction time step. If there are other terms those need
    // to be taken into account. If somehow these values are changed, one needs
    // to update these if conditions. 3 if condition is used that way to avoid
    // 7 if conditions to cover the all of the possibilities.
    if (build_reaction)
        exp_val = exp_reac;

    if (build_diffusion)
        exp_val = exp_diff;

    if (build_advection)
        exp_val = exp_adv;

    const int num_point = ir.num_points;
    for (int qp = 0; qp < num_point; ++qp)
        EXPECT_DOUBLE_EQ(exp_val, fieldValue(local_dt_result, 0, qp));
}

//---------------------------------------------------------------------------//
TEST(RADLocalTimeStepSizeSize2DReaction, Residual)
{
    testEval<panzer::Traits::Residual, 2>(false, true, false);
}

//---------------------------------------------------------------------------//
TEST(RADLocalTimeStepSizeSize2DReaction, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(false, true, false);
}

//---------------------------------------------------------------------------//
TEST(RADLocalTimeStepSizeSize3DReaction, Residual)
{
    testEval<panzer::Traits::Residual, 3>(false, true, false);
}

//---------------------------------------------------------------------------//
TEST(RADLocalTimeStepSizeSize3DReaction, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(false, true, false);
}

//---------------------------------------------------------------------------//
TEST(RADLocalTimeStepSizeSize2DAdvection, Residual)
{
    testEval<panzer::Traits::Residual, 2>(true, false, false);
}

//---------------------------------------------------------------------------//
TEST(RADLocalTimeStepSizeSize2DAdvection, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(true, false, false);
}

//---------------------------------------------------------------------------//
TEST(RADLocalTimeStepSizeSize3DAdvection, Residual)
{
    testEval<panzer::Traits::Residual, 3>(true, false, false);
}

//---------------------------------------------------------------------------//
TEST(RADLocalTimeStepSizeSize3DAdvection, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(true, false, false);
}

//---------------------------------------------------------------------------//
TEST(RADLocalTimeStepSizeSize2DReactionAdvection, Residual)
{
    testEval<panzer::Traits::Residual, 2>(true, true, false);
}

//---------------------------------------------------------------------------//
TEST(RADLocalTimeStepSizeSize2DReactionAdvection, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(true, true, false);
}

//---------------------------------------------------------------------------//
TEST(RADLocalTimeStepSizeSize3DReactionAdvection, Residual)
{
    testEval<panzer::Traits::Residual, 3>(true, true, false);
}

//---------------------------------------------------------------------------//
TEST(RADLocalTimeStepSizeSize3DReactionAdvection, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(true, true, false);
}

//---------------------------------------------------------------------------//
TEST(RADLocalTimeStepSizeSize2DRAD, Residual)
{
    testEval<panzer::Traits::Residual, 2>(true, true, true);
}

//---------------------------------------------------------------------------//
TEST(RADLocalTimeStepSizeSize2DRAD, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(true, true, true);
}

//---------------------------------------------------------------------------//
TEST(RADLocalTimeStepSizeSize3DRAD, Residual)
{
    testEval<panzer::Traits::Residual, 3>(true, true, true);
}

//---------------------------------------------------------------------------//
TEST(RADLocalTimeStepSizeSize3DRAD, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(true, true, true);
}

//---------------------------------------------------------------------------//
TEST(RADLocalTimeStepSizeSize2DDiffusion, Residual)
{
    testEval<panzer::Traits::Residual, 2>(false, false, true);
}

//---------------------------------------------------------------------------//
TEST(RADLocalTimeStepSizeSize2DDiffusion, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(false, false, true);
}

//---------------------------------------------------------------------------//
TEST(RADLocalTimeStepSizeSize3DDiffusion, Residual)
{
    testEval<panzer::Traits::Residual, 3>(false, false, true);
}

//---------------------------------------------------------------------------//
TEST(RADLocalTimeStepSizeSize3DDiffusion, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(false, false, true);
}

} // end namespace Test
} // end namespace VertexCFD
