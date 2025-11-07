#include <VertexCFD_EvaluatorTestHarness.hpp>
#include <closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp>

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleLocalTimeStepSize.hpp"

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
        this->setName(
            "Incompressible LocalTimeStepSize Unit Test Dependencies");
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
void testEval()
{
    // Setup test fixture.
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Eval dependencies.
    const double u = -1.5;
    const double v = 2.0;
    const double w
        = num_space_dim == 3 ? -0.5 : std::numeric_limits<double>::quiet_NaN();
    const Kokkos::Array<double, 3> h = {
        0.25,
        0.5,
        num_space_dim == 3 ? 0.75 : std::numeric_limits<double>::quiet_NaN()};

    auto dep_eval = Teuchos::rcp(
        new Dependencies<EvalType>(*test_fixture.ir, u, v, w, h));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    // Create test evaluator.
    auto dt_eval = Teuchos::rcp(
        new ClosureModel::IncompressibleLocalTimeStepSize<EvalType,
                                                          panzer::Traits,
                                                          NumSpaceDim>(
            *test_fixture.ir));
    test_fixture.registerEvaluator<EvalType>(dt_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(dt_eval->_local_dt);

    // Evaluate test fields.
    test_fixture.evaluate<EvalType>();

    // Check the test fields.
    auto local_dt_result
        = test_fixture.getTestFieldData<EvalType>(dt_eval->_local_dt);

    const int num_qp = num_space_dim == 2 ? 4 : 8;
    const double result = num_space_dim == 2 ? 0.1 : 0.09375;
    for (int qp = 0; qp < num_qp; ++qp)
        EXPECT_DOUBLE_EQ(result, fieldValue(local_dt_result, 0, qp));
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLocalTimeStepSize2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>();
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLocalTimeStepSize2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLocalTimeStepSize3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>();
}

//---------------------------------------------------------------------------//
TEST(IncompressibleLocalTimeStepSize3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//---------------------------------------------------------------------------//

template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.type_name = "IncompressibleLocalTimeStepSize";
    test_fixture.eval_name = "Incompressible Local Time Step Size";
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.template buildAndTest<
        ClosureModel::IncompressibleLocalTimeStepSize<EvalType,
                                                      panzer::Traits,
                                                      num_space_dim>,
        num_space_dim>();
}

TEST(IncompressibleLocalTimeStepSize_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(IncompressibleLocalTimeStepSize_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

} // end namespace Test
} // end namespace VertexCFD
