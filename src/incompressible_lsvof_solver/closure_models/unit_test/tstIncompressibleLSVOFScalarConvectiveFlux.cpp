#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFScalarConvectiveFlux.hpp"

#include "utils/VertexCFD_Utils_PhaseIndex.hpp"
#include "utils/VertexCFD_Utils_PhaseLayout.hpp"
#include "utils/VertexCFD_Utils_ScalarToVector.hpp"

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

    double _u;
    double _v;
    double _w;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> vel_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> vel_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> vel_2;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> scalar;

    Dependencies(const panzer::IntegrationRule& ir,
                 const double u,
                 const double v,
                 const double w)
        : _u(u)
        , _v(v)
        , _w(w)
        , vel_0("velocity_0", ir.dl_scalar)
        , vel_1("velocity_1", ir.dl_scalar)
        , vel_2("velocity_2", ir.dl_scalar)
        , scalar("alpha_air", ir.dl_scalar)

    {
        this->addEvaluatedField(vel_0);
        this->addEvaluatedField(vel_1);
        this->addEvaluatedField(vel_2);
        this->addEvaluatedField(scalar);

        this->setName(
            "Incompressible LSVOF Convective Flux for Scalar Equation Unit "
            "Test "
            "Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData) override
    {
        vel_0.deep_copy(_u);
        vel_1.deep_copy(_v);
        vel_2.deep_copy(_w);
        scalar.deep_copy(0.3);
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

    // Phase list
    std::vector<std::string> phase_list;
    phase_list.push_back("alpha_air");

    // Create combined array of phase fraction fields
    const auto phase_vec
        = Utils::ScalarToVector<EvalType, PhaseIndex>::createFromList(
            ir, "volume_fractions", phase_list, false, false);

    test_fixture.registerEvaluator<EvalType>(phase_vec);

    // Initialize velocity components and dependents
    const double _nanval = std::numeric_limits<double>::quiet_NaN();
    const double u = 0.25;
    const double v = 0.5;
    const double w = num_space_dim > 2 ? 0.125 : _nanval;
    const double scalar = 0.3;

    auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir, u, v, w));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize class object to test
    const auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleLSVOFScalarConvectiveFlux<EvalType,
                                                                  panzer::Traits,
                                                                  num_space_dim>(
            ir, 0, 1, "alpha_air_equation"));

    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_scalar_flux);

    test_fixture.evaluate<EvalType>();

    const auto fc_scalar
        = test_fixture.getTestFieldData<EvalType>(eval->_scalar_flux);

    const int num_point = ir.num_points;

    // Expected values
    const double exp_scalar_flux[3] = {scalar * u, scalar * v, scalar * w};

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int dim = 0; dim < num_space_dim; dim++)
        {
            EXPECT_EQ(exp_scalar_flux[dim], fieldValue(fc_scalar, 0, qp, dim));
        }
    }
}

//-----------------------------------------------------------------//
TEST(TestLSVOFScalarConvective2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(TestLSVOFScalarConvective2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(TestLSVOFScalarConvective3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(TestLSVOFScalarConvective3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
