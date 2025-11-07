#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFBuoyancySource.hpp"

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

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> rho;

    Dependencies(const panzer::IntegrationRule& ir)
        : rho("density", ir.dl_scalar)
    {
        this->addEvaluatedField(rho);

        this->setName(
            "Incompressible LSVOF Buoyancy Source Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "LSVOF buoyancy source test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = rho.extent(1);
        for (int qp = 0; qp < num_point; ++qp)
        {
            rho(c, qp) = 10.0;
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

    // Initialize class object to test
    Teuchos::Array<double> gravity(num_space_dim);
    gravity[0] = 2.0;
    gravity[1] = -3.0;
    if (num_space_dim == 3)
        gravity[2] = 4.0;

    Teuchos::ParameterList user_params;
    user_params.set("Gravity", gravity);

    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    const auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleLSVOFBuoyancySource<EvalType,
                                                            panzer::Traits,
                                                            num_space_dim>(
            ir, user_params));

    test_fixture.registerEvaluator<EvalType>(eval);
    for (int dim = 0; dim < num_space_dim; ++dim)
        test_fixture.registerTestField<EvalType>(
            eval->_buoyancy_momentum_source[dim]);

    test_fixture.evaluate<EvalType>();
    const auto fc_mom_0 = test_fixture.getTestFieldData<EvalType>(
        eval->_buoyancy_momentum_source[0]);
    const auto fc_mom_1 = test_fixture.getTestFieldData<EvalType>(
        eval->_buoyancy_momentum_source[1]);

    const int num_point = ir.num_points;

    const double exp_mom_source_3d[3] = {20.0, -30.0, 40.0};

    const double exp_mom_source_2d[2] = {20.0, -30.0};

    const double* exp_mom_source = num_space_dim == 3 ? exp_mom_source_3d
                                                      : exp_mom_source_2d;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_mom_source[0], fieldValue(fc_mom_0, 0, qp));
        EXPECT_DOUBLE_EQ(exp_mom_source[1], fieldValue(fc_mom_1, 0, qp));
        if (num_space_dim > 2)
        {
            const auto fc_mom_2 = test_fixture.getTestFieldData<EvalType>(
                eval->_buoyancy_momentum_source[2]);
            EXPECT_DOUBLE_EQ(exp_mom_source[2], fieldValue(fc_mom_2, 0, qp));
        }
    }
}

//-----------------------------------------------------------------//
TEST(TestLSVOFBuoyancySource2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(TestLSVOFBuoyancySource2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(TestLSVOFBuoyancySource3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(TestLSVOFBuoyancySource3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//-----------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD
