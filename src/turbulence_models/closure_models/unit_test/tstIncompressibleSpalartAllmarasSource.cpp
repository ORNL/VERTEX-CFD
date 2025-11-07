#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleSpalartAllmarasSource.hpp"

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
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_sa_var;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> nu;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> sa_var;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> wall_dist;

    const double _sa_var_value;

    Dependencies(const panzer::IntegrationRule& ir, const double sa_var_value)
        : grad_vel_0("GRAD_velocity_0", ir.dl_vector)
        , grad_vel_1("GRAD_velocity_1", ir.dl_vector)
        , grad_vel_2("GRAD_velocity_2", ir.dl_vector)
        , grad_sa_var("GRAD_spalart_allmaras_variable", ir.dl_vector)
        , nu("kinematic_viscosity", ir.dl_scalar)
        , sa_var("spalart_allmaras_variable", ir.dl_scalar)
        , wall_dist("distance", ir.dl_scalar)
        , _sa_var_value(sa_var_value)
    {
        this->addEvaluatedField(grad_vel_0);
        this->addEvaluatedField(grad_vel_1);
        this->addEvaluatedField(grad_vel_2);
        this->addEvaluatedField(grad_sa_var);
        this->addEvaluatedField(nu);
        this->addEvaluatedField(sa_var);
        this->addEvaluatedField(wall_dist);
        this->setName(
            "Spalart-Allmaras Incompressible Source Unit "
            "Test "
            "Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "SA source test dependencies",
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
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                const int sign = pow(-1, dim + 1);
                const int dimqp = (dim + 1) * sign;
                grad_vel_0(c, qp, dim) = 0.250 * dimqp;
                grad_vel_1(c, qp, dim) = 0.500 * dimqp;
                grad_vel_2(c, qp, dim) = num_space_dim == 3 ? 0.125 * dimqp
                                                            : _nanval;

                grad_sa_var(c, qp, dim) = 0.750 * dimqp;
            }

            nu(c, qp) = 0.25;
            sa_var(c, qp) = _sa_var_value;
            wall_dist(c, qp) = 0.1;
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval(const double sa_var_value)
{
    const int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Eval dependencies
    const auto deps
        = Teuchos::rcp(new Dependencies<EvalType>(ir, sa_var_value));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize and register
    auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleSpalartAllmarasSource<EvalType,
                                                              panzer::Traits,
                                                              NumSpaceDim>(ir));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_sa_source);
    test_fixture.evaluate<EvalType>();

    // Evaluate test fields
    const auto fv_var
        = test_fixture.getTestFieldData<EvalType>(eval->_sa_source);

    // Expected values
    const int num_point = ir.num_points;

    double exp_sa_source = 0.0;

    if (num_space_dim < 3)
        exp_sa_source = sa_var_value < 0 ? 2917.866397598156
                                         : -5842.742478724074;
    else
        exp_sa_source = sa_var_value < 0 ? 2922.6799728440574
                                         : -5837.973707512366;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_sa_source, fieldValue(fv_var, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleSASourcePositive2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>(3.0);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSASourcePositive2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>(3.0);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSASourceNegative2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>(-3.0);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSASourceNegative2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>(-3.0);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSASourcePositive3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>(3.0);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSASourcePositive3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>(3.0);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSASourceNegative3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>(-3.0);
}

//-----------------------------------------------------------------//
TEST(IncompressibleSASourceNegative3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>(-3.0);
}

//-----------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD
