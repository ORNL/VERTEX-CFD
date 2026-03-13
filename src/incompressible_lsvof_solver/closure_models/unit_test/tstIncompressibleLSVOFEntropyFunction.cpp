#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFEntropyFunction.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{

// entropy types
enum EntropyType
{
    quadratic,
    log,
    biquadratic
};
template<class EvalType>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    const double _nanval = std::numeric_limits<double>::quiet_NaN();

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> vel_0;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> vel_1;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> vel_2;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_scalar;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> scalar;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> dxdt_scalar;

    Dependencies(const panzer::IntegrationRule& ir)
        : vel_0("velocity_0", ir.dl_scalar)
        , vel_1("velocity_1", ir.dl_scalar)
        , vel_2("velocity_2", ir.dl_scalar)
        , grad_scalar("GRAD_alpha_air", ir.dl_vector)
        , scalar("alpha_air", ir.dl_scalar)
        , dxdt_scalar("DXDT_alpha_air", ir.dl_scalar)

    {
        this->addEvaluatedField(vel_0);
        this->addEvaluatedField(vel_1);
        this->addEvaluatedField(vel_2);
        this->addEvaluatedField(grad_scalar);
        this->addEvaluatedField(scalar);
        this->addEvaluatedField(dxdt_scalar);

        this->setName(
            "Incompressible LSVOF Entropy Function for Scalar Equation "
            "Unit "
            "Test "
            "Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "LSVOF entropy viscosity test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = grad_scalar.extent(1);
        const int num_space_dim = grad_scalar.extent(2);

        for (int qp = 0; qp < num_point; ++qp)
        {
            vel_0(c, qp) = 0.25;
            vel_1(c, qp) = 0.5;
            vel_2(c, qp) = num_space_dim > 2 ? 0.125 : _nanval;

            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                const int sign = pow(-1, dim + 1);
                const int dimqp = (dim + 1) * sign;
                grad_scalar(c, qp, dim)
                    = abs((vel_0(c, qp) + vel_1(c, qp)) * dimqp);
            }

            scalar(c, qp) = 0.3;
            dxdt_scalar(c, qp) = 0.01;
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval(const EntropyType entropy_type)
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Initialize velocity components and dependents
    const auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Closure parameters
    Teuchos::ParameterList closure_params;
    closure_params.set("Time Derivative Entropy Scaling", 1.0);
    if (entropy_type == EntropyType::quadratic)
    {
        closure_params.set("Entropy Type", "Quadratic");
    }
    else if (entropy_type == EntropyType::log)
    {
        closure_params.set("Entropy Type", "Log");
    }
    else if (entropy_type == EntropyType::biquadratic)
    {
        closure_params.set("Entropy Type", "Biquadratic");
    }

    // Initialize class object to test
    const auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleLSVOFEntropyFunction<EvalType,
                                                             panzer::Traits,
                                                             num_space_dim>(
            ir, closure_params, "alpha_air", "alpha_air_equation"));

    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_entropy_function);
    test_fixture.registerTestField<EvalType>(eval->_entropy_residual);

    test_fixture.evaluate<EvalType>();

    const auto fc_entr
        = test_fixture.getTestFieldData<EvalType>(eval->_entropy_function);

    const auto fc_entr_res
        = test_fixture.getTestFieldData<EvalType>(eval->_entropy_residual);

    const int num_point = ir.num_points;

    // Expected values

    double exp_entropy_func = 0.0;
    double exp_entropy_res_func = 0.0;

    if (entropy_type == EntropyType::quadratic)
    {
        exp_entropy_func = 0.045;
        exp_entropy_res_func = (num_space_dim == 3) ? 0.368625 : 0.28425;
    }
    if (entropy_type == EntropyType::log)
    {
        exp_entropy_func = -0.6108643020548935;
        exp_entropy_res_func = (num_space_dim == 3) ? -1.0411172459507765
                                                    : -0.8028147227168755;
    }
    if (entropy_type == EntropyType::biquadratic)
    {
        exp_entropy_func = 0.0819;
        exp_entropy_res_func = (num_space_dim == 3) ? 0.604545 : 0.46617;
    }

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_entropy_func, fieldValue(fc_entr, 0, qp));

        EXPECT_DOUBLE_EQ(exp_entropy_res_func, fieldValue(fc_entr_res, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(TestLSVOFEntropyFunction2DQuad, Residual)
{
    testEval<panzer::Traits::Residual, 2>(EntropyType::quadratic);
}

TEST(TestLSVOFEntropyFunction2DQuad, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(EntropyType::quadratic);
}

TEST(TestLSVOFEntropyFunction3DQuad, Residual)
{
    testEval<panzer::Traits::Residual, 3>(EntropyType::quadratic);
}

TEST(TestLSVOFEntropyFunction3DQuad, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(EntropyType::quadratic);
}

//-----------------------------------------------------------------//
TEST(TestLSVOFEntropyFunction2DLog, Residual)
{
    testEval<panzer::Traits::Residual, 2>(EntropyType::log);
}

TEST(TestLSVOFEntropyFunction2DLog, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(EntropyType::log);
}

TEST(TestLSVOFEntropyFunction3DLog, Residual)
{
    testEval<panzer::Traits::Residual, 3>(EntropyType::log);
}

TEST(TestLSVOFEntropyFunction3DLog, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(EntropyType::log);
}

//-----------------------------------------------------------------//
TEST(TestLSVOFEntropyFunction2DBiquad, Residual)
{
    testEval<panzer::Traits::Residual, 2>(EntropyType::biquadratic);
}

TEST(TestLSVOFEntropyFunction2DBiquad, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(EntropyType::biquadratic);
}

TEST(TestLSVOFEntropyFunction3DBiquad, Residual)
{
    testEval<panzer::Traits::Residual, 3>(EntropyType::biquadratic);
}

TEST(TestLSVOFEntropyFunction3DBiquad, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(EntropyType::biquadratic);
}

} // namespace Test
} // namespace VertexCFD
