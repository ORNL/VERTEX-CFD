#include <VertexCFD_EvaluatorTestHarness.hpp>
#include <closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp>

#include <utils/VertexCFD_Utils_VectorField.hpp>

#include "closure_models/VertexCFD_Closure_VectorFieldDivergence.hpp"

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
    double _nanval = std::numeric_limits<double>::quiet_NaN();

    Kokkos::Array<
        PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        3>
        grad_test_field;

    Dependencies(const panzer::IntegrationRule& ir,
                 const std::string field_name)
    {
        Utils::addEvaluatedVectorField(
            *this, ir.dl_vector, grad_test_field, "GRAD_" + field_name + "_");

        this->setName("Vector Field Divergence Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "vector field divergence test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = grad_test_field[0].extent(1);
        const int num_grad_dim = grad_test_field[0].extent(2);
        for (int qp = 0; qp < num_point; ++qp)
        {
            const double qp1 = qp + 1.0;
            for (int dim = 0; dim < num_grad_dim; ++dim)
            {
                const double dqp1 = qp1 * (dim + 1.0);
                grad_test_field[0](c, qp, dim) = 0.375 * dqp1;
                grad_test_field[1](c, qp, dim) = 0.225 * dqp1;
                grad_test_field[2](c, qp, dim)
                    = num_grad_dim == 2 ? _nanval : 0.125 * dqp1;
            }
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval(const int num_grad_dim,
              const std::string& field_name,
              const bool use_abs)
{
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_grad_dim, integration_order, basis_order);

    auto& ir = *test_fixture.ir;

    auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir, field_name));
    test_fixture.registerEvaluator<EvalType>(deps);

    const std::string closure_name = use_abs ? "AbsVectorFieldDivergence"
                                             : "VectorFieldDivergence";
    const auto eval = Teuchos::rcp(
        new ClosureModel::VectorFieldDivergence<EvalType, panzer::Traits, NumSpaceDim>(
            ir, field_name, closure_name));
    test_fixture.registerEvaluator<EvalType>(eval);

    test_fixture.registerTestField<EvalType>(eval->_vector_field_divergence);

    test_fixture.evaluate<EvalType>();

    const auto div_field = test_fixture.getTestFieldData<EvalType>(
        eval->_vector_field_divergence);

    const auto grad_test_field_0
        = test_fixture.getTestFieldData<EvalType>(deps->grad_test_field[0]);
    const auto grad_test_field_1
        = test_fixture.getTestFieldData<EvalType>(deps->grad_test_field[1]);
    const auto grad_test_field_2
        = test_fixture.getTestFieldData<EvalType>(deps->grad_test_field[2]);

    const int num_point = ir.num_points;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        auto exp_div = fieldValue(grad_test_field_0, 0, qp, 0)
                       + fieldValue(grad_test_field_1, 0, qp, 1);
        if (num_grad_dim > 2)
            exp_div += fieldValue(grad_test_field_2, 0, qp, 2);
        if (use_abs)
            exp_div = std::abs(exp_div);
        EXPECT_DOUBLE_EQ(exp_div, fieldValue(div_field, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(Vector2DDivergence2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(2, "foo", false);
}

TEST(Vector2DDivergence2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(2, "foo", false);
}

//-----------------------------------------------------------------//
TEST(Vector3DDivergence2D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(2, "foo", false);
}

TEST(Vector3DDivergence2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(2, "foo", false);
}
//-----------------------------------------------------------------//
TEST(Vector3DDivergence3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(3, "bar", false);
}

TEST(Vector3DDivergence3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(3, "bar", false);
}

//-----------------------------------------------------------------//
TEST(Vector3DAbsDivergence3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(3, "bar", true);
}

TEST(Vector3DAbsDivergence3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(3, "bar", true);
}
//-----------------------------------------------------------------//
template<class EvalType, int NumGradDim>
void testFactory(const std::string& abs_pre = "")
{
    constexpr int num_grad_dim = NumGradDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.type_name = abs_pre + "VectorFieldDivergence";
    test_fixture.eval_name = "Vector Field Divergence 2D";
    test_fixture.model_params.set("Field Names", "foo");
    test_fixture.template buildAndTest<
        ClosureModel::VectorFieldDivergence<EvalType, panzer::Traits, num_grad_dim>,
        num_grad_dim>();
}

TEST(VectorFieldDivergence_Factory2D, Residual)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(VectorFieldDivergence_Factory2D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(VectorFieldDivergence_Factory3D, Residual)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(VectorFieldDivergence_Factory3D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

TEST(VectorFieldAbsDivergence_Factory3D, Residual)
{
    testFactory<panzer::Traits::Residual, 3>("Abs");
}

TEST(VectorFieldAbsDivergence_Factory3D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 3>("Abs");
}

} // namespace Test
} // namespace VertexCFD
