#include "VertexCFD_ClosureModelFactoryTestHarness.hpp"
#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "closure_models/VertexCFD_Closure_VariableOldValue.hpp"

#include <Panzer_Traits.hpp>

#include <Teuchos_RCP.hpp>

#include <gtest/gtest.h>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
enum class OldValueType
{
    LastTime,
    LastStage
};

//---------------------------------------------------------------------------//
template<class EvalType>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    int _counter;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> var;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_var;

    Dependencies(const panzer::IntegrationRule& ir)
        : _counter(0)
        , var("var", ir.dl_scalar)
        , grad_var("GRAD_var", ir.dl_vector)

    {
        this->addEvaluatedField(var);
        this->addEvaluatedField(grad_var);

        this->setName("Variable Old Value Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        ++_counter;

        Kokkos::parallel_for(
            "Variable old value test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = grad_var.extent(1);
        const int num_space_dim = grad_var.extent(2);

        for (int qp = 0; qp < num_point; ++qp)
        {
            var(c, qp) = 2.0 * _counter;

            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                grad_var(c, qp, dim) = var(c, qp) * (dim + 1);
            }
        }
    }
};

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval(const OldValueType old_value_type)
{
    // Setup test fixture.
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);
    const auto& ir = *test_fixture.ir;

    using scalar_type = typename EvalType::ScalarT;

    // Time stepping parameters
    const int n_steps = 3;
    const double dt = 0.1;
    const double t0 = 0.0;
    const int n_stages = 2;
    const int n_iters = 3;

    // Parameter list
    Teuchos::ParameterList params;
    params.set("Field Name", "var");
    params.set("Save Old Gradient", true);
    if (old_value_type == OldValueType::LastTime)
        params.set("Old Value Type", "LastTime");
    else
        params.set("Old Value Type", "LastStage");

    // Create evaluator.
    auto eval = Teuchos::rcp(
        new ClosureModel::VariableOldValue<EvalType, panzer::Traits>(
            *test_fixture.ir, params));
    test_fixture.registerEvaluator<EvalType>(eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(eval->_var_old_val);

    // Create dependencies
    const auto deps
        = Teuchos::rcp(new Dependencies<EvalType>(*test_fixture.ir));

    test_fixture.registerEvaluator<EvalType>(deps);

    test_fixture.postRegSetup();

    const int num_point = ir.num_points;

    // Do time stepping
    for (int step = 0; step < n_steps; ++step)
    {
        const double time = t0 + step * dt;
        test_fixture.setTime(time);

        for (int stage = 0; stage < n_stages; ++stage)
        {
            test_fixture.setStageNumber(stage);

            for (int iter = 0; iter < n_iters; ++iter)
            {
                test_fixture.evaluate<EvalType>(false);

                const auto var_result = test_fixture.getTestFieldData<EvalType>(
                    eval->_var_old_val);

                const auto grad_var_result
                    = test_fixture.getTestFieldData<EvalType>(
                        eval->_grad_var_old_val);

                for (int qp = 0; qp < num_point; ++qp)
                {
                    double exp_val = 2.0 * (1.0 + step * (n_stages * n_iters));

                    if (old_value_type == OldValueType::LastStage)
                    {
                        exp_val = 2.0
                                  * (1.0 + (stage * n_iters)
                                     + (step * n_stages * n_iters));
                    }

                    if constexpr (Sacado::IsADType<scalar_type>::value)
                    {
                        exp_val = 0.0;
                    }

                    EXPECT_DOUBLE_EQ(exp_val, fieldValue(var_result, 0, qp));

                    for (int dim = 0; dim < num_space_dim; ++dim)
                    {
                        EXPECT_DOUBLE_EQ(
                            exp_val * (dim + 1.0),
                            fieldValue(grad_var_result, 0, qp, dim));
                    }
                }
            }
        }
    }
}

//---------------------------------------------------------------------------//
TEST(VariableOldValueLastTime, Residual2D)
{
    testEval<panzer::Traits::Residual, 2>(OldValueType::LastTime);
}

//---------------------------------------------------------------------------//
TEST(VariableOldValueLastTime, Jacobian2D)
{
    testEval<panzer::Traits::Jacobian, 2>(OldValueType::LastTime);
}

//---------------------------------------------------------------------------//
TEST(VariableOldValueLastTime, Residual3D)
{
    testEval<panzer::Traits::Residual, 3>(OldValueType::LastTime);
}

//---------------------------------------------------------------------------//
TEST(VariableOldValueLastTime, Jacobian3D)
{
    testEval<panzer::Traits::Jacobian, 3>(OldValueType::LastTime);
}

//---------------------------------------------------------------------------//
TEST(VariableOldValueLastStage, Residual2D)
{
    testEval<panzer::Traits::Residual, 2>(OldValueType::LastStage);
}

//---------------------------------------------------------------------------//
TEST(VariableOldValueLastStage, Jacobian2D)
{
    testEval<panzer::Traits::Jacobian, 2>(OldValueType::LastStage);
}

//---------------------------------------------------------------------------//
TEST(VariableOldValueLastStage, Residual3D)
{
    testEval<panzer::Traits::Residual, 3>(OldValueType::LastStage);
}

//---------------------------------------------------------------------------//
TEST(VariableOldValueLastStage, Jacobian3D)
{
    testEval<panzer::Traits::Jacobian, 3>(OldValueType::LastStage);
}

//---------------------------------------------------------------------------//

template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.type_name = "VariableOldValue";
    test_fixture.eval_name = "Var Old Value";
    test_fixture.model_params.set("Field Name", "Var");
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.template buildAndTest<
        ClosureModel::VariableOldValue<EvalType, panzer::Traits>,
        num_space_dim>();
}

TEST(VariableOldValueFactory, Residual2D)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(VariableOldValueFactory, Jacobian2D)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

} // end namespace Test
} // end namespace VertexCFD
