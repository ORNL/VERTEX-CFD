#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleCLSSign.hpp"

#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
template<class EvalType>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> phi;
    PHX::MDField<double, panzer::Cell, panzer::Point> phi_star;
    PHX::MDField<double, panzer::Cell, panzer::Point> epsilon;

    bool _eval_star_sign;

    Dependencies(const panzer::IntegrationRule& ir,
                 const std::string& field_prefix,
                 const bool eval_star_sign)
        : phi(field_prefix + "level_set", ir.dl_scalar)
        , phi_star(field_prefix + "STAR_level_set", ir.dl_scalar)
        , epsilon("CLS_epsilon", ir.dl_scalar)
        , _eval_star_sign(eval_star_sign)
    {
        this->addEvaluatedField(phi);
        if (_eval_star_sign)
            this->addEvaluatedField(phi_star);
        this->addEvaluatedField(epsilon);
        this->setName("Incompressible CLS Sign Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "Incompressible CLS sign test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = phi.extent(1);

        for (int qp = 0; qp < num_point; ++qp)
        {
            phi(c, qp) = -1.5 + qp;
            if (_eval_star_sign)
                phi_star(c, qp) = -1.25 + qp;
            epsilon(c, qp) = 0.9;
        }
    }
};

template<class EvalType>
void testEval(const bool eval_star_sign = true,
              const std::string& field_prefix = "")
{
    const int integration_order = 2;
    const int basis_order = 1;
    constexpr int num_space_dim = 2;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Eval dependencies
    const auto deps = Teuchos::rcp(
        new Dependencies<EvalType>(ir, field_prefix, eval_star_sign));
    test_fixture.registerEvaluator<EvalType>(deps);

    const auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleCLSSign<EvalType, panzer::Traits>(
            ir, field_prefix, eval_star_sign));
    test_fixture.registerEvaluator<EvalType>(eval);
    test_fixture.registerTestField<EvalType>(eval->_heaviside);
    test_fixture.registerTestField<EvalType>(eval->_sign);
    if (eval_star_sign)
        test_fixture.registerTestField<EvalType>(eval->_sign_star);

    test_fixture.evaluate<EvalType>();

    const auto fc_hv
        = test_fixture.getTestFieldData<EvalType>(eval->_heaviside);
    const auto fc_sg = test_fixture.getTestFieldData<EvalType>(eval->_sign);

    // Set expected values
    const double exp_hv[4]
        = {0.0, 0.06548520603898994, 0.93451479396101012, 1.0};
    const double exp_sg[4]
        = {-1.0, -0.86902958792202012, 0.86902958792202023, 1.0};
    const double exp_ss[4]
        = {-1.0, -0.52161729546191737, 0.99248825191354539, 1.0};

    const int num_point = ir.num_points;

    // Check on values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_hv[qp], fieldValue(fc_hv, 0, qp));
        EXPECT_DOUBLE_EQ(exp_sg[qp], fieldValue(fc_sg, 0, qp));
        if (eval_star_sign)
        {
            const auto fc_ss
                = test_fixture.getTestFieldData<EvalType>(eval->_sign_star);
            EXPECT_DOUBLE_EQ(exp_ss[qp], fieldValue(fc_ss, 0, qp));
        }
    }
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSSign, Residual)
{
    testEval<panzer::Traits::Residual>(false);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSSign, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(false);
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSSignPrefix, Residual)
{
    testEval<panzer::Traits::Residual>(true, "prefix_");
}

//---------------------------------------------------------------------------//
TEST(IncompressibleCLSSignPrefix, Jacobian)
{
    testEval<panzer::Traits::Jacobian>(true, "prefix_");
}

//---------------------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD
