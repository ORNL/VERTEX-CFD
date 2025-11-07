#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_TotalMagneticFieldGradient.hpp"

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <gtest/gtest.h>

#include <string>

namespace VertexCFD
{
namespace Test
{
template<class EvalType, int NumSpaceDim>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;
    static constexpr int num_magnetic_field_dim = 3;

    Kokkos::Array<
        PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        grad_ind_magn_field;
    Kokkos::Array<
        PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_magnetic_field_dim>
        grad_ext_magn_field;

    const Kokkos::Array<double, num_magnetic_field_dim> _bi;
    const Kokkos::Array<double, num_magnetic_field_dim> _b0;

    Dependencies(const panzer::IntegrationRule& ir,
                 const Kokkos::Array<double, num_magnetic_field_dim> bi,
                 const Kokkos::Array<double, num_magnetic_field_dim> b0,
                 const std::string& gradient_prefix)
        : _bi(bi)
        , _b0(b0)
    {
        Utils::addEvaluatedVectorField(
            *this,
            ir.dl_vector,
            grad_ind_magn_field,
            gradient_prefix + "GRAD_induced_magnetic_field_");
        Utils::addEvaluatedVectorField(*this,
                                       ir.dl_vector,
                                       grad_ext_magn_field,
                                       "GRAD_external_magnetic_field_");
        this->setName("Total Magnetic Field Gradient Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "total magnetic field gradient test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = grad_ind_magn_field[0].extent(1);
        const int num_grad_dim = grad_ind_magn_field[0].extent(2);
        for (int qp = 0; qp < num_point; ++qp)
        {
            for (int dim = 0; dim < num_magnetic_field_dim; ++dim)
            {
                const double qpd = (qp + dim + 1);
                for (int grad_dim = 0; grad_dim < num_grad_dim; ++grad_dim)
                {
                    const double mult = qpd / (grad_dim + 1);
                    grad_ind_magn_field[dim](c, qp, grad_dim) = _bi[dim] * mult;
                    grad_ext_magn_field[dim](c, qp, grad_dim) = _b0[dim] * mult;
                }
            }
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval(const std::string& gradient_prefix)
{
    static constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Initialize induced / external magnetic field gradient base values
    static constexpr int num_magnetic_field_dim = 3;
    const Kokkos::Array<double, num_magnetic_field_dim> bi
        = {0.25,
           0.5,
           num_space_dim > 2 ? 0.9 : std::numeric_limits<double>::quiet_NaN()};
    const Kokkos::Array<double, num_magnetic_field_dim> b0
        = {0.325, -0.65, 0.7};

    // Eval dependencies
    const auto deps = Teuchos::rcp(new Dependencies<EvalType, num_space_dim>(
        ir, bi, b0, gradient_prefix));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize and register
    const auto eval = [&]() {
        if (gradient_prefix == "")
        {
            // check evaluator with defaulted field name
            return Teuchos::rcp(
                new ClosureModel::TotalMagneticFieldGradient<EvalType,
                                                             panzer::Traits,
                                                             num_space_dim>(ir));
        }
        else
        {
            // check evaluator with specified field name prefix
            return Teuchos::rcp(
                new ClosureModel::TotalMagneticFieldGradient<EvalType,
                                                             panzer::Traits,
                                                             num_space_dim>(
                    ir, gradient_prefix));
        }
    }();

    test_fixture.registerEvaluator<EvalType>(eval);

    test_fixture.registerTestField<EvalType>(eval->_grad_total_magnetic_field);
    test_fixture.registerTestField<EvalType>(
        eval->_divergence_total_magnetic_field);
    test_fixture.evaluate<EvalType>();

    // Evaluate test fields
    const auto grad_bt = test_fixture.getTestFieldData<EvalType>(
        eval->_grad_total_magnetic_field);
    const auto div_bt = test_fixture.getTestFieldData<EvalType>(
        eval->_divergence_total_magnetic_field);

    // Expected values
    const int num_point = ir.num_points;
    const int num_grad_dim = ir.spatial_dimension;

    EXPECT_EQ(num_magnetic_field_dim, grad_bt.extent(3));

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        double exp_div_b = 0.0;
        for (int fdim = 0; fdim < num_magnetic_field_dim; ++fdim)
        {
            // Currently force assumption of uniform external field, so the
            // expected total field gradient is just the induced field gradient
            const int qfd1 = qp + fdim + 1;
            for (int gdim = 0; gdim < num_grad_dim; ++gdim)
            {
                const double exp_db = num_space_dim < num_magnetic_field_dim
                                              && fdim == 2
                                          ? 0.0
                                          : bi[fdim] * qfd1 / (gdim + 1);
                EXPECT_DOUBLE_EQ(exp_db,
                                 fieldValue(grad_bt, 0, qp, gdim, fdim));
                if (gdim == fdim)
                    exp_div_b += exp_db;
            }
        }
        EXPECT_DOUBLE_EQ(exp_div_b, fieldValue(div_bt, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(TotalMagneticFieldGradient2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>("");
}

//-----------------------------------------------------------------//
TEST(TotalMagneticFieldGradient2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>("");
}

//-----------------------------------------------------------------//
TEST(TotalMagneticFieldGradient3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>("");
}

//-----------------------------------------------------------------//
TEST(TotalMagneticFieldGradient3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>("");
}
//-----------------------------------------------------------------//
TEST(TotalMagneticFieldGradientWithFieldPrefix3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>("FOO_");
}

//-----------------------------------------------------------------//
TEST(TotalMagneticFieldGradientWithFieldPrefix3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>("FOO_");
}

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    static constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    // Gradient closure is built with "TotalMagneticField" in the factory, and
    // will be the second of the two evaluators
    test_fixture.num_evaluators = 2;
    test_fixture.eval_index = 1;
    test_fixture.type_name = "TotalMagneticField";
    test_fixture.eval_name = "Total Magnetic Field Gradient"
                             + std::to_string(num_space_dim) + "D";
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Full Induction MHD Properties");
    test_fixture.factory_type = "Full Induction MHD";
    test_fixture.template buildAndTest<
        ClosureModel::TotalMagneticFieldGradient<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(TotalMagneticFieldGradient_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(TotalMagneticFieldGradient_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(TotalMagneticFieldGradient_Factory3D, residual_test)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(TotalMagneticFieldGradient_Factory3D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

//-----------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD
