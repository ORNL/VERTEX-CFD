#include <VertexCFD_EvaluatorTestHarness.hpp>
#include <closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp>

#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_TotalMagneticField.hpp"

#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <gtest/gtest.h>

#include <string>

namespace VertexCFD
{
namespace Test
{
template<class EvalType>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>, 3>
        ind_magn_field;
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>, 3>
        ext_magn_field;

    const Kokkos::Array<double, 3> _bi;
    const Kokkos::Array<double, 3> _b0;

    Dependencies(const panzer::IntegrationRule& ir,
                 const Kokkos::Array<double, 3> bi,
                 const Kokkos::Array<double, 3> b0,
                 const std::string& field_prefix)
        : _bi(bi)
        , _b0(b0)
    {
        Utils::addEvaluatedVectorField(
            *this,
            ir.dl_scalar,
            ind_magn_field,
            field_prefix + "induced_magnetic_field_");
        Utils::addEvaluatedVectorField(
            *this, ir.dl_scalar, ext_magn_field, "external_magnetic_field_");
        this->setName("Total Magnetic Field Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData /**d**/) override
    {
        for (int dim = 0; dim < 3; ++dim)
        {
            ind_magn_field[dim].deep_copy(_bi[dim]);
            ext_magn_field[dim].deep_copy(_b0[dim]);
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval(const std::string& field_prefix)
{
    static constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 1;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Initialize induced / external  magnetic field components and
    // dependencies
    const Kokkos::Array<double, 3> bi
        = {0.25,
           0.5,
           num_space_dim > 2 ? 0.9 : std::numeric_limits<double>::quiet_NaN()};
    const Kokkos::Array<double, 3> b0 = {0.325, -0.65, 0.7};
    const Kokkos::Array<double, 3> b_exp
        = {bi[0] + b0[0],
           bi[1] + b0[1],
           num_space_dim == 2 ? b0[2] : b0[2] + bi[2]};

    // Eval dependencies
    const auto deps
        = Teuchos::rcp(new Dependencies<EvalType>(ir, bi, b0, field_prefix));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize and register
    const auto eval = [&]() {
        if (field_prefix == "")
        {
            // check evaluator with defaulted field name
            return Teuchos::rcp(
                new ClosureModel::TotalMagneticField<EvalType,
                                                     panzer::Traits,
                                                     num_space_dim>(ir));
        }
        else
        {
            // check evaluator with specified field name prefix
            return Teuchos::rcp(
                new ClosureModel::
                    TotalMagneticField<EvalType, panzer::Traits, num_space_dim>(
                        ir, field_prefix));
        }
    }();

    test_fixture.registerEvaluator<EvalType>(eval);

    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        test_fixture.registerTestField<EvalType>(
            eval->_total_magnetic_field[dim]);
    }
    test_fixture.evaluate<EvalType>();

    // Evaluate test fields
    const auto bt_0 = test_fixture.getTestFieldData<EvalType>(
        eval->_total_magnetic_field[0]);
    const auto bt_1 = test_fixture.getTestFieldData<EvalType>(
        eval->_total_magnetic_field[1]);
    const auto bt_2 = test_fixture.getTestFieldData<EvalType>(
        eval->_total_magnetic_field[2]);

    // Expected values
    const int num_point = ir.num_points;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_EQ(b_exp[0], fieldValue(bt_0, 0, qp));
        EXPECT_EQ(b_exp[1], fieldValue(bt_1, 0, qp));
        EXPECT_EQ(b_exp[2], fieldValue(bt_2, 0, qp));
    }
}

//-----------------------------------------------------------------//
TEST(TotalMagneticField2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>("");
}

//-----------------------------------------------------------------//
TEST(TotalMagneticField2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>("");
}

//-----------------------------------------------------------------//
TEST(TotalMagneticField3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>("");
}

//-----------------------------------------------------------------//
TEST(TotalMagneticField3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>("");
}
//-----------------------------------------------------------------//
TEST(TotalMagneticFieldWithFieldPrefix3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>("FOO_");
}

//-----------------------------------------------------------------//
TEST(TotalMagneticFieldWithFieldPrefix3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>("FOO_");
}

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    static constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    // The factory registers this closure and TotalMagneticFieldGradient for
    // type_name = "TotalMagneticField". This closure model will be the first
    // of the two registered evals
    test_fixture.num_evaluators = 2;
    test_fixture.eval_index = 0;
    test_fixture.type_name = "TotalMagneticField";
    test_fixture.eval_name = "Total Magnetic Field "
                             + std::to_string(num_space_dim) + "D";
    test_fixture.user_params.sublist("Full Induction MHD Properties");
    test_fixture.user_params.sublist("Fluid Properties")
        .set("Kinematic viscosity", 1.5)
        .set("Artificial compressibility", 0.1);
    test_fixture.template buildAndTest<
        ClosureModel::TotalMagneticField<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(TotalMagneticField_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(TotalMagneticField_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(TotalMagneticField_Factory3D, residual_test)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(TotalMagneticField_Factory3D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

//-----------------------------------------------------------------//
} // namespace Test
} // namespace VertexCFD