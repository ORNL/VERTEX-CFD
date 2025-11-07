#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_GodunovPowellSource.hpp"

#include "full_induction_mhd_solver/mhd_properties/VertexCFD_FullInductionMHDProperties.hpp"

#include "utils/VertexCFD_Utils_MagneticDim.hpp"
#include "utils/VertexCFD_Utils_MagneticLayout.hpp"
#include "utils/VertexCFD_Utils_VectorField.hpp"

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{

template<class EvalType, int NumSpaceDim>
struct Dependencies : public panzer::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    static constexpr int num_space_dim = NumSpaceDim;
    using scalar_type = typename EvalType::ScalarT;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> div_magn_field;

    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        velocity;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, MagneticDim>
        tot_magn_field;

    Dependencies(const panzer::IntegrationRule& ir)
        : div_magn_field("divergence_total_magnetic_field", ir.dl_scalar)
        , tot_magn_field(
              "total_magnetic_field",
              Utils::buildMagneticLayout(ir.dl_scalar, num_magnetic_field_dim))
    {
        this->addEvaluatedField(div_magn_field);
        Utils::addEvaluatedVectorField(
            *this, ir.dl_scalar, velocity, "velocity_");
        this->addEvaluatedField(tot_magn_field);

        this->setName("Godunov-Powell Source Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "godunov-powell source test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = div_magn_field.extent(1);
        using std::pow;

        for (int qp = 0; qp < num_point; ++qp)
        {
            div_magn_field(c, qp) = pow(-1.0, qp) * (qp + 1.1);
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                velocity[dim](c, qp) = pow(-0.6, dim) * (qp + dim + 1.2);
            }
            for (int dim = 0; dim < num_magnetic_field_dim; ++dim)
            {
                tot_magn_field(c, qp, dim) = pow(-0.9, dim + 1)
                                             * (qp + dim + 1.3);
            }
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

    auto& ir = *test_fixture.ir;

    // Initialize class object to test
    auto deps = Teuchos::rcp(new Dependencies<EvalType, num_space_dim>(ir));
    test_fixture.registerEvaluator<EvalType>(deps);

    Teuchos::ParameterList full_induction_params;
    full_induction_params.set("Vacuum Magnetic Permeability", 0.05);
    const MHDProperties::FullInductionMHDProperties mhd_props(
        full_induction_params);

    auto eval = Teuchos::rcp(
        new ClosureModel::GodunovPowellSource<EvalType,
                                              panzer::Traits,
                                              num_space_dim>(ir, mhd_props));
    test_fixture.registerEvaluator<EvalType>(eval);
    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        test_fixture.registerTestField<EvalType>(
            eval->_godunov_powell_momentum_source[dim]);
        test_fixture.registerTestField<EvalType>(
            eval->_godunov_powell_induction_source[dim]);
    }

    test_fixture.evaluate<EvalType>();

    const auto src_mom_0 = test_fixture.getTestFieldData<EvalType>(
        eval->_godunov_powell_momentum_source[0]);
    const auto src_mom_1 = test_fixture.getTestFieldData<EvalType>(
        eval->_godunov_powell_momentum_source[1]);

    const auto src_ind_0 = test_fixture.getTestFieldData<EvalType>(
        eval->_godunov_powell_induction_source[0]);
    const auto src_ind_1 = test_fixture.getTestFieldData<EvalType>(
        eval->_godunov_powell_induction_source[1]);

    const int num_point = ir.num_points;

    const double exp_src_mom[8][3] = {{25.74, -40.986, 52.9254},
                                      {-86.94, 112.266, -131.6574},
                                      {184.14, -215.946, 239.5494},
                                      {-317.34, 352.026, -376.6014},
                                      {486.54, -520.506, 542.8134},
                                      {-691.74, 721.386, -738.1854},
                                      {932.94, -954.666, 962.7174},
                                      {-1210.14, 1220.346, -1216.4094}};

    const double exp_src_ind[8][3] = {{-1.32, 1.452, -1.2672},
                                      {4.62, -4.032, 3.1752},
                                      {-9.92, 7.812, -5.8032},
                                      {17.22, -12.792, 9.1512},
                                      {-26.52, 18.972, -13.2192},
                                      {37.82, -26.352, 18.0072},
                                      {-51.12, 34.932, -23.5152},
                                      {66.42, -44.712, 29.7432}};

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_src_mom[qp][0], fieldValue(src_mom_0, 0, qp));
        EXPECT_DOUBLE_EQ(exp_src_mom[qp][1], fieldValue(src_mom_1, 0, qp));
        EXPECT_DOUBLE_EQ(exp_src_ind[qp][0], fieldValue(src_ind_0, 0, qp));
        EXPECT_DOUBLE_EQ(exp_src_ind[qp][1], fieldValue(src_ind_1, 0, qp));
        if (num_space_dim > 2)
        {
            const auto src_mom_2 = test_fixture.getTestFieldData<EvalType>(
                eval->_godunov_powell_momentum_source[2]);
            EXPECT_DOUBLE_EQ(exp_src_mom[qp][2], fieldValue(src_mom_2, 0, qp));
            const auto src_ind_2 = test_fixture.getTestFieldData<EvalType>(
                eval->_godunov_powell_induction_source[2]);
            EXPECT_DOUBLE_EQ(exp_src_ind[qp][2], fieldValue(src_ind_2, 0, qp));
        }
    }
};

//-----------------------------------------------------------------//
TEST(GodunovPowellSource2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>();
}

//-----------------------------------------------------------------//
TEST(GodunovPowellSource2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//-----------------------------------------------------------------//
TEST(GodunovPowellSource3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>();
}

//-----------------------------------------------------------------//
TEST(GodunovPowellSource3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Full Induction MHD Properties")
        .set("Vacuum Magnetic Permeability", 0.1)
        .set("Build Magnetic Correction Potential Equation", false);
    test_fixture.factory_type = "Full Induction MHD";
    test_fixture.type_name = "GodunovPowellSource";
    test_fixture.eval_name = "Godunov-Powell Source "
                             + std::to_string(num_space_dim) + "D";
    test_fixture.template buildAndTest<
        ClosureModel::GodunovPowellSource<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(GodunovPowellSource_Factory2D, residual_test)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(GodunovPowellSource_Factory2D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(GodunovPowellSource_Factory3D, residual_test)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(GodunovPowellSource_Factory3D, jacobian_test)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
