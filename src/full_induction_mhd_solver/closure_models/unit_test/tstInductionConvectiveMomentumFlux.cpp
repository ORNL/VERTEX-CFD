#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_InductionConvectiveMomentumFlux.hpp"

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
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    const Kokkos::Array<double, 3> _f_rhov;
    const Kokkos::Array<double, 3> _b;
    const double _p_mag;

    Kokkos::Array<
        PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        mtm_flux;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, MagneticDim> mag;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> magn_pres;

    Dependencies(const panzer::IntegrationRule& ir,
                 const Kokkos::Array<double, 3>& f_rhov,
                 const Kokkos::Array<double, 3>& b,
                 const double p_mag,
                 const std::string& flux_pre)
        : _f_rhov(f_rhov)
        , _b(b)
        , _p_mag(p_mag)
        , mag("total_magnetic_field",
              Utils::buildMagneticLayout(ir.dl_scalar, 3))
        , magn_pres("magnetic_pressure", ir.dl_scalar)
    {
        Utils::addEvaluatedVectorField(*this,
                                       ir.dl_vector,
                                       mtm_flux,
                                       flux_pre + "CONVECTIVE_FLUX_momentum_");
        this->addEvaluatedField(mag);
        this->addEvaluatedField(magn_pres);

        this->setName(
            "Induction Convective Momentum Flux Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "induction convective momentum flux test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = mag.extent(1);
        for (int qp = 0; qp < num_point; ++qp)
        {
            for (int field_dim = 0; field_dim < 3; ++field_dim)
            {
                mag(c, qp, field_dim) = _b[field_dim];
            }
            for (int vel_dim = 0; vel_dim < num_space_dim; ++vel_dim)
            {
                for (int flux_dim = 0; flux_dim < num_space_dim; ++flux_dim)
                {
                    mtm_flux[vel_dim](c, qp, flux_dim) = _f_rhov[flux_dim]
                                                         * (vel_dim + 1);
                }
            }
            magn_pres(c, qp) = _p_mag;
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval(const std::string& flux_pre)
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Initialize velocity components and dependents
    const Kokkos::Array<double, 3> f_rhov = {0.125, -0.26, 0.377};
    const Kokkos::Array<double, 3> b = {1.1, 2.1, 3.1};
    const double p_mag = 0.8;

    auto deps = Teuchos::rcp(new Dependencies<EvalType, num_space_dim>(
        ir, f_rhov, b, p_mag, flux_pre));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize class object to test
    Teuchos::ParameterList full_induction_params;
    full_induction_params.set("Vacuum Magnetic Permeability", 0.05);

    const MHDProperties::FullInductionMHDProperties mhd_props(
        full_induction_params);

    auto eval = Teuchos::rcp(
        new ClosureModel::InductionConvectiveMomentumFlux<EvalType,
                                                          panzer::Traits,
                                                          num_space_dim>(
            ir, mhd_props, flux_pre));

    test_fixture.registerEvaluator<EvalType>(eval);
    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        test_fixture.registerTestField<EvalType>(eval->_momentum_flux[dim]);
    }

    test_fixture.evaluate<EvalType>();

    // Expected values
    const double exp_mom_0_flux[3] = {-23.275, -46.46, -67.823};
    const double exp_mom_1_flux[3] = {-45.95, -87.92, -129.446};
    const double exp_mom_2_flux[3] = {-67.825, -130.98, -190.269};

    const auto fc_mom_0
        = test_fixture.getTestFieldData<EvalType>(eval->_momentum_flux[0]);
    const auto fc_mom_1
        = test_fixture.getTestFieldData<EvalType>(eval->_momentum_flux[1]);

    const int num_point = ir.num_points;
    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int dim = 0; dim < num_space_dim; dim++)
        {
            EXPECT_DOUBLE_EQ(exp_mom_0_flux[dim],
                             fieldValue(fc_mom_0, 0, qp, dim));
            EXPECT_DOUBLE_EQ(exp_mom_1_flux[dim],
                             fieldValue(fc_mom_1, 0, qp, dim));
            if (num_space_dim > 2)
            {
                const auto fc_mom_2 = test_fixture.getTestFieldData<EvalType>(
                    eval->_momentum_flux[2]);
                EXPECT_DOUBLE_EQ(exp_mom_2_flux[dim],
                                 fieldValue(fc_mom_2, 0, qp, dim));
            }
        }
    }
}

//-----------------------------------------------------------------//
TEST(InductionConvectiveMomentumFlux2D, residual_test)
{
    testEval<panzer::Traits::Residual, 2>("");
}

//-----------------------------------------------------------------//
TEST(InductionConvectiveMomentumFlux2D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 2>("");
}

//-----------------------------------------------------------------//
TEST(InductionConvectiveMomentumFluxPrefixed3D, residual_test)
{
    testEval<panzer::Traits::Residual, 3>("Foo_");
}

//-----------------------------------------------------------------//
TEST(InductionConvectiveMomentumFluxPrefixed3D, jacobian_test)
{
    testEval<panzer::Traits::Jacobian, 3>("Foo_");
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
    test_fixture.type_name = "InductionConvectiveFlux";
    test_fixture.eval_name = "Induction Convective Momentum Flux "
                             + std::to_string(num_space_dim) + "D";
    test_fixture.num_evaluators = 2;
    test_fixture.eval_index = 1;
    test_fixture.template buildAndTest<
        ClosureModel::InductionConvectiveMomentumFlux<EvalType,
                                                      panzer::Traits,
                                                      num_space_dim>,
        num_space_dim>();
}

TEST(InductionConvectiveMomentumFluxFactory2D, Residual)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(InductionConvectiveMomentumFluxFactory2D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(InductionConvectiveMomentumFluxFactory3D, Residual)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(InductionConvectiveMomentumFluxFactory3D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
