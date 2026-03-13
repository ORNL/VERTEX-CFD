#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_InductionConvectiveFlux.hpp"

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

    const Kokkos::Array<double, 3> _v;
    const Kokkos::Array<double, 3> _b;
    const double _psi;
    const double _p_mag;
    const bool _build_magn_corr;

    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        vel;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, MagneticDim> mag;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> magn_pot;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> magn_pres;

    Dependencies(const panzer::IntegrationRule& ir,
                 const Kokkos::Array<double, 3>& v,
                 const Kokkos::Array<double, 3>& b,
                 const double psi,
                 const double p_mag,
                 const bool build_magn_corr,
                 const std::string& field_pre)
        : _v(v)
        , _b(b)
        , _psi(psi)
        , _p_mag(p_mag)
        , _build_magn_corr(build_magn_corr)
        , mag("total_magnetic_field",
              Utils::buildMagneticLayout(ir.dl_scalar, 3))
        , magn_pot(field_pre + "scalar_magnetic_potential", ir.dl_scalar)
        , magn_pres("magnetic_pressure", ir.dl_scalar)
    {
        Utils::addEvaluatedVectorField(
            *this, ir.dl_scalar, vel, field_pre + "velocity_");
        this->addEvaluatedField(mag);
        this->addEvaluatedField(magn_pres);
        if (_build_magn_corr)
        {
            this->addEvaluatedField(magn_pot);
        }

        this->setName("Induction Convective Flux Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        magn_pres.deep_copy(_p_mag);
        for (int d = 0; d < num_space_dim; ++d)
            vel[d].deep_copy(_v[d]);
        if (_build_magn_corr)
            magn_pot.deep_copy(_psi);
        Kokkos::parallel_for(
            "induction convective flux test dependencies",
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
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval(const bool build_magn_corr,
              const std::string& flux_pre,
              const std::string& field_pre)
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Initialize velocity components and dependents
    const Kokkos::Array<double, 3> v = {1.25, 1.5, 1.125};
    const Kokkos::Array<double, 3> b = {1.1, 2.1, 3.1};
    const double psi = 0.4;
    const double p_mag = 0.8;

    auto deps = Teuchos::rcp(new Dependencies<EvalType, num_space_dim>(
        ir, v, b, psi, p_mag, build_magn_corr, field_pre));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize class object to test
    Teuchos::ParameterList full_induction_params;
    full_induction_params.set("Vacuum Magnetic Permeability", 0.05)
        .set("Build Magnetic Correction Potential Equation", build_magn_corr);
    if (build_magn_corr)
    {
        full_induction_params.set("Hyperbolic Divergence Cleaning Speed", 5.0);
    }

    const MHDProperties::FullInductionMHDProperties mhd_props(
        full_induction_params);

    auto eval = Teuchos::rcp(
        new ClosureModel::
            InductionConvectiveFlux<EvalType, panzer::Traits, num_space_dim>(
                ir, mhd_props, flux_pre, field_pre));

    test_fixture.registerEvaluator<EvalType>(eval);
    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        test_fixture.registerTestField<EvalType>(eval->_induction_flux[dim]);
    }
    if (build_magn_corr)
    {
        test_fixture.registerTestField<EvalType>(
            eval->_magnetic_correction_potential_flux);
    }

    test_fixture.evaluate<EvalType>();

    // Expected values
    const double psi_cont = build_magn_corr ? 2. : 0.;

    const double exp_ind_0_flux[3] = {psi_cont, -0.9749999999999999, -2.6375};
    const double exp_ind_1_flux[3] = {0.9749999999999999, psi_cont, -2.2875};
    const double exp_ind_2_flux[3] = {2.6375, 2.2875, psi_cont};

    const double exp_psi_flux[3] = {5.5, 10.5, 15.5};

    const auto fc_ind_0
        = test_fixture.getTestFieldData<EvalType>(eval->_induction_flux[0]);
    const auto fc_ind_1
        = test_fixture.getTestFieldData<EvalType>(eval->_induction_flux[1]);

    const int num_point = ir.num_points;
    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        for (int dim = 0; dim < num_space_dim; dim++)
        {
            EXPECT_DOUBLE_EQ(exp_ind_0_flux[dim],
                             fieldValue(fc_ind_0, 0, qp, dim));
            EXPECT_DOUBLE_EQ(exp_ind_1_flux[dim],
                             fieldValue(fc_ind_1, 0, qp, dim));
            if (num_space_dim > 2)
            {
                const auto fc_ind_2 = test_fixture.getTestFieldData<EvalType>(
                    eval->_induction_flux[2]);
                EXPECT_DOUBLE_EQ(exp_ind_2_flux[dim],
                                 fieldValue(fc_ind_2, 0, qp, dim));
            }

            if (build_magn_corr)
            {
                const auto fc_psi = test_fixture.getTestFieldData<EvalType>(
                    eval->_magnetic_correction_potential_flux);
                EXPECT_DOUBLE_EQ(exp_psi_flux[dim],
                                 fieldValue(fc_psi, 0, qp, dim));
            }
        }
    }
}

//-----------------------------------------------------------------//
TEST(InductionConvectiveFluxNoCleaning2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(false, "", "");
}

//-----------------------------------------------------------------//
TEST(InductionConvectiveFluxNoCleaning2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(false, "", "");
}

//-----------------------------------------------------------------//
TEST(InductionConvectiveFluxNoCleaning3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(false, "Foo_", "Bar_");
}

//-----------------------------------------------------------------//
TEST(InductionConvectiveFluxNoCleaning3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(false, "Foo_", "Bar_");
}

//-----------------------------------------------------------------//
TEST(InductionConvectiveFluxDivCleaning2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(true, "Foo_", "Bar_");
}

//-----------------------------------------------------------------//
TEST(InductionConvectiveFluxNoDivCleaning2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(true, "Foo_", "Bar_");
}

//-----------------------------------------------------------------//
TEST(InductionConvectiveFluxDivCleaning3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(true, "", "");
}

//-----------------------------------------------------------------//
TEST(InductionConvectiveFluxDivCleaning3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(true, "", "");
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
    test_fixture.eval_name = "Induction Convective Flux "
                             + std::to_string(num_space_dim) + "D";
    test_fixture.num_evaluators = 2;
    test_fixture.eval_index = 0;
    test_fixture.template buildAndTest<
        ClosureModel::InductionConvectiveFlux<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(InductionConvectiveFlux_Factory2D, Residual)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(InductionConvectiveFlux_Factory2D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(InductionConvectiveFlux_Factory3D, Residual)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(InductionConvectiveFlux_Factory3D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
