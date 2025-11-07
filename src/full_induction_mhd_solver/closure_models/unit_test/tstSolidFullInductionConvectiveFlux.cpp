#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_SolidFullInductionConvectiveFlux.hpp"

#include "full_induction_mhd_solver/mhd_properties/VertexCFD_FullInductionMHDProperties.hpp"

#include "utils/VertexCFD_Utils_MagneticDim.hpp"
#include "utils/VertexCFD_Utils_MagneticLayout.hpp"

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

    const Kokkos::Array<double, num_magnetic_field_dim> _b;
    const double _psi;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, MagneticDim> _mag;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _magn_pot;

    Dependencies(const panzer::IntegrationRule& ir,
                 const Kokkos::Array<double, num_magnetic_field_dim>& b,
                 const double psi,
                 const std::string& field_pre)
        : _b(b)
        , _psi(psi)
        , _mag("total_magnetic_field",
               Utils::buildMagneticLayout(ir.dl_scalar, num_magnetic_field_dim))
        , _magn_pot(field_pre + "scalar_magnetic_potential", ir.dl_scalar)
    {
        this->addEvaluatedField(_mag);
        this->addEvaluatedField(_magn_pot);

        this->setName(
            "Solid Induction Convective Flux Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        _magn_pot.deep_copy(_psi);
        Kokkos::parallel_for(
            "test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = _mag.extent(1);
        for (int qp = 0; qp < num_point; ++qp)
        {
            for (int d = 0; d < num_magnetic_field_dim; ++d)
            {
                _mag(c, qp, d) = _b[d];
            }
        }
    }
};

template<class EvalType, int NumSpaceDim>
void testEval(const std::string& flux_pre, const std::string& field_pre)
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Initialize velocity components and dependents
    const Kokkos::Array<double, 3> b = {1.1, 2.1, 3.1};
    const double psi = 0.4;

    auto deps = Teuchos::rcp(
        new Dependencies<EvalType, num_space_dim>(ir, b, psi, field_pre));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize class object to test
    Teuchos::ParameterList full_induction_params;
    full_induction_params.set("Vacuum Magnetic Permeability", 0.05);
    full_induction_params.set("Build Magnetic Correction Potential Equation",
                              true);
    full_induction_params.set("Hyperbolic Divergence Cleaning Speed", 5.0);

    const MHDProperties::FullInductionMHDProperties mhd_props(
        full_induction_params);

    auto eval = Teuchos::rcp(
        new ClosureModel::SolidFullInductionConvectiveFlux<EvalType,
                                                           panzer::Traits,
                                                           num_space_dim>(
            ir, mhd_props, flux_pre, field_pre));

    test_fixture.registerEvaluator<EvalType>(eval);
    for (int dim = 0; dim < num_space_dim; ++dim)
    {
        test_fixture.registerTestField<EvalType>(eval->_induction_flux[dim]);
    }
    test_fixture.registerTestField<EvalType>(
        eval->_magnetic_correction_potential_flux);
    test_fixture.evaluate<EvalType>();

    // Expected values
    const double ch_times_psi = 2.;

    const double exp_ind_0_flux[3] = {ch_times_psi, 0.0, 0.0};
    const double exp_ind_1_flux[3] = {0.0, ch_times_psi, 0.0};
    const double exp_ind_2_flux[3] = {0.0, 0.0, ch_times_psi};

    const double exp_psi_flux[3] = {5.5, 10.5, 15.5};

    const auto& fc_ind_0
        = test_fixture.getTestFieldData<EvalType>(eval->_induction_flux[0]);
    const auto& fc_ind_1
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

            const auto fc_psi = test_fixture.getTestFieldData<EvalType>(
                eval->_magnetic_correction_potential_flux);
            EXPECT_DOUBLE_EQ(exp_psi_flux[dim], fieldValue(fc_psi, 0, qp, dim));
        }
    }
}

//-----------------------------------------------------------------//
TEST(SolidFullInductionConvectiveFlux2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>("", "");
}

//-----------------------------------------------------------------//
TEST(SolidFullInductionConvectiveFlux2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>("", "");
}

//-----------------------------------------------------------------//
TEST(SolidFullInductionConvectiveFlux3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>("", "");
}

//-----------------------------------------------------------------//
TEST(SolidFullInductionConvectiveFlux3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>("", "");
}

//-----------------------------------------------------------------//
TEST(SolidFullInductionConvectiveFluxPrefixed3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>("Foo_", "Bar_");
}

//-----------------------------------------------------------------//
TEST(SolidFullInductionConvectiveFluxPrefixed3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>("Foo_", "Bar_");
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
        .set("Build Magnetic Correction Potential Equation", true)
        .set("Hyperbolic Divergence Cleaning Speed", 1.0);
    test_fixture.factory_type = "Solid Full Induction";
    test_fixture.type_name = "MagneticCorrectionDampingSource";
    test_fixture.eval_name = "Solid Full Induction Convective Flux "
                             + std::to_string(num_space_dim) + "D";
    test_fixture.user_params.set("External Magnetic Field Value",
                                 Teuchos::Array<double>({0, 0, 0}));
    test_fixture.num_evaluators = 10;
    test_fixture.eval_index = 8;
    test_fixture.template buildAndTest<
        ClosureModel::SolidFullInductionConvectiveFlux<EvalType,
                                                       panzer::Traits,
                                                       num_space_dim>,
        num_space_dim>();
}

TEST(SolidFullInductionConvectiveFluxFactory, Residual)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(SolidFullInductionConvectiveFlux_Factory, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
