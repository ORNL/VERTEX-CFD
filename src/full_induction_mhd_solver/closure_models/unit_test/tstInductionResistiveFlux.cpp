#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_InductionResistiveFlux.hpp"

#include "full_induction_mhd_solver/mhd_properties/VertexCFD_FullInductionMHDProperties.hpp"

#include "utils/VertexCFD_Utils_MagneticDim.hpp"
#include "utils/VertexCFD_Utils_MagneticLayout.hpp"

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

    const Kokkos::Array<double, 3> _b;
    const double _res;

    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, MagneticDim> mag;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim, MagneticDim>
        grad_mag;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> div_mag;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> eta;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim> grad_eta;

    Dependencies(const panzer::IntegrationRule& ir,
                 const Kokkos::Array<double, 3>& b,
                 const double res,
                 const std::string& grad_pre)
        : _b(b)
        , _res(res)
        , mag("total_magnetic_field",
              Utils::buildMagneticLayout(ir.dl_scalar, 3))
        , grad_mag(grad_pre + "GRAD_total_magnetic_field",
                   Utils::buildMagneticGradLayout(ir.dl_vector, 3))
        , div_mag(grad_pre + "divergence_total_magnetic_field", ir.dl_scalar)
        , eta("resistivity", ir.dl_scalar)
        , grad_eta("GRAD_resistivity", ir.dl_vector)
    {
        this->addEvaluatedField(mag);
        this->addEvaluatedField(grad_mag);
        this->addEvaluatedField(div_mag);
        this->addEvaluatedField(eta);
        this->addEvaluatedField(grad_eta);

        this->setName("Induction Resistive Flux Unit Test Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        Kokkos::parallel_for(
            "induction resistive flux test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = grad_eta.extent(1);
        const int num_space_dim = grad_eta.extent(2);

        for (int qp = 0; qp < num_point; ++qp)
        {
            for (int field_dim = 0; field_dim < 3; ++field_dim)
            {
                mag(c, qp, field_dim) = _b[field_dim];
            }
            eta(c, qp) = _res;
            div_mag(c, qp) = 0.0;
            for (int grad_dim = 0; grad_dim < num_space_dim; ++grad_dim)
            {
                const int b_mult = pow(-1, grad_dim) * (grad_dim + 1);
                for (int field_dim = 0; field_dim < 3; ++field_dim)
                {
                    grad_mag(c, qp, grad_dim, field_dim) = _b[field_dim]
                                                           * b_mult;
                }
                div_mag(c, qp) += grad_mag(c, qp, grad_dim, grad_dim);
                const int eta_mult = pow(-1, grad_dim + 1) * (grad_dim + 2);
                grad_eta(c, qp, grad_dim) = _res * eta_mult;
            }
        }
    }
};

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval(const Teuchos::ParameterList& test_params,
              const std::string& flux_pre,
              const std::string& grad_pre)
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    const auto& ir = *test_fixture.ir;

    // Initialize velocity components and dependents
    const Kokkos::Array<double, 3> b = {1.1, 2.1, 3.1};
    const double eta = 0.16;

    auto deps = Teuchos::rcp(new Dependencies<EvalType>(ir, b, eta, grad_pre));
    test_fixture.registerEvaluator<EvalType>(deps);

    // Initialize class object to test
    Teuchos::ParameterList full_induction_params
        = test_params.sublist("Full Induction MHD Properties");
    full_induction_params.set("Vacuum Magnetic Permeability", 0.05)
        .set("Build Resistive Flux", true);
    const MHDProperties::FullInductionMHDProperties mhd_props(
        full_induction_params);
    const auto build_magn_corr = mhd_props.buildMagnCorr();

    auto eval = Teuchos::rcp(
        new ClosureModel::
            InductionResistiveFlux<EvalType, panzer::Traits, num_space_dim>(
                ir, mhd_props, flux_pre, grad_pre));

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

    const auto exp_ind_0_flux
        = test_params.get<Teuchos::Array<double>>("Exp Ind 0 Flux");
    const auto exp_ind_1_flux
        = test_params.get<Teuchos::Array<double>>("Exp Ind 1 Flux");
    const auto exp_ind_2_flux
        = test_params.get<Teuchos::Array<double>>("Exp Ind 2 Flux");

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
                EXPECT_DOUBLE_EQ(0.0, fieldValue(fc_psi, 0, qp, dim));
            }
        }
    }
}

//-----------------------------------------------------------------//
template<class EvalType>
void testEval2D(const bool build_magn_corr, const bool var_resistivity)
{
    // Expected values 2D
    // Variable resistivity
    const double nanval = std::numeric_limits<double>::signaling_NaN();
    const Teuchos::Array<double> exp_ind_0_flux_var_eta(
        {-6.72, -20.48, nanval});
    const Teuchos::Array<double> exp_ind_1_flux_var_eta({17.28, 3.52, nanval});
    // Constant resistivity
    const Teuchos::Array<double> exp_ind_0_flux_const_eta(
        {13.44, -7.04, nanval});
    const Teuchos::Array<double> exp_ind_1_flux_const_eta(
        {6.72, -3.52, nanval});

    const auto exp_ind_0_flux = var_resistivity ? exp_ind_0_flux_var_eta
                                                : exp_ind_0_flux_const_eta;
    const auto exp_ind_1_flux = var_resistivity ? exp_ind_1_flux_var_eta
                                                : exp_ind_1_flux_const_eta;
    // NOTE: currently we will not have a flux vector for B_z
    // (exp_ind_2_flux) as we are limiting the induced magnetic field to the
    // mesh dimension. The fact that the reference script does compute non-zero
    // expected values for the B_z flux suggests that perhaps we do need to
    // always include the B_z component.
    const Teuchos::Array<double> exp_ind_2_flux({nanval, nanval, nanval});

    Teuchos::ParameterList full_ind_params;
    full_ind_params
        .set("Build Magnetic Correction Potential Equation", build_magn_corr)
        .set("Variable Resistivity", var_resistivity);
    if (build_magn_corr)
    {
        full_ind_params.set("Hyperbolic Divergence Cleaning Speed", 1.1);
    }
    if (!var_resistivity)
    {
        full_ind_params.set("Resistivity", 1.5);
    }
    Teuchos::ParameterList test_params;
    test_params.set("Full Induction MHD Properties", full_ind_params);
    test_params.set("Exp Ind 0 Flux", exp_ind_0_flux);
    test_params.set("Exp Ind 1 Flux", exp_ind_1_flux);
    test_params.set("Exp Ind 2 Flux", exp_ind_2_flux);

    testEval<EvalType, 2>(test_params, "Foo_", "Bar_");
}

//-----------------------------------------------------------------//
TEST(InductionResistiveFluxNoCleaning2D, Residual)
{
    // For now, tests with variable resistivity will throw an error.
    // Test for the appropriate error here, but retain the old test
    // for use once a resistivity closure has been added.
    const std::string exp_msg
        = "No closure models currently exist to evaluate variable "
          "resistivity. Use a constant resistivity only.";
    EXPECT_THROW(
        try {
            testEval2D<panzer::Traits::Residual>(false, true);
        } catch (const std::runtime_error& e) {
            EXPECT_EQ(exp_msg, e.what());
            throw;
        },
        std::runtime_error);
    // testEval2D<panzer::Traits::Residual>(false, true);
}

TEST(InductionResistiveFluxNoCleaning2D, Jacobian)
{
    // For now, tests with variable resistivity will throw an error.
    // Test for the appropriate error here, but retain the old test
    // for use once a resistivity closure has been added.
    const std::string exp_msg
        = "No closure models currently exist to evaluate variable "
          "resistivity. Use a constant resistivity only.";
    EXPECT_THROW(
        try {
            testEval2D<panzer::Traits::Jacobian>(false, true);
        } catch (const std::runtime_error& e) {
            EXPECT_EQ(exp_msg, e.what());
            throw;
        },
        std::runtime_error);
    // testEval2D<panzer::Traits::Jacobian>(false, true);
}

//-----------------------------------------------------------------//
TEST(InductionResistiveFluxDivCleaning2D, Residual)
{
    // For now, tests with variable resistivity will throw an error.
    // Test for the appropriate error here, but retain the old test
    // for use once a resistivity closure has been added.
    const std::string exp_msg
        = "No closure models currently exist to evaluate variable "
          "resistivity. Use a constant resistivity only.";
    EXPECT_THROW(
        try {
            testEval2D<panzer::Traits::Residual>(true, true);
        } catch (const std::runtime_error& e) {
            EXPECT_EQ(exp_msg, e.what());
            throw;
        },
        std::runtime_error);
    // testEval2D<panzer::Traits::Residual>(true, true);
}

TEST(InductionResistiveFluxDivCleaning2D, Jacobian)
{
    // For now, tests with variable resistivity will throw an error.
    // Test for the appropriate error here, but retain the old test
    // for use once a resistivity closure has been added.
    const std::string exp_msg
        = "No closure models currently exist to evaluate variable "
          "resistivity. Use a constant resistivity only.";
    EXPECT_THROW(
        try {
            testEval2D<panzer::Traits::Jacobian>(true, true);
        } catch (const std::runtime_error& e) {
            EXPECT_EQ(exp_msg, e.what());
            throw;
        },
        std::runtime_error);
    // testEval2D<panzer::Traits::Jacobian>(true, true);
}

//-----------------------------------------------------------------//
TEST(InductionResistiveFluxConstantResistivity2D, Residual)
{
    testEval2D<panzer::Traits::Residual>(false, false);
}

TEST(InductionResistiveFluxConstantResistivity2D, Jacobian)
{
    testEval2D<panzer::Traits::Jacobian>(false, false);
}

//-----------------------------------------------------------------//
TEST(InductionResistiveFluxConstantResistivityMagnCorr2D, Residual)
{
    testEval2D<panzer::Traits::Residual>(true, false);
}

TEST(InductionResistiveFluxConstantResistivityMagnCorr2D, Jacobian)
{
    testEval2D<panzer::Traits::Jacobian>(true, false);
}

//-----------------------------------------------------------------//
template<class EvalType>
void testEval3D(const bool build_magn_corr, const bool var_resistivity)
{
    // Expected values 3D
    // Variable resistivity
    const Teuchos::Array<double> exp_ind_0_flux_var_eta({3.2, -20.48, -9.28});
    const Teuchos::Array<double> exp_ind_1_flux_var_eta({17.28, 13.44, 49.92});
    const Teuchos::Array<double> exp_ind_2_flux_var_eta({-4.16, -46.72, -3.2});
    // Constant resistivity
    const Teuchos::Array<double> exp_ind_0_flux_const_eta(
        {-16.32, -7.04, 10.56});
    const Teuchos::Array<double> exp_ind_1_flux_const_eta(
        {6.72, -33.28, 20.16});
    const Teuchos::Array<double> exp_ind_2_flux_const_eta({9.92, -19.84, 9.92});

    const auto exp_ind_0_flux = var_resistivity ? exp_ind_0_flux_var_eta
                                                : exp_ind_0_flux_const_eta;
    const auto exp_ind_1_flux = var_resistivity ? exp_ind_1_flux_var_eta
                                                : exp_ind_1_flux_const_eta;
    const auto exp_ind_2_flux = var_resistivity ? exp_ind_2_flux_var_eta
                                                : exp_ind_2_flux_const_eta;

    Teuchos::ParameterList full_ind_params;
    full_ind_params
        .set("Build Magnetic Correction Potential Equation", build_magn_corr)
        .set("Variable Resistivity", var_resistivity);
    if (build_magn_corr)
    {
        full_ind_params.set("Hyperbolic Divergence Cleaning Speed", 1.1);
    }
    if (!var_resistivity)
    {
        full_ind_params.set("Resistivity", 1.5);
    }
    Teuchos::ParameterList test_params;
    test_params.set("Full Induction MHD Properties", full_ind_params);
    test_params.set("Exp Ind 0 Flux", exp_ind_0_flux);
    test_params.set("Exp Ind 1 Flux", exp_ind_1_flux);
    test_params.set("Exp Ind 2 Flux", exp_ind_2_flux);

    testEval<EvalType, 3>(test_params, "", "");
}

//-----------------------------------------------------------------//
TEST(InductionResistiveFluxNoCleaning3D, Residual)
{
    // For now, tests with variable resistivity will throw an error.
    // Test for the appropriate error here, but retain the old test
    // for use once a resistivity closure has been added.
    const std::string exp_msg
        = "No closure models currently exist to evaluate variable "
          "resistivity. Use a constant resistivity only.";
    EXPECT_THROW(
        try {
            testEval3D<panzer::Traits::Residual>(false, true);
        } catch (const std::runtime_error& e) {
            EXPECT_EQ(exp_msg, e.what());
            throw;
        },
        std::runtime_error);
    // testEval3D<panzer::Traits::Residual>(false, true);
}

TEST(InductionResistiveFluxNoCleaning3D, Jacobian)
{
    // For now, tests with variable resistivity will throw an error.
    // Test for the appropriate error here, but retain the old test
    // for use once a resistivity closure has been added.
    const std::string exp_msg
        = "No closure models currently exist to evaluate variable "
          "resistivity. Use a constant resistivity only.";
    EXPECT_THROW(
        try {
            testEval3D<panzer::Traits::Jacobian>(false, true);
        } catch (const std::runtime_error& e) {
            EXPECT_EQ(exp_msg, e.what());
            throw;
        },
        std::runtime_error);
    // testEval3D<panzer::Traits::Jacobian>(false, true);
}

//-----------------------------------------------------------------//
TEST(InductionResistiveFluxDivCleaning3D, Residual)
{
    // For now, tests with variable resistivity will throw an error.
    // Test for the appropriate error here, but retain the old test
    // for use once a resistivity closure has been added.
    const std::string exp_msg
        = "No closure models currently exist to evaluate variable "
          "resistivity. Use a constant resistivity only.";
    EXPECT_THROW(
        try {
            testEval3D<panzer::Traits::Residual>(true, true);
        } catch (const std::runtime_error& e) {
            EXPECT_EQ(exp_msg, e.what());
            throw;
        },
        std::runtime_error);
    // testEval3D<panzer::Traits::Residual>(true, true);
}

TEST(InductionResistiveFluxDivCleaning3D, Jacobian)
{
    // For now, tests with variable resistivity will throw an error.
    // Test for the appropriate error here, but retain the old test
    // for use once a resistivity closure has been added.
    const std::string exp_msg
        = "No closure models currently exist to evaluate variable "
          "resistivity. Use a constant resistivity only.";
    EXPECT_THROW(
        try {
            testEval3D<panzer::Traits::Jacobian>(true, true);
        } catch (const std::runtime_error& e) {
            EXPECT_EQ(exp_msg, e.what());
            throw;
        },
        std::runtime_error);
    // testEval3D<panzer::Traits::Jacobian>(true, true);
}

//-----------------------------------------------------------------//
TEST(InductionResistiveFluxConstantResistivity3D, Residual)
{
    testEval3D<panzer::Traits::Residual>(false, false);
}

TEST(InductionResistiveFluxConstantResistivity3D, Jacobian)
{
    testEval3D<panzer::Traits::Jacobian>(false, false);
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
    test_fixture.type_name = "InductionResistiveFlux";
    test_fixture.eval_name = "Induction Resistive Flux "
                             + std::to_string(num_space_dim) + "D";
    test_fixture.template buildAndTest<
        ClosureModel::InductionResistiveFlux<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(InductionResistiveFlux_Factory2D, Residual)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(InductionResistiveFlux_Factory2D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(InductionResistiveFlux_Factory3D, Residual)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(InductionResistiveFlux_Factory3D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
