#include "VertexCFD_EvaluatorTestHarness.hpp"
#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "closure_models/VertexCFD_Closure_ConstantScalarField.hpp"

#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_SolidFullInductionLocalTimeStepSize.hpp"
#include "full_induction_mhd_solver/mhd_properties/VertexCFD_FullInductionMHDProperties.hpp"

#include "utils/VertexCFD_Utils_MagneticDim.hpp"
#include "utils/VertexCFD_Utils_MagneticLayout.hpp"

#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>

#include <gtest/gtest.h>

#include <limits>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace VertexCFD
{
namespace Test
{
//---------------------------------------------------------------------------//
// Test data dependencies.
template<class EvalType>
struct Dependencies : public PHX::EvaluatorWithBaseImpl<panzer::Traits>,
                      public PHX::EvaluatorDerived<EvalType, panzer::Traits>
{
    using scalar_type = typename EvalType::ScalarT;

    Kokkos::Array<double, 3> _h;

    PHX::MDField<double, panzer::Cell, panzer::Point, panzer::Dim> _element_length;
    PHX::MDField<double, panzer::Cell, panzer::Point> _solid_density;
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, MagneticDim>
        _tot_magn_field;

    Dependencies(const panzer::IntegrationRule& ir,
                 const Kokkos::Array<double, 3> h)
        : _h(h)
        , _element_length("element_length", ir.dl_vector)
        , _solid_density("solid_density", ir.dl_scalar)
        , _tot_magn_field(
              "total_magnetic_field",
              Utils::buildMagneticLayout(ir.dl_scalar, num_magnetic_field_dim))
    {
        this->addEvaluatedField(_element_length);
        this->addEvaluatedField(_solid_density);
        this->addEvaluatedField(_tot_magn_field);
        this->setName(
            "Solid Full Induction Local Time Step Size Unit Test "
            "Dependencies");
    }

    void evaluateFields(typename panzer::Traits::EvalData d) override
    {
        _solid_density.deep_copy(10.0);
        Kokkos::parallel_for(
            "test dependencies",
            Kokkos::RangePolicy<PHX::exec_space>(0, d.num_cells),
            *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(const int c) const
    {
        const int num_point = _element_length.extent(1);
        for (int qp = 0; qp < num_point; ++qp)
        {
            _element_length(c, qp, 0) = _h[0];
            _element_length(c, qp, 1) = _h[1];
            _element_length(c, qp, 2) = _h[2];
            _tot_magn_field(c, qp, 0) = 1.1;
            _tot_magn_field(c, qp, 1) = -1.2;
            _tot_magn_field(c, qp, 2) = 1.3;
        }
    }
};

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testEval()
{
    // Setup test fixture.
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    // Eval dependencies.
    const double nan_val = std::numeric_limits<double>::quiet_NaN();
    const Kokkos::Array<double, 3> h
        = {0.25, 0.5, num_space_dim == 3 ? 0.75 : nan_val};

    auto dep_eval
        = Teuchos::rcp(new Dependencies<EvalType>(*test_fixture.ir, h));
    test_fixture.registerEvaluator<EvalType>(dep_eval);

    const double c_h = num_space_dim == 2 ? 0.1 : 5.0;
    Teuchos::ParameterList full_induction_params;
    full_induction_params.set("Vacuum Magnetic Permeability", 0.1);
    full_induction_params.set("Build Magnetic Correction Potential Equation",
                              true);
    full_induction_params.set("Hyperbolic Divergence Cleaning Speed", c_h);
    const MHDProperties::FullInductionMHDProperties mhd_props(
        full_induction_params);

    // Create test evaluator.
    auto dt_eval = Teuchos::rcp(
        new ClosureModel::SolidFullInductionLocalTimeStepSize<EvalType,
                                                              panzer::Traits>(
            *test_fixture.ir, mhd_props));
    test_fixture.registerEvaluator<EvalType>(dt_eval);

    // Add required test fields.
    test_fixture.registerTestField<EvalType>(dt_eval->_local_dt);

    // Evaluate test fields.
    test_fixture.evaluate<EvalType>();

    // Check the test fields.
    auto local_dt_result
        = test_fixture.getTestFieldData<EvalType>(dt_eval->_local_dt);

    const int num_qp = num_space_dim == 2 ? 4 : 8;
    const double result = num_space_dim == 2 ? 0.14705882352941174
                                             : 0.027272727272727275;
    for (int qp = 0; qp < num_qp; ++qp)
        EXPECT_DOUBLE_EQ(result, fieldValue(local_dt_result, 0, qp));
}

//---------------------------------------------------------------------------//
TEST(SolidFullInductionLocalTimeStepSize2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>();
}

//---------------------------------------------------------------------------//
TEST(SolidFullInductionLocalTimeStepSize2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>();
}

//---------------------------------------------------------------------------//
TEST(SolidFullInductionLocalTimeStepSize3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>();
}

//---------------------------------------------------------------------------//
TEST(SolidFullInductionLocalTimeStepSize3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>();
}

//---------------------------------------------------------------------------//

template<class EvalType>
void testFactory(const bool build_magn_corr)
{
    static constexpr int num_space_dim = 3;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    test_fixture.type_name = "MagneticCorrectionDampingSource";
    test_fixture.eval_name = build_magn_corr
                                 ? "Solid Full Induction Local Time Step Size"
                                 : "Constant Scalar Field \"local_dt\"";
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Full Induction MHD Properties")
        .set("Vacuum Magnetic Permeability", 0.1)
        .set("Build Magnetic Correction Potential Equation", build_magn_corr)
        .set("Hyperbolic Divergence Cleaning Speed", 1.0);
    test_fixture.user_params.set("External Magnetic Field Value",
                                 Teuchos::Array<double>({0, 0, 0}));
    test_fixture.factory_type = "Solid Full Induction";
    test_fixture.num_evaluators = build_magn_corr ? 10 : 8;
    test_fixture.eval_index = 1;
    if (build_magn_corr)
    {
        test_fixture.template buildAndTest<
            ClosureModel::SolidFullInductionLocalTimeStepSize<EvalType,
                                                              panzer::Traits>,
            num_space_dim>();
    }
    else
    {
        test_fixture.template buildAndTest<
            ClosureModel::ConstantScalarField<EvalType, panzer::Traits>,
            num_space_dim>();
    }
}

TEST(SolidFullInductionLocalTimeStepSizeFactoryNoCleaning, Residual)
{
    testFactory<panzer::Traits::Residual>(false);
}

TEST(SolidFullInductionLocalTimeStepSizeFactoryNoCleaning, Jacobian)
{
    testFactory<panzer::Traits::Jacobian>(false);
}

TEST(SolidFullInductionLocalTimeStepSizeFactory, Residual)
{
    testFactory<panzer::Traits::Residual>(true);
}

TEST(SolidFullInductionLocalTimeStepSizeFactory, Jacobian)
{
    testFactory<panzer::Traits::Residual>(true);
}

} // end namespace Test
} // end namespace VertexCFD
