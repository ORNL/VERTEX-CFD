#include "VertexCFD_EvaluatorTestHarness.hpp"

#include "closure_models/unit_test/VertexCFD_ClosureModelFactoryTestHarness.hpp"

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleConstantSource.hpp"

#include <Panzer_GlobalData.hpp>
#include <Panzer_ParameterLibrary.hpp>
#include <Panzer_ScalarParameterEntry.hpp>
#include <Teuchos_RCP.hpp>

#include <gtest/gtest.h>

namespace VertexCFD
{
namespace Test
{

template<class EvalType, int NumSpaceDim>
void testEval(const bool build_temp_equ,
              const bool const_vol_flow,
              const bool build_ind_less_mhd,
              const int flow_direction_idx = 0)
{
    constexpr int num_space_dim = NumSpaceDim;
    const int integration_order = 2;
    const int basis_order = 1;
    EvaluatorTestFixture test_fixture(
        num_space_dim, integration_order, basis_order);

    auto& ir = *test_fixture.ir;

    auto global_data = panzer::createGlobalData();
    // Initialize class object to test
    if (const_vol_flow)
    {
        // Wall Shear Stress

        auto& pl = *global_data->pl;
        std::string param_fam_name = "Volumetric Flow Rate - velocity_"
                                     + std::to_string(flow_direction_idx);
        pl.addParameterFamily(param_fam_name, true, false);
        pl.addParameterFamily(
            "Wall Shear Stress - wall_shear_stress", true, false);

        Teuchos::RCP<panzer::ScalarParameterEntry<EvalType>> wall_shear;
        wall_shear = Teuchos::rcp(new panzer::ScalarParameterEntry<EvalType>);
        wall_shear->setValue(4.0);

        Teuchos::RCP<panzer::ScalarParameterEntry<EvalType>> vol_flow;
        vol_flow = Teuchos::rcp(new panzer::ScalarParameterEntry<EvalType>);
        vol_flow->setValue(1.0);

        pl.addEntry<EvalType>("Wall Shear Stress - wall_shear_stress",
                              wall_shear);
        pl.addEntry<EvalType>(param_fam_name, vol_flow);

        if (build_ind_less_mhd)
        {
            param_fam_name = "Lorentz Force - VOLUMETRIC_SOURCE_momentum_"
                             + std::to_string(flow_direction_idx);
            pl.addParameterFamily(param_fam_name, true, false);
            Teuchos::RCP<panzer::ScalarParameterEntry<EvalType>> lorentz;
            lorentz = Teuchos::rcp(new panzer::ScalarParameterEntry<EvalType>);
            lorentz->setValue(2.0);
            pl.addEntry<EvalType>(param_fam_name, lorentz);
        }
    }

    Teuchos::Array<double> mom_input_source(num_space_dim);
    mom_input_source[0] = 0.1;
    mom_input_source[1] = 0.2;
    if (num_space_dim == 3)
        mom_input_source[2] = 0.3;

    Teuchos::ParameterList fluid_params;
    fluid_params.set("Build Temperature Equation", build_temp_equ);
    fluid_params.set("Build Inductionless MHD Equation", build_ind_less_mhd);

    Teuchos::ParameterList closure_params;
    if (const_vol_flow)
    {
        closure_params.set("Target Volumetric Flow Rate", 0.85);
        closure_params.set("Bottom Wall Surface Area", 0.7);
        closure_params.set("Inlet Wall Surface Area", 1.2);
        closure_params.set("Constant Volumetric Flow Rate", const_vol_flow);
        if (flow_direction_idx == 2)
            closure_params.set("Flow Direction", "z");
    }
    else
    {
        closure_params.set("Momentum Source", mom_input_source);
    }

    const double energy_input_source
        = build_temp_equ ? 0.4 : std::numeric_limits<double>::quiet_NaN();
    if (build_temp_equ)
    {
        closure_params.set("Energy Source", energy_input_source);
    }

    auto eval = Teuchos::rcp(
        new ClosureModel::IncompressibleConstantSource<EvalType,
                                                       panzer::Traits,
                                                       num_space_dim>(
            ir, fluid_params, global_data, closure_params));
    test_fixture.registerEvaluator<EvalType>(eval);
    for (int dim = 0; dim < num_space_dim; ++dim)
        test_fixture.registerTestField<EvalType>(eval->_momentum_source[dim]);

    test_fixture.evaluate<EvalType>();

    const auto fc_mom_0
        = test_fixture.getTestFieldData<EvalType>(eval->_momentum_source[0]);
    const auto fc_mom_1
        = test_fixture.getTestFieldData<EvalType>(eval->_momentum_source[1]);

    const double exp_mom[3]
        = {const_vol_flow ? (flow_direction_idx == 0
                                 ? (build_ind_less_mhd ? 6.678571428571428
                                                       : 8.095238095238095)
                                 : 0.0)
                          : mom_input_source[0],
           const_vol_flow ? (flow_direction_idx == 1
                                 ? (build_ind_less_mhd ? 6.678571428571428
                                                       : 8.095238095238095)
                                 : 0.0)
                          : mom_input_source[1],
           const_vol_flow ? (flow_direction_idx == 2
                                 ? (build_ind_less_mhd ? 6.678571428571428
                                                       : 8.095238095238095)
                                 : 0.0)
                          : mom_input_source[2]};

    const int num_point = ir.num_points;

    // Assert values
    for (int qp = 0; qp < num_point; ++qp)
    {
        EXPECT_DOUBLE_EQ(exp_mom[0], fieldValue(fc_mom_0, 0, qp));
        EXPECT_DOUBLE_EQ(exp_mom[1], fieldValue(fc_mom_1, 0, qp));
        if (num_space_dim > 2) // 3D mesh
        {
            const auto fc_mom_2 = test_fixture.getTestFieldData<EvalType>(
                eval->_momentum_source[2]);
            EXPECT_DOUBLE_EQ(exp_mom[2], fieldValue(fc_mom_2, 0, qp));
        }
        if (build_temp_equ)
        {
            const auto fc_energy = test_fixture.getTestFieldData<EvalType>(
                eval->_energy_source);
            EXPECT_EQ(energy_input_source, fieldValue(fc_energy, 0, qp));
        }
    }
}

//-----------------------------------------------------------------//
TEST(IncompressibleConstantSourceIsothermal2D, Residual)
{
    testEval<panzer::Traits::Residual, 2>(false, false, false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConstantSourceIsothermal2D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 2>(false, false, false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConstantSourceIsothermal3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(false, false, false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConstantSourceIsothermal3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(false, false, false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConstantSource3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(true, false, false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConstantSource3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(true, false, false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConstantSourceConstMassFlow3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(true, true, false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConstantSourceConstMassFlow3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(true, true, false);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConstantSourceConstMassFlowMHD3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(true, true, true);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConstantSourceConstMassFlowMHD3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(true, true, true);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConstantSourceConstMassFlowZDirectionMHD3D, Residual)
{
    testEval<panzer::Traits::Residual, 3>(true, true, true, 2);
}

//-----------------------------------------------------------------//
TEST(IncompressibleConstantSourceConstMassFlowZDirectionMHD3D, Jacobian)
{
    testEval<panzer::Traits::Jacobian, 3>(true, true, true, 2);
}

//-----------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void testFactory()
{
    constexpr int num_space_dim = NumSpaceDim;
    ClosureModelFactoryTestFixture<EvalType> test_fixture;
    const Teuchos::Array<double> mom_input_source(num_space_dim);
    test_fixture.model_params.set("Momentum Source", mom_input_source);
    test_fixture.closure_params.sublist(test_fixture.model_id)
        .sublist("Fluid Properties")
        .set("Kinematic viscosity", 0.1)
        .set("Artificial compressibility", 2.0);
    test_fixture.type_name = "IncompressibleConstantSource";
    test_fixture.eval_name = "Incompressible Constant Source "
                             + std::to_string(num_space_dim) + "D";
    test_fixture.template buildAndTest<
        ClosureModel::IncompressibleConstantSource<EvalType, panzer::Traits, num_space_dim>,
        num_space_dim>();
}

TEST(IncompressibleConstantSource_Factory2D, Residual)
{
    testFactory<panzer::Traits::Residual, 2>();
}

TEST(IncompressibleConstantSource_Factory2D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 2>();
}

TEST(IncompressibleConstantSource_Factory3D, Residual)
{
    testFactory<panzer::Traits::Residual, 3>();
}

TEST(IncompressibleConstantSource_Factory3D, Jacobian)
{
    testFactory<panzer::Traits::Jacobian, 3>();
}

} // namespace Test
} // namespace VertexCFD
