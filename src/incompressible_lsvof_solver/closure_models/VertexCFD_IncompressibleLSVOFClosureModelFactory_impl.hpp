#ifndef VERTEXCFD_INCOMPRESSIBLELSVOFCLOSUREMODELFACTORY_IMPL_HPP
#define VERTEXCFD_INCOMPRESSIBLELSVOFCLOSUREMODELFACTORY_IMPL_HPP

#include "utils/VertexCFD_Utils_PhaseIndex.hpp"
#include "utils/VertexCFD_Utils_ScalarToVector.hpp"

#include "closure_models/VertexCFD_Closure_ConstantVectorField.hpp"
#include "closure_models/VertexCFD_Closure_MetricTensor.hpp"
#include "closure_models/VertexCFD_Closure_MetricTensorElementLength.hpp"
#include "closure_models/VertexCFD_Closure_VariableConvectiveFlux.hpp"
#include "closure_models/VertexCFD_Closure_VariableOldValue.hpp"

#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleCLSEpsilon.hpp"
#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleCLSLambda.hpp"
#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleCLSNonReconstructedNormal.hpp"
#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleCLSRegularization.hpp"
#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleCLSSign.hpp"
#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleCLSTimeDerivative.hpp"
#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFArtificialCompression.hpp"
#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFBubbleExact.hpp"
#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFBuoyancySource.hpp"
#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFConvectiveFlux.hpp"
#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFEntropyFluctuation.hpp"
#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFEntropyFunction.hpp"
#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFEntropyViscosity.hpp"
#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFErrorNorms.hpp"
#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFNthPhaseFraction.hpp"
#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFScalarConvectiveFlux.hpp"
#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFScalarTimeDerivative.hpp"
#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFScalarViscousFlux.hpp"
#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFSurfaceTensionForce.hpp"
#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFTimeDerivative.hpp"
#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFVariableProperties.hpp"
#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFViscousFlux.hpp"
#include "incompressible_lsvof_solver/closure_models/VertexCFD_Closure_IncompressibleLSVOFVortexVelocity.hpp"

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleLocalTimeStepSize.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void IncompressibleLSVOFFactory<EvalType, NumSpaceDim>::buildClosureModel(
    const std::string& closure_type,
    const Teuchos::RCP<panzer::IntegrationRule>& ir,
    const Teuchos::ParameterList& /*user_params*/,
    const Teuchos::ParameterList& closure_params,
    const Teuchos::RCP<panzer::GlobalData>& /*global_data*/,
    bool& found_model,
    std::string& error_msg,
    Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>>
        evaluators)
{
    // Check if closure is built by default
    if (isDefaultClosureModel(closure_type))
    {
        found_model = true;
    }
    // Build optional closures
    else
    {
        // Constant vortex velocity field
        if (closure_type == "IncompressibleLSVOFVortexVelocity")
        {
            const auto eval_vof_vortex = Teuchos::rcp(
                new IncompressibleLSVOFVortexVelocity<EvalType,
                                                      panzer::Traits,
                                                      num_space_dim>(
                    *ir, closure_params));

            evaluators->push_back(eval_vof_vortex);

            found_model = true;
        }

        // Constant Vector Field
        if (closure_type == "IncompressibleLSVOFConstantVectorField")
        {
            auto eval = Teuchos::rcp(
                new ConstantVectorField<EvalType, panzer::Traits, num_space_dim>(
                    *ir, closure_params));
            evaluators->push_back(eval);
            found_model = true;
        }

        // Error norms
        if (closure_type == "IncompressibleLSVOFErrorNorms")
        {
            // Create error evaluator
            const auto eval_vof_errors = Teuchos::rcp(
                new IncompressibleLSVOFErrorNorms<EvalType,
                                                  panzer::Traits,
                                                  num_space_dim>(
                    *ir, closure_params));

            evaluators->push_back(eval_vof_errors);

            found_model = true;
        }

        // Bubble exact solution
        if (closure_type == "IncompressibleLSVOFBubbleExact")
        {
            const auto eval_bubble_exact = Teuchos::rcp(
                new IncompressibleLSVOFBubbleExact<EvalType,
                                                   panzer::Traits,
                                                   num_space_dim>(
                    *ir, closure_params));

            evaluators->push_back(eval_bubble_exact);

            found_model = true;
        }
    }

    // If model not found, add to error message and return
    error_msg += "Default IncompressibleLSVOF closure models:\n";
    error_msg += "IncompressibleCLSEpsilon\n";
    error_msg += "IncompressibleCLSLambda\n";
    error_msg += "IncompressibleCLSNonReconstructedNormal\n";
    error_msg += "IncompressibleCLSRegularization\n";
    error_msg += "IncompressibleCLSSign\n";
    error_msg += "IncompressibleCLSTimeDerivative\n";
    error_msg += "IncompressibleLocalTimeStepSize\n";
    error_msg += "IncompressibleLSVOFArtificialCompression\n";
    error_msg += "IncompressibleLSVOFBuoyancySource\n";
    error_msg += "IncompressibleLSVOFConvectiveFlux\n";
    error_msg += "IncompressibleLSVOFEntropyFluctuation\n";
    error_msg += "IncompressibleLSVOFEntropyFunction\n";
    error_msg += "IncompressibleLSVOFEntropyViscosity\n";
    error_msg += "IncompressibleLSVOFNthPhaseFraction\n";
    error_msg += "IncompressibleLSVOFScalarConvectiveFlux\n";
    error_msg += "IncompressibleLSVOFSurfaceTensionForce\n";
    error_msg += "IncompressibleLSVOFScalarViscousFlux\n";
    error_msg += "IncompressibleLSVOFTimeDerivative\n";
    error_msg += "IncompressibleLSVOFVariableProperties\n";
    error_msg += "IncompressibleLSVOFViscousFlux\n";
    // Add optional closure models
    error_msg += "\nOptional IncompressibleLSVOF closure models:\n";
    error_msg += "IncompressibleLSVOFBubbleExact\n";
    error_msg += "IncompressibleLSVOFConstantVectorField\n";
    error_msg += "IncompressibleLSVOFErrorNorms\n";
    error_msg += "IncompressibleLSVOFVortexVelocity\n";
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void IncompressibleLSVOFFactory<EvalType, NumSpaceDim>::buildDefaultClosureModels(
    const Teuchos::RCP<panzer::IntegrationRule>& ir,
    const Teuchos::ParameterList& closure_model_list,
    const Teuchos::ParameterList& user_params,
    const Teuchos::RCP<panzer::GlobalData>& global_data,
    Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>>
        evaluators)
{
    // Get closure properties from closure model list
    Teuchos::ParameterList lsvof_params
        = closure_model_list.sublist("LSVOF_Properties");

    // Check that model type is valid
    const auto type_validator = Teuchos::rcp(
        new Teuchos::StringToIntegralParameterEntryValidator<LSVOFModelType>(
            Teuchos::tuple<std::string>("None", "CLS", "VOF"), "None"));

    const LSVOFModelType lsvof_model_name = type_validator->getIntegralValue(
        lsvof_params.get<std::string>("LSVOF Model"));

    if (lsvof_model_name == LSVOFModelType::None)
    {
        std::string msg = "Error: LSVOF Model cannot be missing";
        msg += " or set to 'None' when  Closure Factory Type is set to 'LSVOF'.";

        throw std::runtime_error(msg);
    }

    // Boolean for solving just LSVOF scalar equations
    const bool build_lsvofmom_equ
        = lsvof_params.isType<bool>("Build LSVOF Navier-Stokes Equations")
              ? lsvof_params.get<bool>("Build LSVOF Navier-Stokes Equations")
              : true;

    // Number of phases
    const int num_phases = lsvof_params.get<int>("Number of Phases");
    const int num_lsvof_dofs = num_phases - 1;

    if (num_phases < 2)
    {
        const std::string msg = "Must have 2 or more phases for LSVOF model.";

        throw std::runtime_error(msg);
    }
    else if (num_phases > 2)
    {
        const std::string msg
            = "More than 2 phases not currently supported in LSVOF model.";

        throw std::runtime_error(msg);
    }

    // List of phase names
    std::vector<std::string> phase_names;

    // Booleans for considering buoyancy force and surface tension
    const bool _build_lsvof_buoyancy_source
        = lsvof_params.get<bool>("Build LSVOF Buoyancy Source", true);

    const bool _build_lsvof_surface_tension
        = lsvof_params.get<bool>("Build LSVOF Surface Tension", true);

    // Add closure models for LSVOF transport equations here
    if (lsvof_model_name == LSVOFModelType::VOF)
    {
        for (int n = 1; n <= num_phases; ++n)
        {
            Teuchos::ParameterList vof_names_list;

            const std::string list_name = "Phase " + std::to_string(n);

            Teuchos::ParameterList phase_list = lsvof_params.sublist(list_name);

            const std::string phase_name
                = phase_list.get<std::string>("Phase Name");
            const std::string phase_fraction_name = "alpha_" + phase_name;

            vof_names_list.set("Field Name", phase_fraction_name);
            vof_names_list.set("Equation Name",
                               phase_fraction_name + "_equation");

            // Create closure models for N - 1 phase fraction transport
            // equations
            if (n < num_phases)
            {
                phase_names.push_back(phase_fraction_name);

                // Time derivative
                const auto eval_vof_dqdt = Teuchos::rcp(
                    new IncompressibleLSVOFScalarTimeDerivative<EvalType,
                                                                panzer::Traits>(
                        *ir,
                        phase_fraction_name,
                        phase_fraction_name + "_equation"));

                evaluators->push_back(eval_vof_dqdt);

                // Convective flux
                const auto eval_vof_conv = Teuchos::rcp(
                    new IncompressibleLSVOFScalarConvectiveFlux<EvalType,
                                                                panzer::Traits,
                                                                num_space_dim>(
                        *ir,
                        n - 1,
                        num_lsvof_dofs,
                        phase_fraction_name + "_equation"));

                evaluators->push_back(eval_vof_conv);

                // Artificial Compression term
                const auto eval_vof_compress = Teuchos::rcp(
                    new IncompressibleLSVOFArtificialCompression<EvalType,
                                                                 panzer::Traits,
                                                                 num_space_dim>(
                        *ir,
                        lsvof_params,
                        phase_fraction_name,
                        phase_fraction_name + "_equation"));

                evaluators->push_back(eval_vof_compress);

                // Entropy Function evaluation
                const auto eval_entropy_function = Teuchos::rcp(
                    new IncompressibleLSVOFEntropyFunction<EvalType,
                                                           panzer::Traits,
                                                           num_space_dim>(
                        *ir,
                        lsvof_params,
                        phase_fraction_name,
                        phase_fraction_name + "_equation"));

                evaluators->push_back(eval_entropy_function);

                // Entropy Fluctuation term
                const auto eval_entropy_fluc = Teuchos::rcp(
                    new IncompressibleLSVOFEntropyFluctuation<EvalType,
                                                              panzer::Traits,
                                                              num_space_dim>(
                        *ir,
                        lsvof_params,
                        phase_fraction_name + "_equation",
                        global_data));

                evaluators->push_back(eval_entropy_fluc);

                // Entropy Viscosity term
                const auto eval_entropy_visc = Teuchos::rcp(
                    new IncompressibleLSVOFEntropyViscosity<EvalType,
                                                            panzer::Traits,
                                                            num_space_dim>(
                        *ir,
                        lsvof_params,
                        phase_fraction_name + "_equation",
                        global_data));

                evaluators->push_back(eval_entropy_visc);

                // Viscous flux
                const auto eval_vof_visc = Teuchos::rcp(
                    new IncompressibleLSVOFScalarViscousFlux<EvalType,
                                                             panzer::Traits,
                                                             num_space_dim>(
                        *ir,
                        phase_fraction_name,
                        phase_fraction_name + "_equation"));

                evaluators->push_back(eval_vof_visc);

                // TODO: add compressive flux, source terms, etc. as they
                // are created.
            }
            else
            {
                // Create combined array of phase fraction fields
                const bool include_time_deriv = true;
                const bool include_grads = false;
                const auto phase_vec
                    = Utils::ScalarToVector<EvalType, PhaseIndex>::createFromList(
                        *ir,
                        "volume_fractions",
                        phase_names,
                        include_time_deriv,
                        include_grads);

                evaluators->push_back(phase_vec);

                phase_names.push_back(phase_fraction_name);

                // Closure that calculate phase fraction for Nth phase:
                // $\alpha_n = 1.0 - \alpha_1 - ... - \alpha_{n-1}$
                const auto eval_nth_phase = Teuchos::rcp(
                    new IncompressibleLSVOFNthPhaseFraction<EvalType,
                                                            panzer::Traits>(
                        *ir, phase_names));

                evaluators->push_back(eval_nth_phase);
            }
        }
    }
    else if (lsvof_model_name == LSVOFModelType::CLS)
    {
        // Construct list of phase names
        for (int n = 1; n <= num_phases; ++n)
        {
            Teuchos::ParameterList phase_list
                = lsvof_params.sublist("Phase " + std::to_string(n));

            const std::string phase_name
                = phase_list.get<std::string>("Phase Name");

            phase_names.push_back(phase_name);
        }

        // Check for interface normal reconstruction
        const auto type_validator = Teuchos::rcp(
            new Teuchos::StringToIntegralParameterEntryValidator<
                LSNormalReconstructionType>(Teuchos::tuple<std::string>("NonRe"
                                                                        "const"
                                                                        "ructe"
                                                                        "d"),
                                            "NonReconstructed"));

        const LSNormalReconstructionType reconstruction_type
            = type_validator->getIntegralValue(lsvof_params.get<std::string>(
                "LS Normal Reconstruction Type"));

        // Automatically build metric tensor element length
        const auto eval_mt
            = Teuchos::rcp(new MetricTensor<EvalType, panzer::Traits>(*ir));

        evaluators->push_back(eval_mt);

        const auto eval_mt_el = Teuchos::rcp(
            new MetricTensorElementLength<EvalType, panzer::Traits>(*ir));

        evaluators->push_back(eval_mt_el);

        // Interface thickness, epsilon
        const auto eval_cls_epsilon = Teuchos::rcp(
            new IncompressibleCLSEpsilon<EvalType, panzer::Traits, num_space_dim>(
                *ir, lsvof_params));

        evaluators->push_back(eval_cls_epsilon);

        // Save phi field values from previous stage for time derivative and
        // convective flux (when Crank-Nicolson option is used)
        const std::string old_value_type
            = lsvof_params.get<std::string>("Old Value Type", "LastStage");

        Teuchos::ParameterList old_value_params;
        old_value_params.set("Field Name", "level_set");
        old_value_params.set("Old Value Type", old_value_type);

        const auto eval_phi_star
            = Teuchos::rcp(new VariableOldValue<EvalType, panzer::Traits>(
                *ir, old_value_params));

        evaluators->push_back(eval_phi_star);

        // Smoothed sign function
        const auto eval_cls_sign = Teuchos::rcp(
            new IncompressibleCLSSign<EvalType, panzer::Traits>(*ir));

        evaluators->push_back(eval_cls_sign);

        // Time derivative
        const auto eval_cls_dqdt = Teuchos::rcp(
            new IncompressibleCLSTimeDerivative<EvalType, panzer::Traits>(*ir));

        evaluators->push_back(eval_cls_dqdt);

        // Convective flux
        Teuchos::ParameterList conv_flux_params;
        conv_flux_params.set("Field Name", "CLS_sign");
        conv_flux_params.set("Equation Name", "level_set_equation");

        const auto eval_cls_conv = Teuchos::rcp(
            new VariableConvectiveFlux<EvalType, panzer::Traits, num_space_dim>(
                *ir, conv_flux_params));

        evaluators->push_back(eval_cls_conv);

        // Regularization parameter, lambda
        const auto eval_cls_lambda = Teuchos::rcp(
            new IncompressibleCLSLambda<EvalType, panzer::Traits, num_space_dim>(
                *ir, lsvof_params, global_data));

        evaluators->push_back(eval_cls_lambda);

        // Interface normal
        if (reconstruction_type == LSNormalReconstructionType::NonReconstructed)
        {
            const auto eval_cls_q = Teuchos::rcp(
                new IncompressibleCLSNonReconstructedNormal<EvalType,
                                                            panzer::Traits,
                                                            num_space_dim>(*ir));

            evaluators->push_back(eval_cls_q);
        }

        // Regularization term
        const auto eval_cls_reg = Teuchos::rcp(
            new IncompressibleCLSRegularization<EvalType,
                                                panzer::Traits,
                                                num_space_dim>(*ir));

        evaluators->push_back(eval_cls_reg);
    }

    // Add closure models for Navier-Stokes equations
    if (build_lsvofmom_equ)
    {
        const auto eval_lsvof_props = Teuchos::rcp(
            new IncompressibleLSVOFVariableProperties<EvalType, panzer::Traits>(
                *ir, lsvof_params, phase_names));

        evaluators->push_back(eval_lsvof_props);

        const auto eval_ns_dqdt = Teuchos::rcp(
            new IncompressibleLSVOFTimeDerivative<EvalType,
                                                  panzer::Traits,
                                                  num_space_dim>(
                *ir, lsvof_params));

        evaluators->push_back(eval_ns_dqdt);

        const auto eval_ns_conv = Teuchos::rcp(
            new IncompressibleLSVOFConvectiveFlux<EvalType,
                                                  panzer::Traits,
                                                  num_space_dim>(*ir));

        evaluators->push_back(eval_ns_conv);

        const auto eval_ns_visc
            = Teuchos::rcp(new IncompressibleLSVOFViscousFlux<EvalType,
                                                              panzer::Traits,
                                                              num_space_dim>(
                *ir, lsvof_params));

        evaluators->push_back(eval_ns_visc);

        // Buoyancy
        if (_build_lsvof_buoyancy_source)
        {
            const auto eval_ns_buo = Teuchos::rcp(
                new IncompressibleLSVOFBuoyancySource<EvalType,
                                                      panzer::Traits,
                                                      num_space_dim>(
                    *ir, user_params));

            evaluators->push_back(eval_ns_buo);
        }

        // Surface Tension
        if (_build_lsvof_surface_tension)
        {
            const auto eval_ns_surf = Teuchos::rcp(
                new IncompressibleLSVOFSurfaceTensionForce<EvalType,
                                                           panzer::Traits,
                                                           num_space_dim>(
                    *ir, lsvof_params, phase_names));

            evaluators->push_back(eval_ns_surf);
        }
    }

    // Closures required for global evaluators
    const auto eval_local_dt = Teuchos::rcp(
        new IncompressibleLocalTimeStepSize<EvalType, panzer::Traits, num_space_dim>(
            *ir));

    evaluators->push_back(eval_local_dt);
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
bool IncompressibleLSVOFFactory<EvalType, NumSpaceDim>::isDefaultClosureModel(
    const std::string& closure_type)
{
    std::vector<std::string> default_closures;

    default_closures.push_back("IncompressibleCLSEpsilon");
    default_closures.push_back("IncompressibleCLSLambda");
    default_closures.push_back("IncompressibleCLSNonReconstructedNormal");
    default_closures.push_back("IncompressibleCLSRegularization");
    default_closures.push_back("IncompressibleCLSSign");
    default_closures.push_back("IncompressibleCLSTimeDerivative");
    default_closures.push_back("IncompressibleLocalTimeStepSize");
    default_closures.push_back("IncompressibleLSVOFArtificialCompression");
    default_closures.push_back("IncompressibleLSVOFBuoyancySource");
    default_closures.push_back("IncompressibleLSVOFConvectiveFlux");
    default_closures.push_back("IncompressibleLSVOFEntropyFluctuation");
    default_closures.push_back("IncompressibleLSVOFEntropyFunction");
    default_closures.push_back("IncompressibleLSVOFEntropyViscosity");
    default_closures.push_back("IncompressibleLSVOFProperties");
    default_closures.push_back("IncompressibleLSVOFScalarConvectiveFlux");
    default_closures.push_back("IncompressibleLSVOFScalarViscousFlux");
    default_closures.push_back("IncompressibleLSVOFSurfaceTensionForce");
    default_closures.push_back("IncompressibleLSVOFTimeDerivative");
    default_closures.push_back("IncompressibleLSVOFVariableProperties");
    default_closures.push_back("IncompressibleLSVOFViscousFlux");

    for (const auto& closure : default_closures)
    {
        if (closure == closure_type)
            return true;
    }

    return false;
}
//---------------------------------------------------------------------------//
} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_INCOMPRESSIBLELSVOFCLOSUREMODELFACTORY_IMPL_HPP
