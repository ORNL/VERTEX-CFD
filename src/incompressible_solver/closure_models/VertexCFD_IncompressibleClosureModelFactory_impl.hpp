#ifndef VERTEXCFD_INCOMPRESSIBLECLOSUREMODELFACTORY_IMPL_HPP
#define VERTEXCFD_INCOMPRESSIBLECLOSUREMODELFACTORY_IMPL_HPP

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleBuoyancySource.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleConstantSource.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleConvectiveFlux.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleConvectiveFluxMachineLearning.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleEnstrophy.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleErrorNorms.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleGradDiv.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleLiftDrag.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleLocalTimeStepSize.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleNusseltNumber.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressiblePlanarPoiseuilleExact.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleRotatingAnnulusExact.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleSUPGExactSolution.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleSUPGFlux.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleShearVariables.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleTauSUPG.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleTaylorGreenVortexExactSolution.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleTimeDerivative.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleViscousFlux.hpp"
#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleViscousHeat.hpp"
#include "incompressible_solver/closure_models/VertexCFD_IncompressibleClosureModelFactory.hpp"
#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

#include "closure_models/VertexCFD_Closure_MethodManufacturedSolution.hpp"
#include "closure_models/VertexCFD_Closure_MethodManufacturedSolutionSource.hpp"

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void IncompressibleFactory<EvalType, NumSpaceDim>::buildClosureModel(
    const std::string& closure_type,
    const Teuchos::RCP<panzer::IntegrationRule>& ir,
    const Teuchos::RCP<panzer::GlobalData>& global_data,
    const Teuchos::ParameterList& user_params,
    const Teuchos::ParameterList& closure_params,
    const bool use_turbulence_model,
    bool& found_model,
    std::string& error_msg,
    Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>>
        evaluators)
{
    // Define local variables
    constexpr int num_space_dim = NumSpaceDim;
    const Teuchos::RCP<PHX::Evaluator<panzer::Traits>> eval;

    // Fluid properties
    const Teuchos::ParameterList fluid_prop_list
        = user_params.sublist("Fluid Properties List");
    const FluidProperties::ConstantFluidProperties incompressible_fluidprop_params
        = FluidProperties::ConstantFluidProperties(fluid_prop_list);

    // Stability parameter list
    const Teuchos::ParameterList stability_param_list
        = user_params.sublist("Stability Parameters");

    // Closure models
    if (closure_type == "IncompressibleTimeDerivative")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleTimeDerivative<EvalType, panzer::Traits, num_space_dim>(
                *ir, incompressible_fluidprop_params));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressibleLiftDrag")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleLiftDrag<EvalType, panzer::Traits, num_space_dim>(
                *ir, user_params, use_turbulence_model));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressibleConvectiveFlux")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleConvectiveFlux<EvalType, panzer::Traits, num_space_dim>(
                *ir, incompressible_fluidprop_params, user_params));
        evaluators->push_back(eval);
        found_model = true;
    }

#ifdef VERTEXCFD_HAVE_TENSORFLOW
    if (closure_type == "IncompressibleConvectiveFluxMachineLearning")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleConvectiveFluxMachineLearning<EvalType,
                                                            panzer::Traits,
                                                            num_space_dim>(
                *ir, incompressible_fluidprop_params, closure_params));
        evaluators->push_back(eval);
        found_model = true;
    }
#else
    (void)closure_params;
#endif

    if (closure_type == "IncompressibleViscousFlux")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleViscousFlux<EvalType, panzer::Traits, num_space_dim>(
                *ir,
                incompressible_fluidprop_params,
                user_params,
                use_turbulence_model));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressibleGradDiv")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleGradDiv<EvalType, panzer::Traits, num_space_dim>(
                *ir, stability_param_list));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressibleLocalTimeStepSize")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleLocalTimeStepSize<EvalType,
                                                panzer::Traits,
                                                num_space_dim>(*ir));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressibleConstantSource")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleConstantSource<EvalType, panzer::Traits, num_space_dim>(
                *ir,
                incompressible_fluidprop_params,
                global_data,
                closure_params));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressibleBuoyancySource")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleBuoyancySource<EvalType, panzer::Traits, num_space_dim>(
                *ir, incompressible_fluidprop_params, user_params));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressibleViscousHeat")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleViscousHeat<EvalType, panzer::Traits, num_space_dim>(
                *ir));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressibleRotatingAnnulusExact")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleRotatingAnnulusExact<EvalType,
                                                   panzer::Traits,
                                                   num_space_dim>(
                *ir, closure_params));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressiblePlanarPoiseuilleExact")
    {
        auto eval = Teuchos::rcp(
            new IncompressiblePlanarPoiseuilleExact<EvalType,
                                                    panzer::Traits,
                                                    num_space_dim>(
                *ir, closure_params));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressibleErrorNorm")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleErrorNorms<EvalType, panzer::Traits, num_space_dim>(
                *ir, user_params));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressibleTaylorGreenVortexExactSolution")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleTaylorGreenVortexExactSolution<EvalType,
                                                             panzer::Traits,
                                                             num_space_dim>(
                *ir, closure_params));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressibleShearVariables")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleShearVariables<EvalType, panzer::Traits, num_space_dim>(
                *ir));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressibleNusseltNumber")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleNusseltNumber<EvalType, panzer::Traits>(*ir));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressibleEnstrophy")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleEnstrophy<EvalType, panzer::Traits, num_space_dim>(
                *ir));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressibleTauSUPG")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleTauSUPG<EvalType, panzer::Traits, num_space_dim>(
                *ir, incompressible_fluidprop_params, closure_params));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressibleSUPGFlux")
    {
        auto eval = Teuchos::rcp(
            new IncompressibleSUPGFlux<EvalType, panzer::Traits, num_space_dim>(
                *ir, incompressible_fluidprop_params, closure_params));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "IncompressibleSUPGExactSolution")
    {
        auto eval
            = Teuchos::rcp(new IncompressibleSUPGExactSolution<EvalType,
                                                               panzer::Traits,
                                                               num_space_dim>(
                *ir, closure_params));
        evaluators->push_back(eval);
        found_model = true;
    }

    // NOTE: the method of manufactured solution (MMS) is kept here
    // as a template for future use. This is currently not used in any input
    // files.
    if (closure_type == "MethodManufacturedSolution")
    {
        auto eval = Teuchos::rcp(
            new MethodManufacturedSolution<EvalType, panzer::Traits, num_space_dim>(
                *ir));
        evaluators->push_back(eval);
        found_model = true;
    }

    if (closure_type == "MethodManufacturedSolutionSource")
    {
        bool build_viscous_flux = false;
        if (user_params.isType<bool>("Build Viscous Flux"))
        {
            build_viscous_flux = user_params.get<bool>("Build Viscous Flux");
        }
        auto eval
            = Teuchos::rcp(new MethodManufacturedSolutionSource<EvalType,
                                                                panzer::Traits,
                                                                num_space_dim>(
                *ir, build_viscous_flux, closure_params));
        evaluators->push_back(eval);
        found_model = true;
    }

    // Initialize 'error_msg' with list of closure models for incompressible
    // NS equations
    error_msg += "IncompressibleBuoyancySource\n";
    error_msg += "IncompressibleConstantSource\n";
    error_msg += "IncompressibleConvectiveFlux\n";
    error_msg += "IncompressibleConvectiveFluxMachineLearning\n";
    error_msg += "IncompressibleEnstrophy\n";
    error_msg += "IncompressibleErrorNorm\n";
    error_msg += "IncompressibleGradDiv\n";
    error_msg += "IncompressibleLiftDrag\n";
    error_msg += "IncompressibleLocalTimeStepSize\n";
    error_msg += "IncompressibleNusseltNumber\n";
    error_msg += "IncompressiblePlanarPoiseuilleExact\n";
    error_msg += "IncompressibleRotatingAnnulusExact\n";
    error_msg += "IncompressibleShearVariables\n";
    error_msg += "IncompressibleSUPGExactSolution\n";
    error_msg += "IncompressibleSUPGFlux\n";
    error_msg += "IncompressibleTauSUPG\n";
    error_msg += "IncompressibleTimeDerivative\n";
    error_msg += "IncompressibleViscousFlux\n";
    error_msg += "IncompressibleViscousHeat\n";
    error_msg += "MethodManufacturedSolution\n";
    error_msg += "MethodManufacturedSolutionSource\n";
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_INCOMPRESSIBLECLOSUREMODELFACTORY_IMPL_HPP
