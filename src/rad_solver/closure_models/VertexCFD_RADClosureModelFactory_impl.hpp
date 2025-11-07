#ifndef VERTEXCFD_RADCLOSUREMODELFACTORY_IMPL_HPP
#define VERTEXCFD_RADCLOSUREMODELFACTORY_IMPL_HPP

#include "closure_models/VertexCFD_Closure_ConstantVectorField.hpp"
#include "closure_models/VertexCFD_Closure_ElementLength.hpp"
#include "closure_models/VertexCFD_Closure_ExternalInterpolation.hpp"

#include "rad_solver/closure_models/VertexCFD_Closure_RADAdvectionFlux.hpp"
#include "rad_solver/closure_models/VertexCFD_Closure_RADDiffusionFlux.hpp"
#include "rad_solver/closure_models/VertexCFD_Closure_RADErrorNorms.hpp"
#include "rad_solver/closure_models/VertexCFD_Closure_RADExactSolution.hpp"
#include "rad_solver/closure_models/VertexCFD_Closure_RADFissionSource.hpp"
#include "rad_solver/closure_models/VertexCFD_Closure_RADFissionSourceExactSolution.hpp"
#include "rad_solver/closure_models/VertexCFD_Closure_RADLocalTimeStepSize.hpp"
#include "rad_solver/closure_models/VertexCFD_Closure_RADReaction.hpp"
#include "rad_solver/closure_models/VertexCFD_Closure_RADTimeDerivative.hpp"

#include "rad_solver/closure_models/VertexCFD_RADClosureModelFactory.hpp"

#include "utils/VertexCFD_Utils_Constants.hpp"

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>>
RADFactory<EvalType, NumSpaceDim>::buildClosureModels(
    const std::string& model_id,
    const Teuchos::ParameterList& model_params,
    const panzer::FieldLayoutLibrary&,
    const Teuchos::RCP<panzer::IntegrationRule>& ir,
    const Teuchos::ParameterList& /**default_params**/,
    const Teuchos::ParameterList& user_params,
    const Teuchos::RCP<panzer::GlobalData>& /**global_data**/,
    PHX::FieldManager<panzer::Traits>&) const
{
    constexpr int num_space_dim = NumSpaceDim;
    auto evaluators = Teuchos::rcp(
        new std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>);

    if (!model_params.isSublist(model_id))
    {
        throw std::runtime_error("RAD closure model id " + model_id
                                 + "not in list.");
    }

    // Get closure model list for 'model_id'
    const Teuchos::ParameterList& closure_model_list
        = model_params.sublist(model_id);

    // Check for closure model factory type. If not `RAD` return empty
    // evaluator
    const std::string factory_type
        = closure_model_list.isType<std::string>("Closure Factory Type")
              ? closure_model_list.get<std::string>("Closure Factory Type")
              : "None";

    if (factory_type != "RAD")
        return evaluators;

    if (!closure_model_list.isSublist("Model Parameters"))
    {
        throw std::runtime_error(
            "\n\nERROR: define model parameters in 'Model Parameters' sublist "
            "in 'Closure Models' list.\n\n");
    }

    // Get Model and Reaction Parameters
    auto rad_params = closure_model_list.sublist("Model Parameters");
    Teuchos::ParameterList reaction_params;
    if (rad_params.isSublist("Reaction Parameters"))
    {
        reaction_params = rad_params.sublist("Reaction Parameters");
        // We need to delete reaction parameters for the parameter validation
        rad_params.remove("Reaction Parameters");
    }

    // Define valid parameters and set them
    Teuchos::ParameterList valid_parameters;
    valid_parameters.set("Number of Species", 1, "Number of species");
    valid_parameters.set("Build Advection", false, "Advection term boolean");
    valid_parameters.set("Build Diffusion", false, "Diffusion term boolean");
    valid_parameters.set("Build Reaction", false, "Reaction term boolean");
    valid_parameters.set(
        "Build Fission Source", false, "Fission source boolean");
    valid_parameters.set("Diffusion Coefficient",
                         std::numeric_limits<double>::quiet_NaN(),
                         "Constant diffusion coefficient");
    valid_parameters.set("Neutron Flux Variable Name",
                         "neutron_flux",
                         "External neutron flux name string");

    rad_params.validateParametersAndSetDefaults(valid_parameters);

    const bool build_advection = rad_params.get<bool>("Build Advection");
    const bool build_diffusion = rad_params.get<bool>("Build Diffusion");
    const bool build_reaction = rad_params.get<bool>("Build Reaction");
    const bool build_fission_source
        = rad_params.get<bool>("Build Fission Source");

    // Species properties
    const SpeciesProperties::ConstantSpeciesProperties rad_species_prop_params
        = SpeciesProperties::ConstantSpeciesProperties(rad_params,
                                                       reaction_params);

    // Adding default closure models to solve rad equations
    const std::string default_closure_models
        = "RADAdvectionFlux,\nRADDiffusionFlux,\nRADReaction,"
          "\nRADErrorNorms,\nRADFissionSource,\nRADTimeDerivative,"
          "\nElementLength.\n";

    // Reaction term
    if (build_reaction)
    {
        auto eval_rad = Teuchos::rcp(new RADReaction<EvalType, panzer::Traits>(
            *ir, rad_species_prop_params));
        evaluators->push_back(eval_rad);
    }

    // Advection flux
    if (build_advection)
    {
        auto eval_rad = Teuchos::rcp(
            new RADAdvectionFlux<EvalType, panzer::Traits, num_space_dim>(
                *ir, rad_species_prop_params));
        evaluators->push_back(eval_rad);
    }

    // Advection flux
    if (build_diffusion)
    {
        auto eval_rad
            = Teuchos::rcp(new RADDiffusionFlux<EvalType, panzer::Traits>(
                *ir, rad_species_prop_params));
        evaluators->push_back(eval_rad);
    }

    // Fission source
    if (build_fission_source)
    {
        const std::string neutron_flux_name
            = rad_params.get<std::string>("Neutron Flux Variable Name");
        auto eval_rad
            = Teuchos::rcp(new RADFissionSource<EvalType, panzer::Traits>(
                *ir, rad_species_prop_params, neutron_flux_name));
        evaluators->push_back(eval_rad);
    }

    // RAD error norms
    if (user_params.isSublist("Compute Error Norms"))
    {
        auto eval_rad
            = Teuchos::rcp(new RADErrorNorms<EvalType, panzer::Traits>(
                *ir, rad_species_prop_params));
        evaluators->push_back(eval_rad);
    }

    // RAD time derivative
    {
        auto eval_rad
            = Teuchos::rcp(new RADTimeDerivative<EvalType, panzer::Traits>(
                *ir, rad_species_prop_params));
        evaluators->push_back(eval_rad);
    }

    // Element length
    {
        auto eval
            = Teuchos::rcp(new ElementLength<EvalType, panzer::Traits>(*ir));
        evaluators->push_back(eval);
    }

    // Loop over closure models
    for (const auto& closure_model : closure_model_list)
    {
        bool found_model = false;

        const auto closure_name = closure_model.first;
        if (closure_name == "Closure Factory Type")
            continue;
        if (closure_name == "Model Parameters")
            continue;

        // Get closure parameters for current closure model
        const auto& closure_params
            = Teuchos::getValue<Teuchos::ParameterList>(closure_model.second);

        if (closure_params.isType<std::string>("Type"))
        {
            const auto closure_type = closure_params.get<std::string>("Type");

            // Check if 'closure_type' is a default closure model
            if (default_closure_models.find(closure_type) != std::string::npos)
            {
                std::string msg = "\n\nRAD closure model/type '" + closure_name + "'/'" + closure_type
                                  + "' found in model id '" + model_id + "' is a default closure model and should not be added to the input file.\n";
                msg += "The list of default closure models for the rad equation is:\n";
                msg += default_closure_models;
                throw std::runtime_error(msg);
            }

            // Constant Vector Field
            if (closure_type == "ConstantVectorField")
            {
                auto eval = Teuchos::rcp(
                    new ConstantVectorField<EvalType, panzer::Traits, num_space_dim>(
                        *ir, closure_params));
                evaluators->push_back(eval);
                found_model = true;
            }

            // External Interpolation With ArborX
#ifdef VERTEXCFD_HAVE_ARBORX
            if (closure_type == "ExternalInterpolation")
            {
                auto eval = Teuchos::rcp(
                    new ExternalInterpolation<EvalType, panzer::Traits, num_space_dim>(
                        *ir, closure_params));
                evaluators->push_back(eval);
                found_model = true;
            }
#endif

            // RAD local time step size
            if (closure_type == "RADLocalTimeStepSize")
            {
                auto eval_rad = Teuchos::rcp(
                    new RADLocalTimeStepSize<EvalType, panzer::Traits, num_space_dim>(
                        *ir, rad_species_prop_params));
                evaluators->push_back(eval_rad);
                found_model = true;
            }

            // RAD exact solution
            if (closure_type == "RADExactSolution")
            {
                auto eval_rad = Teuchos::rcp(
                    new RADExactSolution<EvalType, panzer::Traits, num_space_dim>(
                        *ir, rad_species_prop_params, closure_params));
                evaluators->push_back(eval_rad);
                found_model = true;
            }

            // RAD fission source exact solution
            if (closure_type == "RADFissionSourceExactSolution")
            {
                auto eval_rad = Teuchos::rcp(
                    new RADFissionSourceExactSolution<EvalType, panzer::Traits>(
                        *ir, rad_species_prop_params, closure_params));
                evaluators->push_back(eval_rad);
                found_model = true;
            }

            // Error message if closure model not found
            if (!found_model)
            {
                std::string msg = "\n\nRAD closure model/type " + closure_name
                                  + "/" + closure_type
                                  + " failed to build in model id " + model_id
                                  + ".\n";
                msg += "The rad closure models available in Vertex-CFD are:\n";
                msg += "\nConstantVectorField,";
                msg += "\nExternalInterpolation,";
                msg += "\nRADExactSolution";
                msg += "\nRADFissionSourceExactSolution";
                msg += "\nRADLocalTimeStepSize,";
                msg += "\n";

                throw std::runtime_error(msg);
            }
        }
    }

    return evaluators;
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_RADCLOSUREMODELFACTORY_IMPL_HPP
