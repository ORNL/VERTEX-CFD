#ifndef VERTEXCFD_PLASMASPECIESCLOSUREMODELFACTORY_IMPL_HPP
#define VERTEXCFD_PLASMASPECIESCLOSUREMODELFACTORY_IMPL_HPP

#include "plasma_solver/closure_models/VertexCFD_Closure_PlasmaSpeciesConvectiveFlux.hpp"
#include "plasma_solver/closure_models/VertexCFD_Closure_PlasmaSpeciesEOS.hpp"
#include "plasma_solver/closure_models/VertexCFD_Closure_PlasmaSpeciesTimeDerivative.hpp"

#include <string>
#include <vector>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>>
PlasmaSpeciesFactory<EvalType, NumSpaceDim>::buildClosureModels(
    const std::string& model_id,
    const Teuchos::ParameterList& model_params,
    const panzer::FieldLayoutLibrary&,
    const Teuchos::RCP<panzer::IntegrationRule>& ir,
    const Teuchos::ParameterList& /**default_params**/,
    const Teuchos::ParameterList& /**user_params**/,
    const Teuchos::RCP<panzer::GlobalData>& /**global_data**/,
    PHX::FieldManager<panzer::Traits>&) const
{
    // Initialize `evaluators` to store all closure models
    auto evaluators = Teuchos::rcp(
        new std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>);

    // Get closure model list for 'model_id'
    const Teuchos::ParameterList& closure_model_list
        = model_params.sublist(model_id);

    // Check for closure model factory type and return empty evaluators if it
    // is not "Plasma Species MHD"
    const std::string factory_type
        = closure_model_list.isType<std::string>("Closure Factory Type")
              ? closure_model_list.get<std::string>("Closure Factory Type")
              : "None";

    if (factory_type != "Plasma Species MHD")
        return evaluators;

    // Check for species properties
    Teuchos::ParameterList species_prop;
    if (closure_model_list.isSublist("Species Properties"))
    {
        species_prop = closure_model_list.sublist("Species Properties");
    }
    else
    {
        const std::string msg
            = "\n\nThe sublist 'Species Properties' is not found in the "
              "closure "
              "model list with model id '"
              + model_id + "'.\n\n";
        throw std::runtime_error(msg);
    }

    // Add default closure models for each species
    for (Teuchos::ParameterList::ConstIterator sp_ptr = species_prop.begin();
         sp_ptr != species_prop.end();
         ++sp_ptr)
    {
        // Get species name
        const std::string species_name = sp_ptr->first;

        // Get species parameters
        // WARNING: `species_params` does not contain the correct residual name
        // to be used in 'PlasmaSpeciesConvectiveFlux'. It will have to be
        // fixed before running the first case.
        Teuchos::ParameterList& species_params
            = Teuchos::getValue<Teuchos::ParameterList>(sp_ptr->second);
        species_params.set("Species Name", species_name);
        species_params.set("Residual Name", species_name + "CONVECTIVE");

        // Time derivatives
        {
            const auto eval = Teuchos::rcp(
                new PlasmaSpeciesTimeDerivative<EvalType, panzer::Traits, NumSpaceDim>(
                    *ir, species_name));
            evaluators->push_back(eval);
        }

        // Convection terms
        {
            const auto eval = Teuchos::rcp(
                new PlasmaSpeciesConvectiveFlux<EvalType, panzer::Traits, NumSpaceDim>(
                    *ir, species_params));
            evaluators->push_back(eval);
        }

        // EOS
        {
            const auto eval
                = Teuchos::rcp(new PlasmaSpeciesEOS<EvalType, panzer::Traits>(
                    *ir, species_params));
            evaluators->push_back(eval);
        }
    }

    // Loop over closure models found in the input file if any
    for (const auto& closure_model : closure_model_list)
    {
        bool found_model = false;

        // Ignore 'Closure Factory Type' entry
        const auto closure_name = closure_model.first;
        if (closure_name == "Closure Factory Type")
            continue;

        // Get closure parameters and type for current closure model
        const auto& closure_params
            = Teuchos::getValue<Teuchos::ParameterList>(closure_model.second);

        const auto closure_type = closure_params.get<std::string>("Type");

        // Check if `closure_type` is a default closure model
        this->checkDefaultClosureModel(closure_type);

        // List of optional closure models
        if (closure_type == "Dummy")
        {
            found_model = true;
        }

        // Return error message if not found
        if (!found_model)
        {
            const std::string msg = "\n\nPlasma Species closure model/type "
                              + closure_name + "/" + closure_type
                              + " failed to build in model id " + model_id
                              + ".\n"
                              "The closure models available in Vertex-CFD are:\n"
                              + availableClosureModels();
            throw std::runtime_error(msg);
        }
    }

    return evaluators;
}

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void PlasmaSpeciesFactory<EvalType, NumSpaceDim>::checkDefaultClosureModel(
    const std::string& closure_type) const
{
    // List all default closure models
    std::vector<std::string> default_closures;
    default_closures.push_back("PlasmaSpeciesTimeDerivative");
    default_closures.push_back("PlasmaSpeciesConvectiveFlux");
    default_closures.push_back("PlasmaSpeciesEOS");

    // Check if `closure_type` is found in `default_closures` and return error
    // message when appropriate.
    const auto it = std::find(
        default_closures.begin(), default_closures.end(), closure_type);
    if (it != default_closures.end())
    {
        std::string msg = "\n\nThe closure model '" + closure_type
                          + "' was found in the input file." + "'"
                          + closure_type +
                          "' is a default closure model and should be not "
                          "listed in the input file. The list of default "
                          "closure models is provided below:\n";
        for (const std::string& str : default_closures)
            msg += str + "\n";
        msg += "\n";
        throw std::runtime_error(msg);
    }
}

//--------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_PLASMASPECIESCLOSUREMODELFACTORY_IMPL_HPP
