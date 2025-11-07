#ifndef VERTEXCFD_CONDUCTIONCLOSUREMODELFACTORY_IMPL_HPP
#define VERTEXCFD_CONDUCTIONCLOSUREMODELFACTORY_IMPL_HPP

#include "conduction/closure_models/VertexCFD_ConductionClosureModelFactory.hpp"

#include "closure_models/VertexCFD_Closure_ConstantScalarField.hpp"
#include "closure_models/VertexCFD_Closure_ElementLength.hpp"

#include "conduction/closure_models/VertexCFD_Closure_ConductionErrorNorms.hpp"
#include "conduction/closure_models/VertexCFD_Closure_ConductionExactSolution.hpp"
#include "conduction/closure_models/VertexCFD_Closure_ConductionFlux.hpp"
#include "conduction/closure_models/VertexCFD_Closure_ConductionThermalConductivity.hpp"
#include "conduction/closure_models/VertexCFD_Closure_ConductionTimeDerivative.hpp"
#include "conduction/closure_models/VertexCFD_Closure_ConductionTimeStepSize.hpp"
#include "conduction/closure_models/VertexCFD_Closure_ConductionVolumetricSource.hpp"

namespace VertexCFD
{
namespace ClosureModel
{

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>>
ConductionFactory<EvalType, NumSpaceDim>::buildClosureModels(
    const std::string& model_id,
    const Teuchos::ParameterList& model_params,
    const panzer::FieldLayoutLibrary&,
    const Teuchos::RCP<panzer::IntegrationRule>& ir,
    const Teuchos::ParameterList& /**default_params**/,
    const Teuchos::ParameterList& /**user_params**/,
    const Teuchos::RCP<panzer::GlobalData>& /**global_data**/,
    PHX::FieldManager<panzer::Traits>&) const
{
    constexpr int num_space_dim = NumSpaceDim;

    auto evaluators = Teuchos::rcp(
        new std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>);

    if (!model_params.isSublist(model_id))
    {
        throw std::runtime_error("Conduction closure model id " + model_id
                                 + "not in list.");
    }

    // Get closure model list for 'model_id'
    const Teuchos::ParameterList& closure_model_list
        = model_params.sublist(model_id);

    // Check for closure model factory type. If not `Conduction` return empty
    // evaluator
    const std::string factory_type
        = closure_model_list.isType<std::string>("Closure Factory Type")
              ? closure_model_list.get<std::string>("Closure Factory Type")
              : "Navier Stokes";

    if (factory_type != "Conduction")
        return evaluators;

    // Adding default closure models to solve conduction equation
    const std::string default_closure_models
        = "ConductionTimeDerivative,\nConductionFlux,\n"
          "ConductionTimeStepSize,\n"
          "ElementLength.\n";

    // Conduction time derivative
    {
        auto eval_conduction = Teuchos::rcp(
            new ConductionTimeDerivative<EvalType, panzer::Traits>(*ir));
        evaluators->push_back(eval_conduction);
    }

    // Conduction flux
    {
        auto eval_conduction
            = Teuchos::rcp(new ConductionFlux<EvalType, panzer::Traits>(*ir));
        evaluators->push_back(eval_conduction);
    }

    // Time step size
    {
        auto eval_conduction = Teuchos::rcp(
            new ConductionTimeStepSize<EvalType, panzer::Traits>(*ir));
        evaluators->push_back(eval_conduction);
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

        // Get closure parameters for current closure model
        const auto& closure_params
            = Teuchos::getValue<Teuchos::ParameterList>(closure_model.second);

        if (closure_params.isType<std::string>("Type"))
        {
            const auto closure_type = closure_params.get<std::string>("Type");

            // Check if 'closure_type' is a default closure model
            if (default_closure_models.find(closure_type) != std::string::npos)
            {
                std::string msg = "\n\nConduction closure model/type '" + closure_name + "'/'" + closure_type
                                  + "' found in model id '" + model_id + "' is a default closure model and should not be added to the input file.\n";
                msg += "The list of default closure models for the conduction equation is:\n";
                msg += default_closure_models;
                throw std::runtime_error(msg);
            }

            // Error norm
            if (closure_type == "ConductionErrorNorms")
            {
                auto eval = Teuchos::rcp(
                    new ConductionErrorNorms<EvalType, panzer::Traits, num_space_dim>(
                        *ir));
                evaluators->push_back(eval);
                found_model = true;
            }

            // Exact solution
            if (closure_type == "ConductionExactSolution")
            {
                auto eval
                    = Teuchos::rcp(new ConductionExactSolution<EvalType,
                                                               panzer::Traits,
                                                               num_space_dim>(
                        *ir, closure_params));
                evaluators->push_back(eval);
                found_model = true;
            }

            // Volumetric source
            if (closure_type == "ConductionVolumetricSource")
            {
                auto eval = Teuchos::rcp(
                    new ConductionVolumetricSource<EvalType, panzer::Traits>(
                        *ir, closure_params));
                evaluators->push_back(eval);
                found_model = true;
            }

            // Constant material properties
            if (closure_type == "ConstantMaterialProperties")
            {
                // Density
                auto eval_rho = Teuchos::rcp(
                    new ConstantScalarField<EvalType, panzer::Traits>(
                        *ir,
                        "solid_density",
                        closure_params.get<double>("Density Value")));
                evaluators->push_back(eval_rho);

                // Cp
                auto eval_cp = Teuchos::rcp(
                    new ConstantScalarField<EvalType, panzer::Traits>(
                        *ir,
                        "solid_specific_heat_capacity",
                        closure_params.get<double>("Specific Heat Capacity "
                                                   "Value")));
                evaluators->push_back(eval_cp);

                // Thermal conductivity
                auto eval_k = Teuchos::rcp(
                    new ConductionThermalConductivity<EvalType, panzer::Traits>(
                        *ir, closure_params));
                evaluators->push_back(eval_k);
                found_model = true;
            }

            // Error message if closure model not found
            if (!found_model)
            {
                std::string msg = "\n\nConduction closure model/type "
                                  + closure_name + "/" + closure_type
                                  + " failed to build in model id " + model_id
                                  + ".\n";
                msg += "The conduction closure models available in Vertex-CFD are:\n";
                msg += "ConductionErrorNorms,\n";
                msg += "ConductionExactSolution,\n";
                msg += "ConductionVolumetricSource,\n";
                msg += "ConstantMaterialProperties,\n";
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

#endif // end VERTEXCFD_CONDUCTIONCLOSUREMODELFACTORY_IMPL_HPP
