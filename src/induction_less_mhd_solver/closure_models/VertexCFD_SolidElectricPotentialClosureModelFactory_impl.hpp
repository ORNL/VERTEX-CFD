#ifndef VERTEXCFD_SOLIDELECTRICPOTENTIALCLOSUREMODELFACTORY_IMPL_HPP
#define VERTEXCFD_SOLIDELECTRICPOTENTIALCLOSUREMODELFACTORY_IMPL_HPP

#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_SolidElectricConductivity.hpp"
#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_SolidElectricPotentialDiffusionFlux.hpp"
#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_SolidElectricPotentialErrorNorms.hpp"
#include "induction_less_mhd_solver/closure_models/VertexCFD_Closure_SolidElectricPotentialExactSolution.hpp"
#include "induction_less_mhd_solver/closure_models/VertexCFD_SolidElectricPotentialClosureModelFactory.hpp"

#include "closure_models/VertexCFD_Closure_ConstantScalarField.hpp"

#include <vector>

namespace VertexCFD
{
namespace ClosureModel
{

//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>>
SolidElectricPotentialFactory<EvalType, NumSpaceDim>::buildClosureModels(
    const std::string& model_id,
    const Teuchos::ParameterList& model_params,
    const panzer::FieldLayoutLibrary&,
    const Teuchos::RCP<panzer::IntegrationRule>& ir,
    const Teuchos::ParameterList& /**default_params**/,
    const Teuchos::ParameterList& /**user_params**/,
    const Teuchos::RCP<panzer::GlobalData>& /**global_data**/,
    PHX::FieldManager<panzer::Traits>&) const
{
    auto evaluators = Teuchos::rcp(
        new std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>);

    if (!model_params.isSublist(model_id))
    {
        throw std::runtime_error("SolidElectricPotential closure model id "
                                 + model_id + "not in list.");
    }

    constexpr int num_space_dim = NumSpaceDim;

    // Get closure model list for 'model_id'
    const Teuchos::ParameterList& closure_model_list
        = model_params.sublist(model_id);

    // Check for closure model factory type. If not `SolidElectricPotential`
    // return empty evaluator
    const std::string factory_type
        = closure_model_list.isType<std::string>("Closure Factory Type")
              ? closure_model_list.get<std::string>("Closure Factory Type")
              : "None";

    if (factory_type != "SolidElectricPotential")
        return evaluators;

    // Adding default closure models to solve MHD equation
    const std::string default_closure_models
        = "SolidElectricPotentialDiffusionFlux,\n";

    // SolidElectricPotential flux
    {
        auto eval_conduction = Teuchos::rcp(
            new SolidElectricPotentialDiffusionFlux<EvalType, panzer::Traits>(
                *ir));
        evaluators->push_back(eval_conduction);
    }

    // Check for required sublist
    std::vector<std::string> needed_cm = {"Material Properties"};
    for (std::string cm : needed_cm)
    {
        if (!closure_model_list.isSublist(cm))
        {
            throw std::runtime_error(
                "\n\nERROR: the required sublist '" + cm
                + "' could not be found in the closure model list '" + model_id
                + "'\n\n");
        }
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
                std::string msg = "\n\nSolidElectricPotential closure model/type '" + closure_name + "'/'" + closure_type
                                  + "' found in model id '" + model_id + "' is a default closure model and should not be added to the input file.\n";
                msg += "The list of default closure models for the conduction equation is:\n";
                msg += default_closure_models;
                throw std::runtime_error(msg);
            }

            // Electric conductivity
            if (closure_type == "SolidElectricConductivity")
            {
                auto eval = Teuchos::rcp(
                    new SolidElectricConductivity<EvalType, panzer::Traits>(
                        *ir, closure_params));
                evaluators->push_back(eval);
                found_model = true;
            }

            // Error norms
            if (closure_type == "SolidElectricPotentialErrorNorms")
            {
                auto eval = Teuchos::rcp(
                    new SolidElectricPotentialErrorNorms<EvalType,
                                                         panzer::Traits,
                                                         num_space_dim>(*ir));
                evaluators->push_back(eval);
                found_model = true;
            }

            // Exact solution
            if (closure_type == "SolidElectricPotentialExactSolution")
            {
                auto eval = Teuchos::rcp(
                    new SolidElectricPotentialExactSolution<EvalType,
                                                            panzer::Traits,
                                                            num_space_dim>(
                        *ir, closure_params));
                evaluators->push_back(eval);
                found_model = true;
            }

            // Time step size: this is only needed when solving for the
            // induction-less equation in a solid region without any other
            // physics.
            if (closure_type == "SolidElectricTimeStepSize")
            {
                auto eval = Teuchos::rcp(
                    new ConstantScalarField<EvalType, panzer::Traits>(
                        *ir, "local_dt", 10e10));
                evaluators->push_back(eval);
                found_model = true;
            }

            // Error message if closure model not found
            if (!found_model)
            {
                std::string msg
                    = "\n\nSolidElectricPotential closure model/type "
                      + closure_name + "/" + closure_type
                      + " failed to build in model id " + model_id + ".\n";
                msg += "The conduction closure models available in Vertex-CFD are:\n";
                msg += "SolidElectricConductivity,\n";
                msg += "SolidElectricPotentialErrorNorms,\n";
                msg += "SolidElectricPotentialExactSolution,\n";
                msg += "SolidElectricTimeStepSize,\n";
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

#endif // end VERTEXCFD_SOLIDELECTRICPOTENTIALCLOSUREMODELFACTORY_IMPL_HPP
