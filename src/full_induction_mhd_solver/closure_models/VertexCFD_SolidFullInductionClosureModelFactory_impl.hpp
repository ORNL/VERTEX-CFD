#ifndef VERTEXCFD_SOLIDFULLINDUCTIONCLOSUREMODELFACTORY_IMPL_HPP
#define VERTEXCFD_SOLIDFULLINDUCTIONCLOSUREMODELFACTORY_IMPL_HPP

#include "closure_models/VertexCFD_Closure_ConstantScalarField.hpp"
#include "closure_models/VertexCFD_Closure_ElementLength.hpp"
#include "closure_models/VertexCFD_Closure_ExternalMagneticField.hpp"
#include "closure_models/VertexCFD_Closure_VectorFieldDivergence.hpp"

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleVariableTimeDerivative.hpp"

#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_InductionConstantSource.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_InductionResistiveFlux.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_MagneticCorrectionDampingSource.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_SolidFullInductionConvectiveFlux.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_SolidFullInductionLocalTimeStepSize.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_TotalMagneticField.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_TotalMagneticFieldGradient.hpp"

#include "full_induction_mhd_solver/mhd_properties/VertexCFD_FullInductionMHDProperties.hpp"

#include <Panzer_String_Utilities.hpp>

#include <string>
#include <vector>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>>
SolidFullInductionFactory<EvalType, NumSpaceDim>::buildClosureModels(
    const std::string& model_id,
    const Teuchos::ParameterList& model_params,
    const panzer::FieldLayoutLibrary&,
    const Teuchos::RCP<panzer::IntegrationRule>& ir,
    const Teuchos::ParameterList& /**default_params**/,
    const Teuchos::ParameterList& user_params,
    const Teuchos::RCP<panzer::GlobalData>& /**global_data**/,
    PHX::FieldManager<panzer::Traits>&) const
{
    auto evaluators = Teuchos::rcp(
        new std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>);

    if (!model_params.isSublist(model_id))
    {
        throw std::runtime_error("Solid Full Induction closure model id "
                                 + model_id + "not in list.");
    }

    constexpr int num_space_dim = NumSpaceDim;

    // Get closure model list for 'model_id'
    const Teuchos::ParameterList& closure_model_list
        = model_params.sublist(model_id);

    // Check for closure model factory type. If not `SolidFullInduction`
    // return empty evaluator
    const std::string factory_type
        = closure_model_list.isType<std::string>("Closure Factory Type")
              ? closure_model_list.get<std::string>("Closure Factory Type")
              : "None";

    if (factory_type != "Solid Full Induction")
        return evaluators;

    // Properties used by full induction MHD closure models
    const auto full_induction_params
        = closure_model_list.sublist("Full Induction MHD Properties");
    const MHDProperties::FullInductionMHDProperties mhd_props(
        full_induction_params);
    const bool build_magn_corr = mhd_props.buildMagnCorr();

    // Get the list of default-included closure models
    // Note: not all default closures are including, as determined by
    // other input options (e.g. InductionResistiveFlux)
    const std::string default_closure_models
        = "ElementLength,\n"
          "ExternalMagneticField,\n"
          "FullInductionTimeDerivative,\n"
          "Resistivity,\n"
          "InductionResistiveFlux,\n"
          "SolidFullInductionConvectiveFlux,\n"
          "SolidFullInductionLocalTimeStepSize,\n"
          "TotalMagneticField,\n"
          "TotalMagneticFieldGradient,\n";

    // Element length
    {
        auto eval
            = Teuchos::rcp(new ElementLength<EvalType, panzer::Traits>(*ir));
        evaluators->push_back(eval);
    }

    // SolidFullInductionLocalTimeStepSize closure
    {
        if (build_magn_corr)
        {
            auto eval = Teuchos::rcp(
                new SolidFullInductionLocalTimeStepSize<EvalType, panzer::Traits>(
                    *ir, mhd_props));
            evaluators->push_back(eval);
        }
        else
        {
            auto eval = Teuchos::rcp(
                new ConstantScalarField<EvalType, panzer::Traits>(
                    *ir, "local_dt", 1e6));
            evaluators->push_back(eval);
        }
    }

    // Induction time derivative closures
    {
        std::vector<Teuchos::ParameterList> mhd_names_list_vct;
        for (int dim = 0; dim < num_space_dim; ++dim)
        {
            Teuchos::ParameterList ind_names_list;
            ind_names_list.set(
                "Field Name", "induced_magnetic_field_" + std::to_string(dim));
            ind_names_list.set("Equation Name",
                               "induction_" + std::to_string(dim));
            mhd_names_list_vct.push_back(ind_names_list);
        }
        if (build_magn_corr)
        {
            Teuchos::ParameterList magn_corr_names_list;
            magn_corr_names_list.set("Field Name", "scalar_magnetic_potential");
            magn_corr_names_list.set("Equation Name",
                                     "magnetic_correction_potential");
            mhd_names_list_vct.push_back(magn_corr_names_list);
        }
        for (auto& mhd_names_list : mhd_names_list_vct)
        {
            auto eval_time = Teuchos::rcp(
                new IncompressibleVariableTimeDerivative<EvalType, panzer::Traits>(
                    *ir, mhd_names_list));
            evaluators->push_back(eval_time);
        }
    }

    // ExternalMagneticField closure, req'd for total
    {
        // Get external magnetic field parameters from the user params
        auto eval
            = Teuchos::rcp(new ExternalMagneticField<EvalType, panzer::Traits>(
                *ir, user_params.sublist("External Magnetic Field Parameters")));
        evaluators->push_back(eval);
    }

    // TotalMagneticField closure
    {
        auto eval = Teuchos::rcp(
            new TotalMagneticField<EvalType, panzer::Traits, num_space_dim>(
                *ir));
        evaluators->push_back(eval);
    }

    // SolidFullInductionConvectiveFlux closure
    // Required only when using divergence cleaning
    if (build_magn_corr)
    {
        auto eval
            = Teuchos::rcp(new SolidFullInductionConvectiveFlux<EvalType,
                                                                panzer::Traits,
                                                                num_space_dim>(
                *ir, mhd_props));
        evaluators->push_back(eval);
    }

    // Closures required for resistive flux
    if (mhd_props.buildResistiveFlux())
    {
        // TotalMagneticFieldGradient closure
        {
            auto eval = Teuchos::rcp(
                new TotalMagneticFieldGradient<EvalType,
                                               panzer::Traits,
                                               num_space_dim>(*ir));
            evaluators->push_back(eval);
        }

        // Resistivity
        {
            if (mhd_props.variableResistivity())
            {
                throw std::runtime_error(
                    "No closure models currently exist to evaluate variable "
                    "resistivity. Use a constant resistivity only.");
            }
            else
            {
                auto eval = Teuchos::rcp(
                    new ConstantScalarField<EvalType, panzer::Traits>(
                        *ir, "resistivity", mhd_props.resistivity()));
                evaluators->push_back(eval);
            }
        }

        // Resistive Flux
        {
            auto eval = Teuchos::rcp(
                new InductionResistiveFlux<EvalType, panzer::Traits, num_space_dim>(
                    *ir, mhd_props));
            evaluators->push_back(eval);
        }
    }

    // Loop over closure models
    for (const auto& closure_model : closure_model_list)
    {
        bool found_model = false;

        // skip non-closure entries
        const auto closure_name = closure_model.first;
        if (closure_name == "Closure Factory Type"
            || closure_name == "Full Induction MHD Properties")
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
                const std::string msg = "\n\nSolidFullInduction closure model/type '"
                                        + closure_name + "'/'" + closure_type
                                        + "' found in model id '" + model_id
                                        + "' is a default closure model and should "
                                        "not be added to the input file.\n"
                                        "The list of default closure models for "
                                        "the solid full induction equations is:\n"
                                        + default_closure_models;
                throw std::runtime_error(msg);
            }

            if (closure_type == "ConstantMaterialProperties")
            {
                // Density
                auto eval_rho = Teuchos::rcp(
                    new ConstantScalarField<EvalType, panzer::Traits>(
                        *ir,
                        "solid_density",
                        closure_params.get<double>("Density Value")));
                evaluators->push_back(eval_rho);
                found_model = true;
            }

            if (closure_type == "InductionConstantSource")
            {
                auto eval = Teuchos::rcp(
                    new InductionConstantSource<EvalType, panzer::Traits, NumSpaceDim>(
                        *ir, closure_params));
                evaluators->push_back(eval);
                found_model = true;
            }

            if (closure_type == "MagneticCorrectionDampingSource")
            {
                auto eval = Teuchos::rcp(
                    new MagneticCorrectionDampingSource<EvalType, panzer::Traits>(
                        *ir, mhd_props));
                evaluators->push_back(eval);
                found_model = true;
            }

            if (std::string::npos != closure_type.find("VectorFieldDivergence"))
            {
                const auto field_names
                    = closure_params.get<std::string>("Field Names");
                std::vector<std::string> tokens;
                panzer::StringTokenizer(tokens, field_names, ",", true);
                for (auto& field : tokens)
                {
                    auto eval = Teuchos::rcp(
                        new VectorFieldDivergence<EvalType,
                                                  panzer::Traits,
                                                  num_space_dim>(
                            *ir, field, closure_type));
                    evaluators->push_back(eval);
                }
                found_model = true;
            }

            // Error message if closure model not found
            if (!found_model)
            {
                const std::string msg
                    = "\n\nSolidFullInduction closure model/type "
                    + closure_name + "/" + closure_type
                    + " failed to build in model id " + model_id + ".\n"
                    "Solid full induction closure models available in Vertex-CFD are:\n"
                    "ConstantMaterialProperties\n"
                    "InductionConstantSource,\n"
                    "MagneticCorrectionDampingSource,\n"
                    "(Abs)VectorFieldDivergence\n"
                    "\n";
                throw std::runtime_error(msg);
            }
        }
    }

    return evaluators;
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_SOLIDFULLINDUCTIONCLOSUREMODELFACTORY_IMPL_HPP
