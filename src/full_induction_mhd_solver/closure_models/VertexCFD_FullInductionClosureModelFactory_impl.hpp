#ifndef VERTEXCFD_FULLINDUCTIONCLOSUREMODELFACTORY_IMPL_HPP
#define VERTEXCFD_FULLINDUCTIONCLOSUREMODELFACTORY_IMPL_HPP

#include "closure_models/VertexCFD_Closure_ConstantScalarField.hpp"
#include "closure_models/VertexCFD_Closure_ExternalMagneticField.hpp"
#include "closure_models/VertexCFD_Closure_VectorFieldDivergence.hpp"

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleVariableTimeDerivative.hpp"

#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_DivergenceCleaningSource.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_FullInductionLocalTimeStepSize.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_FullInductionModelErrorNorms.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_GodunovPowellSource.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_InductionConstantSource.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_InductionConvectiveFlux.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_InductionConvectiveMomentumFlux.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_InductionResistiveFlux.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_MHDVortexProblemExact.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_MagneticCorrectionDampingSource.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_MagneticPressure.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_TotalMagneticField.hpp"
#include "full_induction_mhd_solver/closure_models/VertexCFD_Closure_TotalMagneticFieldGradient.hpp"
#include "full_induction_mhd_solver/mhd_properties/VertexCFD_FullInductionMHDProperties.hpp"

#include <Panzer_String_Utilities.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>>
FullInductionFactory<EvalType, NumSpaceDim>::buildClosureModels(
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
        throw std::runtime_error("Full Induction closure model id " + model_id
                                 + "not in list.");
    }

    // Get closure model list for 'model_id'
    const Teuchos::ParameterList& closure_model_list
        = model_params.sublist(model_id);

    // Check for closure model factory type and return empty evaluators if it
    // is not "Full Induction MHD"
    const std::string factory_type
        = closure_model_list.isType<std::string>("Closure Factory Type")
              ? closure_model_list.get<std::string>("Closure Factory Type")
              : "None";

    if (factory_type != "Full Induction MHD")
        return evaluators;

    // Properties used by full induction MHD closure models
    const auto full_induction_params
        = closure_model_list.sublist("Full Induction MHD Properties");
    const MHDProperties::FullInductionMHDProperties mhd_props(
        full_induction_params);
    const bool build_magn_corr = mhd_props.buildMagnCorr();

    // Loop over closure models
    for (const auto& closure_model : closure_model_list)
    {
        bool found_model = false;

        const auto closure_name = closure_model.first;
        if (closure_name == "Closure Factory Type"
            || closure_name == "Full Induction MHD Properties"
            || closure_name == "Fluid Properties")
            continue;

        // Get closure parameters for current closure model
        const auto& closure_params
            = Teuchos::getValue<Teuchos::ParameterList>(closure_model.second);

        if (!closure_params.isType<std::string>("Type"))
            continue;

        const auto closure_type = closure_params.get<std::string>("Type");

        // Closure models
        if (closure_type == "InductionConvectiveFlux")
        {
            // Induction and magnetic correction equation fluxes
            auto ind_eval = Teuchos::rcp(
                new InductionConvectiveFlux<EvalType, panzer::Traits, num_space_dim>(
                    *ir, mhd_props));
            evaluators->push_back(ind_eval);

            // Momentum equation fluxes
            auto mtm_eval = Teuchos::rcp(
                new InductionConvectiveMomentumFlux<EvalType,
                                                    panzer::Traits,
                                                    num_space_dim>(*ir,
                                                                   mhd_props));
            evaluators->push_back(mtm_eval);

            found_model = true;
        }

        if (closure_type == "InductionResistiveFlux")
        {
            auto eval = Teuchos::rcp(
                new InductionResistiveFlux<EvalType, panzer::Traits, num_space_dim>(
                    *ir, mhd_props));
            evaluators->push_back(eval);
            found_model = true;
        }

        if (closure_type == "MagneticPressure")
        {
            auto eval = Teuchos::rcp(
                new MagneticPressure<EvalType, panzer::Traits>(*ir, mhd_props));
            evaluators->push_back(eval);
            found_model = true;
        }

        if (closure_type == "FullInductionTimeDerivative")
        {
            std::vector<Teuchos::ParameterList> mhd_names_list_vct;
            for (int dim = 0; dim < num_space_dim; ++dim)
            {
                Teuchos::ParameterList ind_names_list;
                ind_names_list.set(
                    "Field Name",
                    "induced_magnetic_field_" + std::to_string(dim));
                ind_names_list.set("Equation Name",
                                   "induction_" + std::to_string(dim));
                mhd_names_list_vct.push_back(ind_names_list);
            }
            if (build_magn_corr)
            {
                Teuchos::ParameterList magn_corr_names_list;
                magn_corr_names_list.set("Field Name",
                                         "scalar_magnetic_potential");
                magn_corr_names_list.set("Equation Name",
                                         "magnetic_correction_potential");
                mhd_names_list_vct.push_back(magn_corr_names_list);
            }
            for (auto& mhd_names_list : mhd_names_list_vct)
            {
                auto eval_time = Teuchos::rcp(
                    new IncompressibleVariableTimeDerivative<EvalType,
                                                             panzer::Traits>(
                        *ir, mhd_names_list));
                evaluators->push_back(eval_time);
            }
            found_model = true;
        }

        if (closure_type == "TotalMagneticField")
        {
            auto eval = Teuchos::rcp(
                new TotalMagneticField<EvalType, panzer::Traits, num_space_dim>(
                    *ir));
            evaluators->push_back(eval);
            auto eval_grad = Teuchos::rcp(
                new TotalMagneticFieldGradient<EvalType,
                                               panzer::Traits,
                                               num_space_dim>(*ir));
            evaluators->push_back(eval_grad);
            found_model = true;
        }

        if (closure_type == "MHDVortexProblemExact")
        {
            auto eval = Teuchos::rcp(
                new MHDVortexProblemExact<EvalType, panzer::Traits, NumSpaceDim>(
                    *ir, closure_params));
            evaluators->push_back(eval);
            found_model = true;
        }

        if (closure_type == "FullInductionModelErrorNorm")
        {
            auto eval = Teuchos::rcp(
                new FullInductionModelErrorNorms<EvalType,
                                                 panzer::Traits,
                                                 NumSpaceDim>(*ir));
            evaluators->push_back(eval);
            found_model = true;
        }

        if (closure_type == "DivergenceCleaningSource")
        {
            auto eval = Teuchos::rcp(
                new DivergenceCleaningSource<EvalType, panzer::Traits, num_space_dim>(
                    *ir));
            evaluators->push_back(eval);
            found_model = true;
        }

        if (closure_type == "GodunovPowellSource")
        {
            auto eval = Teuchos::rcp(
                new GodunovPowellSource<EvalType, panzer::Traits, NumSpaceDim>(
                    *ir, mhd_props));
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

        if (closure_type == "FullInductionLocalTimeStepSize")
        {
            auto eval = Teuchos::rcp(
                new FullInductionLocalTimeStepSize<EvalType,
                                                   panzer::Traits,
                                                   NumSpaceDim>(*ir, mhd_props));
            evaluators->push_back(eval);
            found_model = true;
        }

        if (closure_type == "Resistivity")
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
                found_model = true;
            }
        }

        if (closure_type == "InductionConstantSource")
        {
            auto eval = Teuchos::rcp(
                new InductionConstantSource<EvalType, panzer::Traits, NumSpaceDim>(
                    *ir, closure_params));
            evaluators->push_back(eval);
            found_model = true;
        }

        if (closure_type == "ExternalMagneticField")
        {
            // Get external magnetic field parameters from the user params
            auto eval = Teuchos::rcp(
                new ExternalMagneticField<EvalType, panzer::Traits>(
                    *ir,
                    user_params.sublist("External Magnetic Field "
                                        "Parameters")));
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
                if (std::string::npos != field.find("total_magnetic_field"))
                {
                    throw std::runtime_error(
                        "The layout of \"total_magnetic_field\" is "
                        "incompatible with the current implementation of "
                        "\"VectorFieldDivergence\", and should not be "
                        "included in the list of field names for this closure "
                        "model. The field \"divergence_total_magnetic_field\" "
                        "is evaluated within the "
                        "\"TotalMagneticFieldGradient\" closure model.");
                }

                auto eval = Teuchos::rcp(
                    new VectorFieldDivergence<EvalType, panzer::Traits, num_space_dim>(
                        *ir, field, closure_type));
                evaluators->push_back(eval);
            }
            found_model = true;
        }

        if (!found_model)
        {
            const std::string msg = "\n\nFull Induction MHD closure model/type "
                              + closure_name + "/" + closure_type
                              + " failed to build in model id " + model_id
                              + ".\n"
                              "The conduction closure models available in Vertex-CFD are:\n"
                              + availableClosureModels();
            throw std::runtime_error(msg);
        }
    }

    return evaluators;
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_FULLINDUCTIONCLOSUREMODELFACTORY_IMPL_HPP
