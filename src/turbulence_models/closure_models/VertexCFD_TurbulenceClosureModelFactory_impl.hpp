#ifndef VERTEXCFD_TURBULENCECLOSUREMODELFACTORY_IMPL_HPP
#define VERTEXCFD_TURBULENCECLOSUREMODELFACTORY_IMPL_HPP

#include "closure_models/VertexCFD_Closure_ElementLength.hpp"
#include "closure_models/VertexCFD_Closure_MeasureElementLength.hpp"
#include "closure_models/VertexCFD_Closure_MetricTensorElementLength.hpp"
#include "closure_models/VertexCFD_Closure_SingularValueElementLength.hpp"
#include "closure_models/VertexCFD_Closure_VariableConvectiveFlux.hpp"
#include "closure_models/VertexCFD_Closure_VariableDiffusionFlux.hpp"
#include "closure_models/VertexCFD_Closure_VariableSUPGFlux.hpp"
#include "closure_models/VertexCFD_Closure_VariableTauSUPG.hpp"

#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleChienKEpsilonEddyViscosity.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleChienKEpsilonSource.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleKEpsilonDiffusivityCoefficient.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleKEpsilonEddyViscosity.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleKEpsilonSource.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleKOmegaDiffusivityCoefficient.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleKOmegaEddyViscosity.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleKOmegaSource.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleKTauDiffusivityCoefficient.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleKTauEddyViscosity.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleKTauSource.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleRealizableKEpsilonEddyViscosity.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleRealizableKEpsilonSource.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleSSTDiffusivityCoefficient.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleSSTEddyViscosity.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleSSTSource.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleSpalartAllmarasDiffusivityCoefficient.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleSpalartAllmarasEddyViscosity.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleSpalartAllmarasSource.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleWALEEddyViscosity.hpp"
#include "turbulence_models/closure_models/VertexCFD_TurbulenceClosureModelFactory.hpp"

#include "incompressible_solver/closure_models/VertexCFD_Closure_IncompressibleVariableTimeDerivative.hpp"
#include "incompressible_solver/fluid_properties/VertexCFD_ConstantFluidProperties.hpp"

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
void TurbulenceFactory<EvalType, NumSpaceDim>::buildClosureModel(
    const Teuchos::RCP<panzer::IntegrationRule>& ir,
    const Teuchos::RCP<panzer::GlobalData>& global_data,
    const Teuchos::ParameterList& user_params,
    const std::string& turbulence_model_name,
    Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>>
        evaluators)
{
    // Turbulence parameters
    const auto turb_params = user_params.isSublist("Turbulence Parameters")
                                 ? user_params.sublist("Turbulence Parameters")
                                 : Teuchos::ParameterList();

    // Field name and variable name for each turbulence model
    std::vector<Teuchos::ParameterList> turb_names_list_vct;

    if (std::string::npos != turbulence_model_name.find("Spalart-Allmaras"))
    {
        Teuchos::ParameterList sa_names_list;
        sa_names_list.set("Field Name", "spalart_allmaras_variable");
        sa_names_list.set("Equation Name", "spalart_allmaras_equation");
        turb_names_list_vct.push_back(sa_names_list);
    }
    else if (std::string::npos != turbulence_model_name.find("K-Epsilon"))
    {
        Teuchos::ParameterList k_names_list;
        k_names_list.set("Field Name", "turb_kinetic_energy");
        k_names_list.set("Equation Name", "turb_kinetic_energy_equation");
        turb_names_list_vct.push_back(k_names_list);

        Teuchos::ParameterList epsilon_names_list;
        epsilon_names_list.set("Field Name", "turb_dissipation_rate");
        epsilon_names_list.set("Equation Name",
                               "turb_dissipation_rate_equation");
        turb_names_list_vct.push_back(epsilon_names_list);
    }
    else if (std::string::npos != turbulence_model_name.find("K-Omega"))
    {
        Teuchos::ParameterList k_names_list;
        k_names_list.set("Field Name", "turb_kinetic_energy");
        k_names_list.set("Equation Name", "turb_kinetic_energy_equation");
        turb_names_list_vct.push_back(k_names_list);

        Teuchos::ParameterList omega_names_list;
        omega_names_list.set("Field Name", "turb_specific_dissipation_rate");
        omega_names_list.set("Equation Name",
                             "turb_specific_dissipation_rate_equation");
        turb_names_list_vct.push_back(omega_names_list);
    }
    else if (std::string::npos != turbulence_model_name.find("SST"))
    {
        Teuchos::ParameterList k_names_list;
        k_names_list.set("Field Name", "turb_kinetic_energy");
        k_names_list.set("Equation Name", "turb_kinetic_energy_equation");
        turb_names_list_vct.push_back(k_names_list);

        Teuchos::ParameterList omega_names_list;
        omega_names_list.set("Field Name", "turb_specific_dissipation_rate");
        omega_names_list.set("Equation Name",
                             "turb_specific_dissipation_rate_equation");
        turb_names_list_vct.push_back(omega_names_list);
    }
    else if (std::string::npos != turbulence_model_name.find("K-Tau"))
    {
        Teuchos::ParameterList k_names_list;
        k_names_list.set("Field Name", "turb_kinetic_energy");
        k_names_list.set("Equation Name", "turb_kinetic_energy_equation");
        turb_names_list_vct.push_back(k_names_list);

        Teuchos::ParameterList tau_names_list;
        tau_names_list.set("Field Name", "turb_specific_dissipation_rate");
        tau_names_list.set("Equation Name",
                           "turb_specific_dissipation_rate_equation");
        turb_names_list_vct.push_back(tau_names_list);
    }

    // Add generic closure models for each variable in turbulence model
    for (const auto& turb_names_list : turb_names_list_vct)
    {
        auto eval_time = Teuchos::rcp(
            new IncompressibleVariableTimeDerivative<EvalType, panzer::Traits>(
                *ir, turb_names_list));
        evaluators->push_back(eval_time);

        auto eval_conv = Teuchos::rcp(
            new VariableConvectiveFlux<EvalType, panzer::Traits, num_space_dim>(
                *ir, turb_names_list));
        evaluators->push_back(eval_conv);

        auto eval_diff
            = Teuchos::rcp(new VariableDiffusionFlux<EvalType, panzer::Traits>(
                *ir, turb_names_list));
        evaluators->push_back(eval_diff);

        // SUPG numerical method
        if (turb_params.isSublist("SUPG Parameters"))
        {
            const auto& supg_params = turb_params.sublist("SUPG Parameters");
            auto eval_tau_supg = Teuchos::rcp(
                new VariableTauSUPG<EvalType, panzer::Traits, num_space_dim>(
                    *ir, turb_names_list, supg_params));
            evaluators->push_back(eval_tau_supg);

            auto eval_supg_flux = Teuchos::rcp(
                new VariableSUPGFlux<EvalType, panzer::Traits, num_space_dim>(
                    *ir, turb_names_list));
            evaluators->push_back(eval_supg_flux);
        }
    }

    // Spalart-Allmaras closure models
    if (std::string::npos != turbulence_model_name.find("Spalart-Allmaras"))
    {
        auto eval_coeff = Teuchos::rcp(
            new IncompressibleSpalartAllmarasDiffusivityCoefficient<EvalType,
                                                                    panzer::Traits>(
                *ir));
        evaluators->push_back(eval_coeff);

        auto eval_source = Teuchos::rcp(
            new IncompressibleSpalartAllmarasSource<EvalType,
                                                    panzer::Traits,
                                                    num_space_dim>(*ir));
        evaluators->push_back(eval_source);

        auto eval_eddy = Teuchos::rcp(
            new IncompressibleSpalartAllmarasEddyViscosity<EvalType,
                                                           panzer::Traits>(*ir));
        evaluators->push_back(eval_eddy);
    }
    // K-Epsilon model family closure models
    else if (std::string::npos != turbulence_model_name.find("K-Epsilon"))
    {
        // Realizable K-Epsilon closure models
        if (std::string::npos
            != turbulence_model_name.find("Realizable K-Epsilon"))
        {
            auto eval_coeff = Teuchos::rcp(
                new IncompressibleKEpsilonDiffusivityCoefficient<EvalType,
                                                                 panzer::Traits>(
                    *ir, 1.0, 1.2));
            evaluators->push_back(eval_coeff);

            auto eval_eddy = Teuchos::rcp(
                new IncompressibleRealizableKEpsilonEddyViscosity<EvalType,
                                                                  panzer::Traits,
                                                                  num_space_dim>(
                    *ir));
            evaluators->push_back(eval_eddy);

            auto eval_source = Teuchos::rcp(
                new IncompressibleRealizableKEpsilonSource<EvalType,
                                                           panzer::Traits,
                                                           num_space_dim>(*ir));
            evaluators->push_back(eval_source);
        }
        // Chien K-Epsilon closure models
        else if (std::string::npos
                 != turbulence_model_name.find("Chien K-Epsilon"))
        {
            auto eval_coeff = Teuchos::rcp(
                new IncompressibleKEpsilonDiffusivityCoefficient<EvalType,
                                                                 panzer::Traits>(
                    *ir));
            evaluators->push_back(eval_coeff);

            auto eval_eddy = Teuchos::rcp(
                new IncompressibleChienKEpsilonEddyViscosity<EvalType,
                                                             panzer::Traits>(
                    *ir, global_data, user_params));
            evaluators->push_back(eval_eddy);

            auto eval_source = Teuchos::rcp(
                new IncompressibleChienKEpsilonSource<EvalType,
                                                      panzer::Traits,
                                                      num_space_dim>(
                    *ir, global_data, user_params));
            evaluators->push_back(eval_source);
        }
        // Standard K-Epsilon closure models
        else
        {
            auto eval_coeff = Teuchos::rcp(
                new IncompressibleKEpsilonDiffusivityCoefficient<EvalType,
                                                                 panzer::Traits>(
                    *ir));
            evaluators->push_back(eval_coeff);

            auto eval_eddy = Teuchos::rcp(
                new IncompressibleKEpsilonEddyViscosity<EvalType, panzer::Traits>(
                    *ir));
            evaluators->push_back(eval_eddy);

            auto eval_source = Teuchos::rcp(
                new IncompressibleKEpsilonSource<EvalType,
                                                 panzer::Traits,
                                                 num_space_dim>(*ir));
            evaluators->push_back(eval_source);
        }
    }
    // K-Omega model family closure models
    else if (std::string::npos != turbulence_model_name.find("K-Omega"))
    {
        auto eval_coeff = Teuchos::rcp(
            new IncompressibleKOmegaDiffusivityCoefficient<EvalType,
                                                           panzer::Traits>(
                *ir, user_params));
        evaluators->push_back(eval_coeff);

        auto eval_source = Teuchos::rcp(
            new IncompressibleKOmegaSource<EvalType, panzer::Traits, num_space_dim>(
                *ir, user_params));
        evaluators->push_back(eval_source);

        auto eval_eddy = Teuchos::rcp(
            new IncompressibleKOmegaEddyViscosity<EvalType,
                                                  panzer::Traits,
                                                  num_space_dim>(*ir,
                                                                 user_params));
        evaluators->push_back(eval_eddy);
    }
    else if (std::string::npos != turbulence_model_name.find("SST"))
    {
        auto eval_coeff = Teuchos::rcp(
            new IncompressibleSSTDiffusivityCoefficient<EvalType, panzer::Traits>(
                *ir, user_params));
        evaluators->push_back(eval_coeff);

        auto eval_source = Teuchos::rcp(
            new IncompressibleSSTSource<EvalType, panzer::Traits, num_space_dim>(
                *ir, user_params));
        evaluators->push_back(eval_source);

        auto eval_eddy
            = Teuchos::rcp(new IncompressibleSSTEddyViscosity<EvalType,
                                                              panzer::Traits,
                                                              num_space_dim>(
                *ir, user_params));
        evaluators->push_back(eval_eddy);
    }
    // K-Tau model family closure models
    else if (std::string::npos != turbulence_model_name.find("K-Tau"))
    {
        auto eval_coeff = Teuchos::rcp(
            new IncompressibleKTauDiffusivityCoefficient<EvalType, panzer::Traits>(
                *ir, user_params));
        evaluators->push_back(eval_coeff);

        auto eval_source = Teuchos::rcp(
            new IncompressibleKTauSource<EvalType, panzer::Traits, num_space_dim>(
                *ir, user_params));
        evaluators->push_back(eval_source);

        auto eval_eddy = Teuchos::rcp(
            new IncompressibleKTauEddyViscosity<EvalType,
                                                panzer::Traits,
                                                num_space_dim>(*ir));
        evaluators->push_back(eval_eddy);
    }
    // WALE algebraic LES model
    else if (std::string::npos != turbulence_model_name.find("WALE"))
    {
        // WALE eddy viscosity
        auto eval_eddy
            = Teuchos::rcp(new IncompressibleWALEEddyViscosity<EvalType,
                                                               panzer::Traits,
                                                               num_space_dim>(
                *ir, user_params));
        evaluators->push_back(eval_eddy);

        // Delta (mesh length scale) evaluator
        const std::string delta_prefix = "les_";
        const std::string delta_type
            = turb_params.isType<std::string>("LES Element Length")
                  ? turb_params.get<std::string>("LES Element Length")
                  : "ElementLength";

        if (delta_type == "ElementLength")
        {
            auto eval_delta = Teuchos::rcp(
                new ElementLength<EvalType, panzer::Traits>(*ir, delta_prefix));

            evaluators->push_back(eval_delta);
        }
        else if (delta_type == "MeasureElementLength")
        {
            auto eval_delta = Teuchos::rcp(
                new MeasureElementLength<EvalType, panzer::Traits>(
                    *ir, delta_prefix));

            evaluators->push_back(eval_delta);
        }
        else if (delta_type == "MetricTensorElementLength")
        {
            auto eval_delta = Teuchos::rcp(
                new MetricTensorElementLength<EvalType, panzer::Traits>(
                    *ir, delta_prefix));

            evaluators->push_back(eval_delta);
        }
        else if (delta_type == "SingularValueElementLength")
        {
            const auto method
                = turb_params.get<std::string>("Element Length Method");

            auto eval_delta = Teuchos::rcp(
                new SingularValueElementLength<EvalType, panzer::Traits>(
                    *ir, method, delta_prefix));

            evaluators->push_back(eval_delta);
        }
        else
        {
            std::string msg = "Unknown Delta Closure Model:\n";

            msg += delta_type;
            msg += "\n";
            msg += "Please choose from:\n";
            msg += "ElementLength\n";
            msg += "MeasureElementLength\n";
            msg += "MetricTensorElementLength\n";
            msg += "SingularValueElementLength\n";

            throw std::runtime_error(msg);
        }
    }
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_TURBULENCECLOSUREMODELFACTORY_IMPL_HPP
