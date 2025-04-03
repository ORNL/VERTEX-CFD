#ifndef VERTEXCFD_TURBULENCECLOSUREMODELFACTORY_IMPL_HPP
#define VERTEXCFD_TURBULENCECLOSUREMODELFACTORY_IMPL_HPP

#include "closure_models/VertexCFD_Closure_ElementLength.hpp"
#include "closure_models/VertexCFD_Closure_MeasureElementLength.hpp"
#include "closure_models/VertexCFD_Closure_MetricTensorElementLength.hpp"
#include "closure_models/VertexCFD_Closure_SingularValueElementLength.hpp"

#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleKEpsilonDiffusivityCoefficient.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleKEpsilonEddyViscosity.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleKEpsilonSource.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleKOmegaDiffusivityCoefficient.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleKOmegaEddyViscosity.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleKOmegaSource.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleRealizableKEpsilonEddyViscosity.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleRealizableKEpsilonSource.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleSpalartAllmarasDiffusivityCoefficient.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleSpalartAllmarasEddyViscosity.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleSpalartAllmarasSource.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleVariableConvectiveFlux.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleVariableDiffusionFlux.hpp"
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
    const Teuchos::ParameterList& user_params,
    const std::string& turbulence_model_name,
    Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>>
        evaluators)
{
    // Fluid properties
    Teuchos::ParameterList fluid_prop_list
        = user_params.sublist("Fluid Properties");
    const bool build_temp_equ
        = user_params.isType<bool>("Build Temperature Equation")
              ? user_params.get<bool>("Build Temperature Equation")
              : false;
    const bool build_buoyancy_source
        = user_params.isType<bool>("Build Buoyancy Source")
              ? user_params.get<bool>("Build Buoyancy Source")
              : false;
    const bool build_ind_less_equ
        = user_params.isType<bool>("Build Inductionless MHD Equation")
              ? user_params.get<bool>("Build Inductionless MHD Equation")
              : false;
    fluid_prop_list.set<bool>("Build Temperature Equation", build_temp_equ);
    fluid_prop_list.set<bool>("Build Buoyancy Source", build_buoyancy_source);
    fluid_prop_list.set<bool>("Build Inductionless MHD Equation",
                              build_ind_less_equ);
    FluidProperties::ConstantFluidProperties incompressible_fluidprop_params
        = FluidProperties::ConstantFluidProperties(fluid_prop_list);

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

    // Add generic closure models for each variable in turbulence model
    for (auto& turb_names_list : turb_names_list_vct)
    {
        auto eval_time = Teuchos::rcp(
            new IncompressibleVariableTimeDerivative<EvalType, panzer::Traits>(
                *ir, turb_names_list));
        evaluators->push_back(eval_time);

        auto eval_conv = Teuchos::rcp(
            new IncompressibleVariableConvectiveFlux<EvalType,
                                                     panzer::Traits,
                                                     num_space_dim>(
                *ir, turb_names_list));
        evaluators->push_back(eval_conv);

        auto eval_diff = Teuchos::rcp(
            new IncompressibleVariableDiffusionFlux<EvalType,
                                                    panzer::Traits,
                                                    num_space_dim>(
                *ir, turb_names_list));
        evaluators->push_back(eval_diff);
    }

    // Spalart-Allmaras closure models
    if (std::string::npos != turbulence_model_name.find("Spalart-Allmaras"))
    {
        auto eval_coeff = Teuchos::rcp(
            new IncompressibleSpalartAllmarasDiffusivityCoefficient<EvalType,
                                                                    panzer::Traits>(
                *ir, incompressible_fluidprop_params));
        evaluators->push_back(eval_coeff);

        auto eval_source = Teuchos::rcp(
            new IncompressibleSpalartAllmarasSource<EvalType,
                                                    panzer::Traits,
                                                    num_space_dim>(
                *ir, incompressible_fluidprop_params));
        evaluators->push_back(eval_source);

        auto eval_eddy = Teuchos::rcp(
            new IncompressibleSpalartAllmarasEddyViscosity<EvalType,
                                                           panzer::Traits>(
                *ir, incompressible_fluidprop_params));
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
                    *ir, incompressible_fluidprop_params, 1.0, 1.2));
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
                                                           num_space_dim>(
                    *ir, incompressible_fluidprop_params));
            evaluators->push_back(eval_source);
        }
        // Standard K-Epsilon closure models
        else
        {
            auto eval_coeff = Teuchos::rcp(
                new IncompressibleKEpsilonDiffusivityCoefficient<EvalType,
                                                                 panzer::Traits>(
                    *ir, incompressible_fluidprop_params));
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
                *ir, incompressible_fluidprop_params, user_params));
        evaluators->push_back(eval_coeff);

        auto eval_source = Teuchos::rcp(
            new IncompressibleKOmegaSource<EvalType, panzer::Traits, num_space_dim>(
                *ir, user_params));
        evaluators->push_back(eval_source);

        auto eval_eddy = Teuchos::rcp(
            new IncompressibleKOmegaEddyViscosity<EvalType,
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
        const auto turb_params
            = user_params.isSublist("Turbulence Parameters")
                  ? user_params.sublist("Turbulence Parameters")
                  : Teuchos::ParameterList();
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
