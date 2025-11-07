#ifndef VERTEXCFD_CLOSUREMODELFACTORY_IMPL_HPP
#define VERTEXCFD_CLOSUREMODELFACTORY_IMPL_HPP

#include "VertexCFD_Closure_ElementLength.hpp"
#include "VertexCFD_Closure_ExternalInterpolation.hpp"
#include "VertexCFD_Closure_ExternalMagneticField.hpp"
#include "VertexCFD_Closure_MeasureElementLength.hpp"
#include "VertexCFD_Closure_MetricTensor.hpp"
#include "VertexCFD_Closure_MetricTensorElementLength.hpp"
#include "VertexCFD_Closure_SingularValueElementLength.hpp"
#include "VertexCFD_Closure_VectorFieldDivergence.hpp"
#include "VertexCFD_Closure_WallDistance.hpp"

#include "incompressible_lsvof_solver/closure_models/VertexCFD_IncompressibleLSVOFClosureModelFactory.hpp"
#include "incompressible_solver/closure_models/VertexCFD_IncompressibleClosureModelFactory.hpp"
#include "incompressible_solver/fluid_properties/VertexCFD_Closure_IncompressibleFluidProperties.hpp"
#include "induction_less_mhd_solver/closure_models/VertexCFD_InductionlessClosureModelFactory.hpp"
#include "turbulence_models/closure_models/VertexCFD_TurbulenceClosureModelFactory.hpp"

#include "mesh/VertexCFD_Mesh_GeometryData.hpp"
#include "utils/VertexCFD_Utils_VectorizeOutputFieldNames.hpp"

#include <optional>

namespace VertexCFD
{
namespace ClosureModel
{
//---------------------------------------------------------------------------//
template<class EvalType, int NumSpaceDim>
Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>>
Factory<EvalType, NumSpaceDim>::buildClosureModels(
    const std::string& model_id,
    const Teuchos::ParameterList& model_params,
    const panzer::FieldLayoutLibrary&,
    const Teuchos::RCP<panzer::IntegrationRule>& ir,
    const Teuchos::ParameterList&,
    const Teuchos::ParameterList& user_params_orig,
    const Teuchos::RCP<panzer::GlobalData>& global_data,
    PHX::FieldManager<panzer::Traits>&) const
{
    auto evaluators = Teuchos::rcp(
        new std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>);

    constexpr int num_space_dim = NumSpaceDim;

    // Make a hard copy of `user_params_const` that we can modify in here
    Teuchos::ParameterList user_params = user_params_orig;

    // Check for closure model, and make local copy
    Teuchos::ParameterList closure_model_list;
    if (!model_params.isSublist(model_id))
    {
        throw std::runtime_error("Closure model id not in list");
    }
    else
    {
        closure_model_list = model_params.sublist(model_id);
    }

    // Check for closure model factory type and remove it if found
    std::string factory_type = "Navier Stokes";
    if (closure_model_list.isType<std::string>("Closure Factory Type"))
    {
        factory_type
            = closure_model_list.get<std::string>("Closure Factory Type");
        closure_model_list.remove("Closure Factory Type");
    }

    // Return empty evaluators if the factory type is not found in
    // 'factory_list'
    const std::string factory_list = "LSVOF, Navier Stokes";
    if (factory_list.find(factory_type) == std::string::npos)
        return evaluators;

    // Build LSVOF factory if LSVOF factory type specified
    std::optional<IncompressibleLSVOFFactory<EvalType, NumSpaceDim>> lsvof_factory
        = factory_type == "LSVOF"
              ? std::make_optional<
                    IncompressibleLSVOFFactory<EvalType, NumSpaceDim>>()
              : std::nullopt;
    std::string lsvof_error_msg = "None\n";

    if (lsvof_factory)
    {
        lsvof_factory->buildDefaultClosureModels(
            ir, closure_model_list, user_params, evaluators);
    }

    // Incompressible factory model object required for "Navier Stokes"
    std::optional<IncompressibleFactory<EvalType, NumSpaceDim>> incomp_factory
        = factory_type == "Navier Stokes"
              ? std::make_optional<IncompressibleFactory<EvalType, NumSpaceDim>>()
              : std::nullopt;
    std::string incomp_error_msg = "None\n";

    // Inductionless solver factory object
    std::optional<InductionlessFactory<EvalType, NumSpaceDim>> inductionless_factory
        = user_params.isType<bool>("Build Inductionless MHD Equation")
              ? std::make_optional<InductionlessFactory<EvalType, NumSpaceDim>>()
              : std::nullopt;
    std::string ind_less_error_msg = "None\n";

    // Get fluid properties for incompressible NS equations: read the Fluid
    // Properties from the closure model list and remove it from the list.
    Teuchos::ParameterList fluid_prop_list;
    if (incomp_factory)
    {
        fluid_prop_list = closure_model_list.sublist("Fluid Properties");
        closure_model_list.remove("Fluid Properties");
        const bool build_temp_equ
            = user_params.isType<bool>("Build Temperature Equation")
                  ? user_params.get<bool>("Build Temperature Equation")
                  : false;
        const bool build_buoyancy_source
            = user_params.isType<bool>("Build Buoyancy Source")
                  ? user_params.get<bool>("Build Buoyancy Source")
                  : false;
        fluid_prop_list.set<bool>("Build Inductionless MHD Equation",
                                  inductionless_factory.has_value());
        fluid_prop_list.set<bool>("Build Temperature Equation", build_temp_equ);
        fluid_prop_list.set<bool>("Build Buoyancy Source",
                                  build_buoyancy_source);
        user_params.set<Teuchos::ParameterList>("Fluid Properties List",
                                                fluid_prop_list);

        auto eval = Teuchos::rcp(
            new FluidProperties::IncompressibleFluidProperties<EvalType,
                                                               panzer::Traits>(
                *ir, fluid_prop_list));
        evaluators->push_back(eval);
    }

    // Get stability parameters for incompressible NS equations: read the
    // Stability Parameters from the closure model list and remove it from the
    // list.
    Teuchos::ParameterList stability_param_list;
    if (incomp_factory && closure_model_list.isSublist("Stability Parameters"))
    {
        stability_param_list
            = closure_model_list.sublist("Stability Parameters");
        closure_model_list.remove("Stability Parameters");
        user_params.set<Teuchos::ParameterList>("Stability Parameters",
                                                stability_param_list);
    }
    else if (incomp_factory
             && !closure_model_list.isSublist("Stability Parameters"))
    {
        // We need to define an empty list for incompressible factory.
        user_params.set<Teuchos::ParameterList>("Stability Parameters",
                                                stability_param_list);
    }

    // Turbulence model parameters
    const std::string turbulence_model_name
        = user_params.isType<std::string>("Turbulence Model")
              ? user_params.get<std::string>("Turbulence Model")
              : "No Turbulence Model";
    std::optional<TurbulenceFactory<EvalType, NumSpaceDim>> tm_factory
        = (turbulence_model_name != "No Turbulence Model")
              ? std::make_optional<TurbulenceFactory<EvalType, NumSpaceDim>>()
              : std::nullopt;
    const bool use_turbulence_model = tm_factory.has_value();

    if (tm_factory)
    {
        tm_factory->buildClosureModel(
            ir, global_data, user_params, turbulence_model_name, evaluators);
    }

    // Closure model block in XML input file for `model_id`
    for (const auto& closure_model : closure_model_list)
    {
        bool found_model = false;

        const auto closure_name = closure_model.first;

        const auto& closure_params
            = Teuchos::getValue<Teuchos::ParameterList>(closure_model.second);

        if (closure_params.isType<std::string>("Type"))
        {
            const auto closure_type = closure_params.get<std::string>("Type");

            // Incompressible factory
            if (incomp_factory)
            {
                incomp_factory->buildClosureModel(closure_type,
                                                  ir,
                                                  global_data,
                                                  user_params,
                                                  closure_params,
                                                  use_turbulence_model,
                                                  found_model,
                                                  incomp_error_msg,
                                                  evaluators);
            }
            // LSVOF factory
            else if (lsvof_factory)
            {
                lsvof_factory->buildClosureModel(closure_type,
                                                 ir,
                                                 user_params,
                                                 closure_params,
                                                 found_model,
                                                 lsvof_error_msg,
                                                 evaluators);
            }

            // Inductionless MHD factory
            if (inductionless_factory)
            {
                inductionless_factory->buildClosureModel(closure_type,
                                                         ir,
                                                         user_params,
                                                         closure_params,
                                                         found_model,
                                                         ind_less_error_msg,
                                                         evaluators);
            }

            if (closure_type == "ExternalMagneticField")
            {
                auto eval = Teuchos::rcp(
                    new ExternalMagneticField<EvalType, panzer::Traits>(
                        *ir, user_params));
                evaluators->push_back(eval);
                found_model = true;
            }

            if (closure_type == "MetricTensor")
            {
                auto eval = Teuchos::rcp(
                    new MetricTensor<EvalType, panzer::Traits>(*ir));
                evaluators->push_back(eval);
                found_model = true;
            }

            if (closure_type == "ElementLength")
            {
                auto eval = Teuchos::rcp(
                    new ElementLength<EvalType, panzer::Traits>(*ir));
                evaluators->push_back(eval);
                found_model = true;
            }
            else if (closure_type == "MetricTensorElementLength")
            {
                auto eval = Teuchos::rcp(
                    new MetricTensorElementLength<EvalType, panzer::Traits>(
                        *ir));
                evaluators->push_back(eval);
                found_model = true;
            }
            else if (closure_type == "MeasureElementLength")
            {
                auto eval = Teuchos::rcp(
                    new MeasureElementLength<EvalType, panzer::Traits>(*ir));
                evaluators->push_back(eval);
                found_model = true;
            }
            else if (closure_type == "SingularValueElementLength")
            {
                const auto method
                    = closure_params.get<std::string>("Element Length Method");
                auto eval = Teuchos::rcp(
                    new SingularValueElementLength<EvalType, panzer::Traits>(
                        *ir, method));
                evaluators->push_back(eval);
                found_model = true;
            }

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

            if (closure_type == "WallDistance")
            {
                auto eval = Teuchos::rcp(
                    new WallDistance<EvalType, panzer::Traits, num_space_dim>(
                        *ir,
                        user_params
                            .get<Teuchos::RCP<Mesh::Topology::SidesetGeometry>>(
                                "Sideset Geometry")));
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
        }

        if (!found_model)
        {
            std::string msg = "Closure model " + closure_name
                              + " failed to build.\n";
            msg += "The closure models implemented in VertexCFD are:\n";
            msg += "MeasureElementLength\n";
            msg += "MetricTensor\n";
            msg += "MetricTensorElementLength\n";
            msg += "SingularValueElementLength\n";
            msg += "ThermalConductivity\n";
            msg += "VectorFieldDivergence\n";
            msg += "AbsVectorFieldDivergence\n";
            msg += "=================================\n";
            msg += "Incompressible closure models:\n";
            msg += incomp_error_msg;
            msg += "=================================\n";
            msg += "LSVOF closure models:\n";
            msg += lsvof_error_msg;
            msg += "=================================\n";
            msg += "Inductionless MHD closure models:\n";
            msg += ind_less_error_msg;

            throw std::runtime_error(msg);
        }
    }

    return evaluators;
}

//---------------------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // end VERTEXCFD_CLOSUREMODELFACTORY_IMPL_HPP
