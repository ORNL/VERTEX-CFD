#ifndef VERTEXCFD_TURBULENCEBOUNDARYSTATE_FACTORY_HPP
#define VERTEXCFD_TURBULENCEBOUNDARYSTATE_FACTORY_HPP

#include "closure_models/VertexCFD_Closure_ElementLength.hpp"
#include "closure_models/VertexCFD_Closure_MeasureElementLength.hpp"
#include "closure_models/VertexCFD_Closure_MetricTensorElementLength.hpp"
#include "closure_models/VertexCFD_Closure_SingularValueElementLength.hpp"

#include "turbulence_models/boundary_conditions/VertexCFD_BoundaryState_TurbulenceExtrapolate.hpp"
#include "turbulence_models/boundary_conditions/VertexCFD_BoundaryState_TurbulenceFixed.hpp"
#include "turbulence_models/boundary_conditions/VertexCFD_BoundaryState_TurbulenceInletOutlet.hpp"
#include "turbulence_models/boundary_conditions/VertexCFD_BoundaryState_TurbulenceKEpsilonWallFunction.hpp"
#include "turbulence_models/boundary_conditions/VertexCFD_BoundaryState_TurbulenceKOmegaWallResolved.hpp"
#include "turbulence_models/boundary_conditions/VertexCFD_BoundaryState_TurbulenceSymmetry.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleKEpsilonDiffusivityCoefficient.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleKEpsilonEddyViscosity.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleKOmegaDiffusivityCoefficient.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleKOmegaEddyViscosity.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleRealizableKEpsilonEddyViscosity.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleSpalartAllmarasDiffusivityCoefficient.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleSpalartAllmarasEddyViscosity.hpp"
#include "turbulence_models/closure_models/VertexCFD_Closure_IncompressibleWALEEddyViscosity.hpp"

#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
template<class EvalType, class Traits, int NumSpaceDim>
class TurbulenceBoundaryStateFactory
{
  public:
    static std::vector<Teuchos::RCP<PHX::Evaluator<Traits>>>
    create(const panzer::IntegrationRule& ir,
           const Teuchos::ParameterList& bc_params,
           const Teuchos::ParameterList& user_params,
           const std::string _turbulence_model_name,
           const FluidProperties::ConstantFluidProperties& fluid_prop)
    {
        // Evaluator vector to return
        std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>> evaluators;

        // Adding diffusivity coefficient closure model and initializing
        // `turb_field_names_vct` for each equation of the turbulence model
        std::vector<std::string> turb_field_names_vct;
        if (_turbulence_model_name != "No Turbulence Model")
        {
            // Spalart-Allmaras model family
            if (std::string::npos
                != _turbulence_model_name.find("Spalart-Allmaras"))
            {
                // Variable name
                turb_field_names_vct.push_back("spalart_allmaras_variable");

                // Diffusive coefficient
                const auto diffusive_coeff_op = Teuchos::rcp(
                    new ClosureModel::IncompressibleSpalartAllmarasDiffusivityCoefficient<
                        EvalType,
                        panzer::Traits>(ir, fluid_prop));
                evaluators.push_back(diffusive_coeff_op);

                // Turbulent eddy viscosity
                const auto eddy_visc_op = Teuchos::rcp(
                    new ClosureModel::IncompressibleSpalartAllmarasEddyViscosity<
                        EvalType,
                        panzer::Traits>(ir, fluid_prop));
                evaluators.push_back(eddy_visc_op);
            }
            // K-Epsilon model family
            else if (std::string::npos
                     != _turbulence_model_name.find("K-Epsilon"))
            {
                // K-Epsilon turbulence variable names
                turb_field_names_vct.push_back("turb_kinetic_energy");
                turb_field_names_vct.push_back("turb_dissipation_rate");

                // Realizable K-Epsilon closure models
                if (std::string::npos
                    != _turbulence_model_name.find("Realizable K-Epsilon"))
                {
                    // Diffusive coefficient with non-standard coefficients
                    const auto diffusive_coeff_op = Teuchos::rcp(
                        new ClosureModel::IncompressibleKEpsilonDiffusivityCoefficient<
                            EvalType,
                            panzer::Traits>(
                            ir, fluid_prop, 1.0, 1.2, "BOUNDARY_"));
                    evaluators.push_back(diffusive_coeff_op);

                    // Turbulent eddy viscosity
                    const auto eddy_visc_op = Teuchos::rcp(
                        new ClosureModel::IncompressibleRealizableKEpsilonEddyViscosity<
                            EvalType,
                            panzer::Traits,
                            NumSpaceDim>(ir));
                    evaluators.push_back(eddy_visc_op);
                }
                // Standard K-Epsilon closure models
                else
                {
                    // Diffusive coefficient
                    const auto diffusive_coeff_op = Teuchos::rcp(
                        new ClosureModel::IncompressibleKEpsilonDiffusivityCoefficient<
                            EvalType,
                            panzer::Traits>(
                            ir, fluid_prop, 1.0, 1.3, "BOUNDARY_"));
                    evaluators.push_back(diffusive_coeff_op);

                    // Turbulent eddy viscosity
                    const auto eddy_visc_op = Teuchos::rcp(
                        new ClosureModel::IncompressibleKEpsilonEddyViscosity<
                            EvalType,
                            panzer::Traits>(ir));
                    evaluators.push_back(eddy_visc_op);
                }
            }
            // K-Omega model family
            else if (std::string::npos
                     != _turbulence_model_name.find("K-Omega"))
            {
                // K-Omega turbulence variable names
                turb_field_names_vct.push_back("turb_kinetic_energy");
                turb_field_names_vct.push_back(
                    "turb_specific_dissipation_rate");

                // K-Omega closure models
                const auto eval_coeff = Teuchos::rcp(
                    new ClosureModel::IncompressibleKOmegaDiffusivityCoefficient<
                        EvalType,
                        panzer::Traits>(ir, fluid_prop, user_params));
                evaluators.push_back(eval_coeff);

                const auto eval_eddy = Teuchos::rcp(
                    new ClosureModel::IncompressibleKOmegaEddyViscosity<
                        EvalType,
                        panzer::Traits,
                        NumSpaceDim>(ir));
                evaluators.push_back(eval_eddy);
            }
            // WALE algebraic LES model
            else if (std::string::npos != _turbulence_model_name.find("WALE"))
            {
                // Sub-grid eddy viscosity
                const auto eddy_visc_op = Teuchos::rcp(
                    new ClosureModel::IncompressibleWALEEddyViscosity<
                        EvalType,
                        panzer::Traits,
                        NumSpaceDim>(ir, user_params));
                evaluators.push_back(eddy_visc_op);

                // Delta (element length) closure model
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
                        new ClosureModel::ElementLength<EvalType, panzer::Traits>(
                            ir, delta_prefix));

                    evaluators.push_back(eval_delta);
                }
                else if (delta_type == "MeasureElementLength")
                {
                    auto eval_delta = Teuchos::rcp(
                        new ClosureModel::MeasureElementLength<EvalType,
                                                               panzer::Traits>(
                            ir, delta_prefix));

                    evaluators.push_back(eval_delta);
                }
                else if (delta_type == "MetricTensorElementLength")
                {
                    auto eval_delta = Teuchos::rcp(
                        new ClosureModel::
                            MetricTensorElementLength<EvalType, panzer::Traits>(
                                ir, delta_prefix));

                    evaluators.push_back(eval_delta);
                }
                else if (delta_type == "SingularValueElementLength")
                {
                    const auto method = turb_params.get<std::string>(
                        "Element Length Method");

                    auto eval_delta = Teuchos::rcp(
                        new ClosureModel::SingularValueElementLength<
                            EvalType,
                            panzer::Traits>(ir, method, delta_prefix));

                    evaluators.push_back(eval_delta);
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

        // Loop over boundary conditions found in input file for each
        // turbulence model
        bool found_model = false;
        if (bc_params.isType<std::string>("Type")
            && (turb_field_names_vct.size() > 0))
        {
            const auto bc_type = bc_params.get<std::string>("Type");
            if (bc_type == "Fixed")
            {
                // Fixed boundary condition
                for (auto& variable_name : turb_field_names_vct)
                {
                    const auto state
                        = Teuchos::rcp(new TurbulenceFixed<EvalType, Traits>(
                            ir, bc_params, variable_name));
                    evaluators.push_back(state);
                }
                found_model = true;
            }

            else if (bc_type == "Extrapolate")
            {
                // Extrapolate boundary condition
                for (auto& variable_name : turb_field_names_vct)
                {
                    const auto state = Teuchos::rcp(
                        new TurbulenceExtrapolate<EvalType, Traits>(
                            ir, variable_name));
                    evaluators.push_back(state);
                }
                found_model = true;
            }

            else if (bc_type == "InletOutlet")
            {
                // Inlet/outlet boundary condition
                for (auto& variable_name : turb_field_names_vct)
                {
                    const auto state = Teuchos::rcp(
                        new TurbulenceInletOutlet<EvalType, Traits, NumSpaceDim>(
                            ir, bc_params, variable_name));
                    evaluators.push_back(state);
                }
                found_model = true;
            }

            else if (bc_type == "Symmetry")
            {
                // Symmetry boundary condition
                for (auto& variable_name : turb_field_names_vct)
                {
                    const auto state = Teuchos::rcp(
                        new TurbulenceSymmetry<EvalType, Traits>(
                            ir, variable_name));
                    evaluators.push_back(state);
                }
                found_model = true;
            }

            else if (bc_type == "K-Epsilon Wall Function")
            {
                // Wall functions for use with high Re K-Epsilon models
                const auto state = Teuchos::rcp(
                    new TurbulenceKEpsilonWallFunction<EvalType, Traits, NumSpaceDim>(
                        ir, bc_params, fluid_prop));
                evaluators.push_back(state);

                found_model = true;
            }

            else if (bc_type == "K-Omega Wall-Resolved")
            {
                const auto state = Teuchos::rcp(
                    new TurbulenceKOmegaWallResolved<EvalType, Traits>(
                        ir, bc_params));
                evaluators.push_back(state);

                found_model = true;
            }

            // Error message if model not found
            if (!found_model)
            {
                std::string msg = "\n\nBoundary state " + bc_type
                                  + " failed to build.\n";
                msg += "The boundary conditions implemented in VERTEX-CFD\n";
                msg += "for the turbulence model equations are:\n";
                msg += "Extrapolate,\n";
                msg += "Fixed,\n";
                msg += "InletOutlet,\n";
                msg += "K-Epsilon Wall Function,\n";
                msg += "K-Omega Wall-Resolved,\n";
                msg += "Symmetry\n";
                msg += "\n";
                throw std::runtime_error(msg);
            }
        }

        // Return vector of evaluators
        return evaluators;
    }
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // end VERTEXCFD_TURBULENCEBOUNDARYSTATE_FACTORY_HPP
