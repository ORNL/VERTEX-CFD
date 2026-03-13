#ifndef VERTEXCFD_CONDUCTIONBOUNDARYSTATE_FACTORY_HPP
#define VERTEXCFD_CONDUCTIONBOUNDARYSTATE_FACTORY_HPP

/**
 * @file VertexCFD_ConductionBoundaryState_Factory.hpp
 * @brief Factory for creating conduction‑equation boundary‑state evaluators.
 *
 * This header defines a templated factory class that inspects user‑provided
 * parameter lists and material‑property closure models to instantiate the
 * appropriate PHX::Evaluator objects for conduction boundary conditions.
 */

/**
 * @defgroup ConductionBoundaryStateFactory Conduction Boundary State Factory
 * @brief Doxygen group for the conduction boundary‑state factory
 * implementation.
 *
 * The factory creates evaluators for adiabatic walls, temperature
 * extrapolation, and fixed temperature boundary conditions, and registers the
 * required thermal conductivity closure model.
 * @{
 */

#include "closure_models/VertexCFD_Closure_ConstantScalarField.hpp"

#include "conduction/boundary_conditions/VertexCFD_BoundaryState_AdiabaticWall.hpp"
#include "conduction/closure_models/VertexCFD_Closure_ConductionThermalConductivity.hpp"

#include "boundary_conditions/VertexCFD_BoundaryState_VariableExtrapolate.hpp"
#include "boundary_conditions/VertexCFD_BoundaryState_VariableFixed.hpp"
#include "boundary_conditions/VertexCFD_BoundaryState_VariableFlux.hpp"

#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{
/**
 * @class ConductionBoundaryStateFactory
 * @brief Factory for creating boundary‑state evaluators for the conduction
 * equation.
 *
 * This factory examines the supplied boundary‑condition parameters and the
 * material‑property closure model to instantiate the appropriate
 * PHX::Evaluator objects.  The created evaluators enforce the specified
 * boundary state (e.g., adiabatic wall, extrapolated temperature, or fixed
 * temperature) and provide the required material property (thermal
 * conductivity) on the boundary.
 *
 * @tparam EvalType   Evaluation type (e.g., Residual, Jacobian) used by the
 *                    evaluators.
 * @tparam Traits     Traits class defining the evaluation infrastructure.
 * @tparam NumSpaceDim Spatial dimension of the problem (e.g., 2 or 3).
 */
template<class EvalType, class Traits, int NumSpaceDim>
class ConductionBoundaryStateFactory
{
  public:
    /** @brief Number of spatial dimensions (alias for the template parameter).
     */
    static constexpr int num_space_dim = NumSpaceDim;

    /**
     * @brief Create a set of boundary‑state evaluators for a given integration
     * rule.
     *
     * The method reads the material‑property list from the supplied closure
     * model and registers the appropriate thermal‑conductivity evaluator.  It
     * then examines the boundary‑condition type and creates the corresponding
     * state evaluator.
     *
     * @param[in] ir            Integration rule providing quadrature
     * information.
     * @param[in] bc_params     Parameter list describing the boundary
     * condition.
     * @param[in] closure_model Parameter list containing closure models,
     * including a sublist "Material Properties" that defines the material
     * property type.
     *
     * @return Vector of reference‑counted pointers to the created evaluators.
     *
     * @throws std::runtime_error if the boundary‑condition type is not
     * recognized.
     */
    static std::vector<Teuchos::RCP<PHX::Evaluator<Traits>>>
    create(const panzer::IntegrationRule& ir,
           const Teuchos::ParameterList& bc_params,
           const Teuchos::ParameterList& closure_model)
    {
        // Evaluator vector to return
        std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>> evaluators;

        // Get material property. Note: it is assumed that the material
        // property is listed under 'Material Properties' in the closure model
        // of the input file.
        const auto mat_prop_list
            = closure_model.sublist("Material Properties");

        // Register thermal conductivity to be used on the boundary based on
        // the material property type set in the input file
        const auto closure_type = mat_prop_list.get<std::string>("Type");
        if (closure_type == "ConstantMaterialProperties")
        {
            const auto mat_prop = Teuchos::rcp(
                new ClosureModel::ConductionThermalConductivity<EvalType,
                                                                panzer::Traits>(
                    ir, mat_prop_list));
            evaluators.push_back(mat_prop);
        }

        // Loop over boundary conditions found in input file
        Teuchos::RCP<PHX::Evaluator<Traits>> state;
        bool found_model = false;
        if (bc_params.isType<std::string>("Type"))
        {
            const auto bc_type = bc_params.get<std::string>("Type");

            if (bc_type == "AdiabaticWall")
            {
                state = Teuchos::rcp(new AdiabaticWall<EvalType, Traits>(ir));
                evaluators.push_back(state);
                found_model = true;
            }

            if (bc_type == "HeatFlux")
            {
                state = Teuchos::rcp(new VariableFlux<EvalType, Traits>(
                    ir, bc_params, "temperature", "thermal_conductivity"));
                evaluators.push_back(state);
                found_model = true;
            }

            if (bc_type == "Extrapolate")
            {
                state = Teuchos::rcp(new VariableExtrapolate<EvalType, Traits>(
                    ir, "temperature"));
                evaluators.push_back(state);
                found_model = true;
            }

            if (bc_type == "Fixed")
            {
                state = Teuchos::rcp(new VariableFixed<EvalType, Traits>(
                    ir, bc_params, "temperature"));
                evaluators.push_back(state);
                found_model = true;
            }

            // Error message if model not found
            if (!found_model)
            {
                std::string msg = "\n\nBoundary state " + bc_type
                                  + " failed to build.\n";
                msg += "The boundary conditions implemented in VERTEX-CFD\n";
                msg += "for the conduction equation are:\n";
                msg += "AdiabaticWall,\n";
                msg += "Extrapolate,\n";
                msg += "Fixed,\n";
                msg += "HeatFlux,\n";
                msg += "\n";
                throw std::runtime_error(msg);
            }
        }

        // Return evaluators
        return evaluators;
    }
};

//--------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

/** @} */ // end of ConductionBoundaryStateFactory group

#endif // VERTEXCFD_CONDUCTIONBOUNDARYSTATE_FACTORY_HPP