#ifndef VERTEXCFD_TURBULENCECLOSUREMODELFACTORY_HPP
#define VERTEXCFD_TURBULENCECLOSUREMODELFACTORY_HPP

/**
 * @file TurbulenceClosureModelFactory.hpp
 */

/**
 * @defgroup VertexCFD_ClosureModel Turbulence Closure Model
 * @brief Turbulence closure model utilities.
 *
 * This module contains factories and helpers for constructing turbulence
 * closure model evaluators used within the VertexCFD framework.
 * @{
 */

#include <Panzer_GlobalData.hpp>
#include <Panzer_Traits.hpp>

#include <Phalanx_Evaluator.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

namespace VertexCFD
{
namespace ClosureModel
{

/**
 * @class TurbulenceFactory
 * @brief Factory class that creates turbulence closure model evaluators for a
 * given turbulence model.
 *
 * @tparam EvalType   Evaluation type used by Phalanx (e.g., Residual,
 * Jacobian, Tangent). Determines the scalar type of the generated evaluators.
 * @tparam NumSpaceDim Number of spatial dimensions of the simulation.
 *
 * The factory examines the supplied turbulence model name and parameters, then
 * creates the appropriate set of Phalanx evaluators (e.g., time‑derivative,
 * convective flux, diffusion flux, SUPG terms, and model‑specific diffusivity,
 * source, and eddy‑viscosity closures). The created evaluators are appended to
 * the provided container.
 */
template<class EvalType, int NumSpaceDim>
class TurbulenceFactory
{
  public:
    /**
     * @brief Number of spatial dimensions (alias for the template parameter).
     */
    static constexpr int num_space_dim = NumSpaceDim;

    /**
     * @brief Construct closure model evaluators for the specified turbulence
     * model.
     *
     * This method examines the supplied turbulence model name and parameters,
     * then creates the appropriate set of Phalanx evaluators (e.g.,
     * time‑derivative, convective flux, diffusion flux, SUPG terms, and
     * model‑specific diffusivity, source, and eddy‑viscosity closures). The
     * created evaluators are appended to the provided container.
     *
     * @param[in]  ir          Integration rule providing quadrature points and
     * weights for the element.
     * @param[in]  global_data Global data object from Panzer containing mesh
     * and solution information.
     * @param[in]  turb_params Parameter list describing the turbulence model,
     * its coefficients, and optional SUPG settings.
     * @param[out] evaluators  Vector of RCPs to Phalanx evaluators; the newly
     * created evaluators are added to this container.
     */
    void buildClosureModel(
        const Teuchos::RCP<panzer::IntegrationRule>& ir,
        const Teuchos::RCP<panzer::GlobalData>& global_data,
        const Teuchos::ParameterList& turb_params,
        Teuchos::RCP<std::vector<Teuchos::RCP<PHX::Evaluator<panzer::Traits>>>>
            evaluators);
};

/// @} // end of VertexCFD_ClosureModel group

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_TURBULENCECLOSUREMODELFACTORY_HPP