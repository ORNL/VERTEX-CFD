#ifndef VERTEXCFD_BOUNDARYCONDITION_CONDUCTIONBOUNDARYFLUX_HPP
#define VERTEXCFD_BOUNDARYCONDITION_CONDUCTIONBOUNDARYFLUX_HPP

/**
 * @file VertexCFD_BCStrategy_ConductionBoundaryFlux.hpp
 * @brief Implements a Neumann (flux) boundary condition for thermal
 * conduction.
 *
 * This header defines the @c ConductionBoundaryFlux evaluator, which
 * constructs, registers, and evaluates the conduction flux on a side set
 * following the Panzer/Phalanx evaluator pattern.
 */

/**
 * @defgroup BoundaryCondition Boundary Condition Module
 * @brief Boundary‑condition strategies for VertexCFD.
 * @{
 */

/**
 * @defgroup ConductionBoundaryFlux Conduction Boundary Flux
 * @brief Neumann (flux) boundary condition for thermal conduction.
 * @}
 */

#include "boundary_conditions/VertexCFD_BCStrategy_BoundaryFluxBase.hpp"

#include <Panzer_BCStrategy.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_GlobalDataAcceptor_DefaultImpl.hpp>
#include <Panzer_PhysicsBlock.hpp>
#include <Panzer_Traits.hpp>

#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_MDField.hpp>

#include <Teuchos_RCP.hpp>

#include <unordered_map>

namespace VertexCFD
{
namespace BoundaryCondition
{

/**
 * @class ConductionBoundaryFlux
 * @brief Implements a Neumann (flux) boundary condition for thermal
 * conduction.
 *
 * This class derives from @c BoundaryFluxBase and provides the machinery to
 * construct, register, and evaluate the conduction flux on a side set.  The
 * implementation follows the standard Panzer/Phalanx evaluator pattern.
 *
 * @tparam EvalType   Evaluation type (e.g., Residual, Jacobian) used by
 * Panzer.
 * @tparam NumSpaceDim Spatial dimension of the problem (e.g., 2 or 3).
 */
template<class EvalType, int NumSpaceDim>
class ConductionBoundaryFlux : public BoundaryFluxBase<EvalType, NumSpaceDim>
{
  public:
    /**
     * @brief Constructor.
     *
     * @param[in] bc          Boundary condition descriptor from Panzer.
     * @param[in] global_data Shared global data required by Panzer evaluators.
     *
     * The constructor stores the BC information and registers the required
     * fields with the base class.
     */
    ConductionBoundaryFlux(const panzer::BC& bc,
                           const Teuchos::RCP<panzer::GlobalData>& global_data);

    /** @brief Number of spatial dimensions (compile‑time constant). */
    static constexpr int num_space_dim = NumSpaceDim;

    /**
     * @brief Set up the evaluator using side physics block information.
     *
     * @param[in] side_pb   Physics block describing the side set.
     * @param[in] user_data User‑provided parameter list containing additional
     * data needed for setup.
     *
     * This method extracts required data from the physics block and stores
     * any internal mappings needed for later evaluator construction.
     */
    void setup(const panzer::PhysicsBlock& side_pb,
               const Teuchos::ParameterList& user_data) override;

    /**
     * @brief Build and register field evaluators for the conduction flux.
     *
     * @param[in,out] fm        Field manager to which evaluators are added.
     * @param[in]     side_pb   Physics block describing the side set.
     * @param[in]     factory   Closure model factory for constructing model
     *                          evaluators.
     * @param[in]     models    Parameter list containing model specifications.
     * @param[in]     user_data Additional user data required for evaluator
     *                          construction.
     *
     * The method creates evaluators for the flux, temperature gradient,
     * and material conductivity, then registers them with @c fm.
     */
    void buildAndRegisterEvaluators(
        PHX::FieldManager<panzer::Traits>& fm,
        const panzer::PhysicsBlock& side_pb,
        const panzer::ClosureModelFactory_TemplateManager<panzer::Traits>& factory,
        const Teuchos::ParameterList& models,
        const Teuchos::ParameterList& user_data) const override;

    /**
     * @brief Build and register scatter evaluators for the boundary flux.
     *
     * @param[in,out] fm        Field manager to which scatter evaluators are
     * added.
     * @param[in]     side_pb   Physics block describing the side set.
     * @param[in]     lof       Linear object factory for constructing scatter
     *                          operations.
     * @param[in]     user_data Additional user data required for scatter
     *                          evaluator construction.
     *
     * Scatter evaluators write the computed flux contributions into the global
     * linear system.
     */
    void buildAndRegisterScatterEvaluators(
        PHX::FieldManager<panzer::Traits>& fm,
        const panzer::PhysicsBlock& side_pb,
        const panzer::LinearObjFactory<panzer::Traits>& lof,
        const Teuchos::ParameterList& user_data) const override;

  private:
    /** @brief Mapping from equation name to associated DOF condition name. */
    std::unordered_map<std::string, std::string> _equ_dof_cond_pair;
};

/** @} */ // end of ConductionBoundaryFlux group
/** @} */ // end of BoundaryCondition group

} // namespace BoundaryCondition
} // namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYCONDITION_CONDUCTIONBOUNDARYFLUX_HPP
