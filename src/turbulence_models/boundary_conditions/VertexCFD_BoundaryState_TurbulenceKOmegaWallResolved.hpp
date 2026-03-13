#ifndef VERTEXCFD_BOUNDARYSTATE_TURBULENCEKOMEGAWALLRESOLVED_HPP
#define VERTEXCFD_BOUNDARYSTATE_TURBULENCEKOMEGAWALLRESOLVED_HPP

/**
 * @file TurbulenceKOmegaWallResolved.hpp
 */

/**
 * @defgroup BoundaryCondition Boundary Condition
 * @brief Boundary condition evaluators.
 * @{
 */

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <string>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------//
/**
 * @class TurbulenceKOmegaWallResolved
 * @brief Wall‑resolved K‑Omega turbulence boundary condition.
 *
 * This evaluator provides the boundary values and gradients for the
 * turbulent kinetic energy \f$k\f$ and specific dissipation rate \f$\omega\f$
 * on wall faces.  It supports optional ramping of the wall value of
 * \f$\omega\f$ from an initial value to a target value over a prescribed
 * time interval.
 *
 * @tparam EvalType  Evaluation type (e.g., Residual, Jacobian).
 * @tparam Traits    Traits class providing the evaluation data type.
 */
template<class EvalType, class Traits>
class TurbulenceKOmegaWallResolved
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    /**
     * @brief Construct the evaluator.
     *
     * @param[in] ir        Integration rule defining the evaluation points.
     * @param[in] bc_params Parameter list containing wall‑model settings:
     *                      - "Omega Wall Value" (required)
     *                      - "Omega Wall Initial Value" (optional)
     *                      - "Omega Ramp Time" (optional)
     */
    TurbulenceKOmegaWallResolved(const panzer::IntegrationRule& ir,
                                 const Teuchos::ParameterList& bc_params);

    /**
     * @brief Evaluate the boundary fields for a workset.
     *
     * This method extracts the current simulation time from the workset,
     * creates a Kokkos team policy, and launches the parallel kernel that
     * fills the boundary fields.
     *
     * @param[in] workset Evaluation data for the current workset.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /**
     * @brief Parallel kernel executed by Kokkos.
     *
     * For each cell and integration point the kernel copies the interior
     * turbulent kinetic energy, applies the (possibly ramped) wall value of
     * \f$\omega\f$, and projects the gradient of \f$k\f$ onto the wall tangent
     * plane by removing the normal component.
     *
     * @param[in] team Kokkos team member representing a cell.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    /** Boundary value of turbulent kinetic energy \f$k\f$. */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_k;
    /** Boundary value of specific dissipation rate \f$\omega\f$. */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_w;

    /** Boundary gradient of \f$k\f$ (tangential components). */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _boundary_grad_k;
    /** Boundary gradient of \f$\omega\f$ (tangential components). */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _boundary_grad_w;

  private:
    /** Interior turbulent kinetic energy \f$k\f$. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _k;
    /** Interior gradient of \f$k\f$. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_k;
    /** Interior gradient of \f$\omega\f$. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_w;
    /** Outward unit normal vectors at the boundary. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _normals;

    /** Current simulation time (set in evaluateFields). */
    double _time;
    /** Target wall value of \f$\omega\f$. */
    double _omega_wall;
    /** Initial wall value of \f$\omega\f$ for ramping. */
    double _omega_wall_init;
    /** Duration of the ramp from initial to target value. */
    double _omega_ramp_time;
    /** Number of spatial dimensions for gradient fields. */
    int _num_grad_dim;
};

//-----------------------------------------------//

/** @} */ // end of BoundaryCondition

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_TURBULENCEKOMEGAWALLRESOLVED_HPP