#ifndef VERTEXCFD_CLOSURE_CONDUCTIONERRORNORMS_HPP
#define VERTEXCFD_CLOSURE_CONDUCTIONERRORNORMS_HPP

/**
 * @file VertexCFD_Closure_ConductionErrorNorms.hpp
 * @brief Declaration of the ConductionErrorNorms evaluator.
 *
 * Provides the evaluator that computes L1 and L2 error norms for
 * conduction problems.
 */

/** @defgroup VertexCFD_Closure_ConductionErrorNorms Conduction Error Norms
 * @brief Doxygen group for the conduction error norm closure model.
 * @{
 */

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Teuchos_ParameterList.hpp>

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
/** @class ConductionErrorNorms
 * @brief Compute error norms between exact and numerical conduction solutions.
 *
 * This evaluator computes the L1 and L2 error norms for the continuity,
 * momentum, and energy equations of a conduction problem. The errors are
 * evaluated at each integration point of the workset and stored in MDFields
 * that can be inspected by downstream diagnostics or unit tests.
 *
 * @tparam EvalType   Type providing the scalar type (e.g., Residual,
 * Jacobian).
 * @tparam Traits     PHX traits class defining the evaluation data type.
 * @tparam NumSpaceDim Spatial dimension of the problem (e.g., 2 or 3).
 */
template<class EvalType, class Traits, int NumSpaceDim>
class ConductionErrorNorms : public panzer::EvaluatorWithBaseImpl<Traits>,
                             public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    /** @brief Alias for the scalar type used by the evaluation type. */
    using scalar_type = typename EvalType::ScalarT;

    /** @brief Number of spatial dimensions (compile‑time constant). */
    static constexpr int num_space_dim = NumSpaceDim;

    /** @brief Constructor.
     *
     *  Registers the fields that will be evaluated and the dependent fields
     *  required for the error computation.
     *
     *  @param[in] ir Integration rule providing the data layout for fields.
     */
    ConductionErrorNorms(const panzer::IntegrationRule& ir);

    /** @brief Evaluate the error norm fields over the workset.
     *
     *  This method is called by the PHX framework during the evaluation
     *  phase.  It launches a Kokkos parallel kernel that computes the L1 and
     *  L2 errors for each cell, integration point, and spatial component.
     *
     *  @param[in] workset Evaluation data containing cell topology and
     *                     integration point information.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /** @brief Kokkos functor invoked for each team of threads.
     *
     *  The implementation performs the actual error norm calculations.
     *
     *  @param[in] team Team policy member representing a group of threads.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    /** @brief L1 error of the continuity equation (scalar field). */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _L1_error_continuity;

    /** @brief L1 error of the momentum equations (vector field).
     *
     *  The array size equals the spatial dimension; each entry stores the
     *  error component for one coordinate direction.
     */
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _L1_error_momentum;

    /** @brief L1 error of the energy equation (scalar field). */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _L1_error_energy;

    /** @brief L2 error of the continuity equation (scalar field). */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _L2_error_continuity;

    /** @brief L2 error of the momentum equations (vector field). */
    Kokkos::Array<PHX::MDField<scalar_type, panzer::Cell, panzer::Point>,
                  num_space_dim>
        _L2_error_momentum;

    /** @brief L2 error of the energy equation (scalar field). */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _L2_error_energy;

    /** @brief Cell volume at each integration point (used for weighting). */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _volume;

  private:
    /** @brief Exact temperature field (read‑only). */
    PHX::MDField<const double, panzer::Cell, panzer::Point> _exact_temperature;

    /** @brief Numerical temperature field (read‑only). */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _temperature;
};

} // namespace ClosureModel
} // namespace VertexCFD

/** @} */ // end of VertexCFD_Closure_ConductionErrorNorms

#include "VertexCFD_Closure_ConductionErrorNorms_impl.hpp"

#endif // VERTEXCFD_CLOSURE_CONDUCTIONERRORNORMS_HPP