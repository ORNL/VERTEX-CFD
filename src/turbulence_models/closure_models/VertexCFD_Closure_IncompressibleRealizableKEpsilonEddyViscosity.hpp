#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEREALIZABLEKEPSILONEDDYVISCOSITY_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEREALIZABLEKEPSILONEDDYVISCOSITY_HPP

/** @file IncompressibleRealizableKEpsilonEddyViscosity.hpp */

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace ClosureModel
{
/** @defgroup ClosureModel Closure Model
 *  @brief Turbulence closure models.
 *  @{
 */

/** @class IncompressibleRealizableKEpsilonEddyViscosity
 *  @brief Eddy‑viscosity closure model for the incompressible realizable
 * K‑Epsilon formulation.
 *
 *  This class computes the turbulent eddy viscosity \f$\nu_t\f$ based on the
 *  turbulent kinetic energy \f$k\f$ and its dissipation rate
 * \f$\varepsilon\f$.
 *
 *  @tparam EvalType  Evaluation type providing the scalar type.
 *  @tparam Traits    Traits class required by the Phalanx evaluator framework.
 *  @tparam NumSpaceDim Number of spatial dimensions (e.g., 2 or 3).
 */
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleRealizableKEpsilonEddyViscosity
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    /** @brief Constructor.
     *
     *  Registers dependent fields (turbulent kinetic energy, dissipation rate,
     *  and velocity gradient) and the evaluated eddy‑viscosity field.
     *
     *  @param ir Integration rule providing the data layout for fields.
     */
    IncompressibleRealizableKEpsilonEddyViscosity(
        const panzer::IntegrationRule& ir);

    /** @brief Evaluate the eddy‑viscosity field over the workset.
     *
     *  This method launches a Kokkos team‑parallel kernel that computes
     *  \f$\nu_t\f$ at each quadrature point.
     *
     *  @param workset Evaluation data containing cell information.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /** @brief Kokkos functor operator implementing the per‑cell, per‑point
     * computation.
     *
     *  The algorithm follows the realizable K‑Epsilon formulation:
     *  - Compute the symmetric and skew‑symmetric parts of the velocity
     * gradient.
     *  - Evaluate the invariants \f$S^2\f$, \f$\Omega^2\f$, and the third
     * invariant \f$W\f$.
     *  - Determine the model constants \f$C_{\nu}\f$ and finally \f$\nu_t\f$.
     *
     *  @param team Kokkos team policy member representing a cell.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    // Dependent fields
    // -----------------------------------------

    /** @brief Turbulent kinetic energy \f$k\f$ (scalar) at each cell and
     * quadrature point. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_kinetic_energy;

    /** @brief Turbulent dissipation rate \f$\varepsilon\f$ (scalar) at each
     * cell and quadrature point. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_dissipation_rate;

    /** @brief Velocity gradient \f$\partial u_i / \partial x_j\f$ (tensor) at
     * each cell and point. */
    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_velocity;

    /** @brief Model constant \f$A_0 = 4.0\f$ used in the realizable
     * formulation. */
    double _A_0;

  public:
    // Evaluated field
    // -----------------------------------------

    /** @brief Computed turbulent eddy viscosity \f$\nu_t\f$ (scalar) at each
     * cell and point. */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _nu_t;
};

/** @} */ // end of ClosureModel

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_INCOMPRESSIBLEREALIZABLEKEPSILONEDDYVISCOSITY_HPP