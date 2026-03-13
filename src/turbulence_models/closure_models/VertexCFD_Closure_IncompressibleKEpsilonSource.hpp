#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEKEPSILONSOURCE_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEKEPSILONSOURCE_HPP

/**
 * @file
 */

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
//--------------------------------//
/**
 * @class IncompressibleKEpsilonSource
 * @brief Source term evaluator for the standard incompressible
 * K‑\f$\varepsilon\f$ turbulence model.
 *
 * The class computes the production (\f$P_k\f$), destruction (\f$D_k\f$),
 * and total source (\f$S_k\f$) terms for the turbulent kinetic energy
 * equation, as well as the analogous terms (\f$P_{\varepsilon}\f$,
 * \f$D_{\varepsilon}\f$, \f$S_{\varepsilon}\f$) for the dissipation‑rate
 * equation.  The implementation follows the classic model constants \f$C_1\f$
 * and \f$C_2\f$ and uses a smoothed max function to ensure positivity of the
 * turbulent quantities.
 *
 * @tparam EvalType  Automatic‑differentiation type used by the evaluator.
 * @tparam Traits    Traits class providing the evaluation data type.
 * @tparam NumSpaceDim  Number of spatial dimensions (e.g. 2 or 3).
 */
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleKEpsilonSource
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    /**
     * @brief Constructor.
     *
     * Registers dependent and evaluated fields with the evaluator framework.
     *
     * @param[in] ir Integration rule providing the data layout for fields.
     */
    IncompressibleKEpsilonSource(const panzer::IntegrationRule& ir);

    /**
     * @brief Evaluate the source terms over the workset.
     *
     * This method launches a Kokkos parallel kernel that computes the
     * production, destruction and total source contributions for each cell
     * and quadrature point.
     *
     * @param[in] workset Evaluation data containing cell information.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /**
     * @brief Kokkos functor executed by each team.
     *
     * Computes the source terms for a single cell.  The implementation
     * follows the standard K‑\f$\varepsilon\f$ model:
     *
     * \f[
     *   P_k = 2 \nu_t \, S_{ij} S_{ij}, \qquad
     *   D_k = -\varepsilon, \qquad
     *   S_k = P_k + D_k,
     * \f],
     *
     * @param[in] team Kokkos team policy member representing a cell.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    /** @brief Turbulent eddy viscosity \f$\nu_t\f$ (input field). */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu_t;

    /** @brief Turbulent kinetic energy \f$k\f$ (input field). */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_kinetic_energy;

    /** @brief Turbulent dissipation rate \f$\varepsilon\f$ (input field). */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_dissipation_rate;

    /** @brief Gradient of the velocity field \f$\partial u_i / \partial x_j\f$
     *  (input vector field).  The outer dimension corresponds to the spatial
     *  component \f$i\f$, the inner dimension to the derivative direction
     * \f$j\f$.
     */
    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_velocity;

    /** @brief Model constant \f$C_1\f$ used in the production term of the
     *  dissipation‑rate equation. */
    double _C_1;

    /** @brief Model constant \f$C_2\f$ used in the destruction term of the
     *  dissipation‑rate equation. */
    double _C_2;

  public:
    /** @brief Total source term for the turbulent kinetic energy equation
     *  (\f$S_k = P_k + D_k\f$). */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k_source;

    /** @brief Production term for the turbulent kinetic energy equation
     * (\f$P_k\f$). */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k_prod;

    /** @brief Destruction (or dissipation) term for the turbulent kinetic
     * energy equation (\f$D_k\f$). */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k_dest;

    /** @brief Total source term for the turbulent dissipation‑rate equation
     *  (\f$S_\varepsilon = P_\varepsilon + D_\varepsilon\f$). */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _e_source;

    /** @brief Production term for the turbulent dissipation‑rate equation
     *  (\f$P_\varepsilon\f$). */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _e_prod;

    /** @brief Destruction term for the turbulent dissipation‑rate equation
     *  (\f$D_\varepsilon\f$). */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _e_dest;
};

//------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_INCOMPRESSIBLEKEPSILONSOURCE_HPP
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLEKEPSILONSOURCE_HPP