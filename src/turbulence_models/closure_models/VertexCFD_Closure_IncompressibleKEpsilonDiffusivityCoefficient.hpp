#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEKEPSILONDIFFUSIVITYCOEFFICIENT_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEKEPSILONDIFFUSIVITYCOEFFICIENT_HPP

/**
 * @file IncompressibleKEpsilonDiffusivityCoefficient.hpp
 */

/**
 * @defgroup ClosureModel Closure Model
 * @brief Diffusion coefficients for standard K‑Epsilon turbulence model.
 * @{
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

/**
 * @class IncompressibleKEpsilonDiffusivityCoefficient
 * @brief Evaluator that computes the diffusivity coefficients for the
 * incompressible K‑Epsilon turbulence model.
 *
 * The diffusivity for the turbulent kinetic energy \f$k\f$ and the turbulent
 * dissipation rate \f$\varepsilon\f$ are given by

\f[
 D_k = \nu + \frac{\nu_t}{\sigma_k}, \qquad D_\varepsilon = \nu +
\frac{\nu_t}{\sigma_\varepsilon} \f]

where \f$\nu\f$ is the molecular kinematic viscosity and \f$\nu_t\f$ is the
turbulent eddy viscosity.
 *
 * @tparam EvalType Evaluation type (e.g., double or AD type).
 * @tparam Traits  Traits type.
 */
template<class EvalType, class Traits>
class IncompressibleKEpsilonDiffusivityCoefficient
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    /** @typedef scalar_type
     *  The scalar type used for the evaluation (e.g., double or AD type).
     */
    using scalar_type = typename EvalType::ScalarT;

    /**
     * @brief Constructor registers dependent and evaluated fields and stores
     * model parameters.
     *
     * @param ir        Integration rule providing cell topology, quadrature
     * points and spatial dimension.
     * @param sigma_k   Turbulent Prandtl number for the kinetic‑energy
     * equation (default = 1.0).
     * @param sigma_e   Turbulent Prandtl number for the dissipation‑rate
     * equation (default = 1.3).
     * @param field_prefix Optional prefix added to the turbulent eddy
     * viscosity field name.
     */
    IncompressibleKEpsilonDiffusivityCoefficient(
        const panzer::IntegrationRule& ir,
        const double sigma_k = 1.0,
        const double sigma_e = 1.3,
        const std::string field_prefix = "");

    /**
     * @brief Compute diffusivity fields for all cells in the supplied workset.
     *
     * @param workset Evaluation data containing the number of cells and other
     * context.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /**
     * @brief Kokkos functor executed by a team policy; computes diffusivity at
     * each quadrature point.
     *
     * @param team Kokkos team member representing a single cell.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    /** @var _nu
     *  Molecular kinematic viscosity (input field).
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu;

    /** @var _nu_t
     *  Turbulent eddy viscosity (input field).
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu_t;

    /** @var _sigma_k
     *  Turbulent Prandtl number for \f$k\f$.
     */
    double _sigma_k;

    /** @var _sigma_e
     *  Turbulent Prandtl number for \f$\varepsilon\f$.
     */
    double _sigma_e;

    /** @var _num_grad_dim
     *  Spatial dimension of the problem (e.g., 2 or 3).
     */
    int _num_grad_dim;

  public:
    /** @var _diffusivity_var_k
     *  Diffusivity coefficient for turbulent kinetic energy \f$k\f$ (output
     * field).
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _diffusivity_var_k;

    /** @var _diffusivity_var_e
     *  Diffusivity coefficient for turbulent dissipation rate
     * \f$\varepsilon\f$ (output field).
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _diffusivity_var_e;
};

/** @} */ // end of ClosureModel group

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_INCOMPRESSIBLEKEPSILONDIFFUSIVITYCOEFFICIENT_HPP