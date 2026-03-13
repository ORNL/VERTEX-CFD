#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEKOMEGASOURCE_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEKOMEGASOURCE_HPP

/**
 * @file IncompressibleKOmegaSource.hpp
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
 * @defgroup ClosureModel Turbulence Closure Models
 * @brief Collection of turbulence closure model evaluators.
 * @{
 */

/**
 * @class IncompressibleKOmegaSource
 * @brief Source term evaluator for the incompressible Wilcox (2006) K‑Omega
 * turbulence model.
 *
 * The class computes production, destruction and cross‑diffusion contributions
 * for the turbulent kinetic energy \f$k\f$ and the specific dissipation rate
 * \f$\omega\f$ and provides them as fields for use in the residual. The
 * implementation supports an optional limiter on the production term.
 *
 * @tparam EvalType  Evaluation type providing the scalar type (e.g. Residual,
 * Jacobian).
 * @tparam Traits    Traits class required by the Phalanx evaluator framework.
 * @tparam NumSpaceDim Number of spatial dimensions (e.g. 2 or 3).
 */
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleKOmegaSource
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    /**
     * @brief Scalar type used for the evaluation (e.g. double, Sacado::Fad).
     */
    using scalar_type = typename EvalType::ScalarT;

    /**
     * @brief Number of spatial dimensions (compile‑time constant).
     */
    static constexpr int num_space_dim = NumSpaceDim;

    /**
     * @brief Constructor.
     * @param ir Integration rule providing cell topology and quadrature
     * information.
     * @param turb_params Parameter list containing model coefficients
     * (beta_star, gamma, etc.).
     */
    IncompressibleKOmegaSource(const panzer::IntegrationRule& ir,
                               const Teuchos::ParameterList& turb_params);

    /**
     * @brief Evaluate the source fields over the workset.
     * @param workset Evaluation data containing cell and point information.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /**
     * @brief Functor invoked by Kokkos parallel_for to compute source terms.
     * @param team Team policy member representing a cell.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    /**
     * @var _nu_t Turbulent eddy viscosity \f$\nu_t\f$ (input field).
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu_t;

    /**
     * @var _turb_kinetic_energy Turbulent kinetic energy \f$k\f$ (input
     * field).
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_kinetic_energy;

    /**
     * @var _turb_specific_dissipation_rate Specific dissipation rate
     * \f$\omega\f$ (input field).
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_specific_dissipation_rate;

    /**
     * @var _grad_turb_kinetic_energy Gradient of \f$k\f$ (input field).
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_turb_kinetic_energy;

    /**
     * @var _grad_turb_specific_dissipation_rate Gradient of \f$\omega\f$
     * (input field).
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_turb_specific_dissipation_rate;

    /**
     * @var _grad_velocity Gradient of the velocity vector \f$\nabla
     * \mathbf{u}\f$ (input field).
     */
    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_velocity;

    /**
     * @var _beta_star Model constant \f$\beta^{*}=0.09\f$ (default).
     */
    double _beta_star;

    /**
     * @var _gamma Model constant \f$\gamma=0.52\f$ (default).
     */
    double _gamma;

    /**
     * @var _beta_0 Model constant \f$\beta_{0}=0.0708\f$ (default).
     */
    double _beta_0;

    /**
     * @var _sigma_d Model constant \f$\sigma_{d}=0.125\f$ (default).
     */
    double _sigma_d;

    /**
     * @var _limit_production Flag to enable limiter on production term.
     */
    bool _limit_production;

  public:
    // Output fields – source, production, destruction and cross‑diffusion
    // terms.

    /**
     * @var _k_source Source term for the \f$k\f$ equation.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k_source;

    /**
     * @var _k_prod Production term for the \f$k\f$ equation.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k_prod;

    /**
     * @var _k_dest Destruction term for the \f$k\f$ equation.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k_dest;

    /**
     * @var _w_source Source term for the \f$\omega\f$ equation.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _w_source;

    /**
     * @var _w_prod Production term for the \f$\omega\f$ equation.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _w_prod;

    /**
     * @var _w_dest Destruction term for the \f$\omega\f$ equation.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _w_dest;

    /**
     * @var _w_cross Cross‑diffusion term for the \f$\omega\f$ equation.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _w_cross;
};

/** @} */ // end of ClosureModel

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_INCOMPRESSIBLEKOMEGASOURCE_HPP