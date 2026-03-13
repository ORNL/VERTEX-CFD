#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLESSTSOURCE_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLESSTSOURCE_HPP

/**
 * @file IncompressibleSSTSource.hpp
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
 * @defgroup ClosureModel Closure Model
 * @brief Turbulence model closure implementations.
 * @{
 */

/**
 * @class IncompressibleSSTSource
 * @brief Source term for Menter's SST K‑Omega turbulence model.
 *
 * This evaluator computes the production, destruction, and cross‑diffusion
 * contributions to the turbulent kinetic energy (\f$k\f$) and specific
 * dissipation rate (\f$\omega\f$) equations.  The implementation follows the
 * formulation described in Menter, *J. Fluid Mech.*, 1994.
 *
 * @tparam EvalType Evaluation type (e.g., Residual, Jacobian).
 * @tparam Traits   Traits class for the evaluator.
 * @tparam NumSpaceDim Number of spatial dimensions.
 */
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleSSTSource : public panzer::EvaluatorWithBaseImpl<Traits>,
                                public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    /**
     * @brief Constructor.
     * @param ir Integration rule providing cell topology and quadrature
     * points.
     * @param turb_params Parameter list containing turbulence model
     * coefficients.
     */
    IncompressibleSSTSource(const panzer::IntegrationRule& ir,
                            const Teuchos::ParameterList& turb_params);

    /**
     * @brief Evaluate the source terms for each cell in the workset.
     * @param workset Evaluation data containing cell information.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /**
     * @brief Kokkos functor executed in parallel over cells and quadrature
     * points.
     * @param team Kokkos team member representing a cell.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    // Dependent fields
    // ---------------------------------------------------------

    /** @brief Turbulent eddy viscosity, \f$\nu_t\f$. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu_t;
    /** @brief Turbulent kinetic energy, \f$k\f$. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_kinetic_energy;
    /** @brief Specific dissipation rate, \f$\omega\f$. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_specific_dissipation_rate;
    /** @brief Gradient of turbulent kinetic energy, \f$\nabla k\f$. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_turb_kinetic_energy;
    /** @brief Gradient of specific dissipation rate, \f$\nabla \omega\f$. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_turb_specific_dissipation_rate;
    /** @brief Blending function \f$F_1\f$ used in the SST model. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _sst_blending_function;

    /** @brief Gradient of the velocity field, \f$\nabla \mathbf{u}\f$. */
    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_velocity;

    // Model constants
    // ---------------------------------------------------------

    /** @brief Model constant \f$\beta^* = 0.09\f$ (default). */
    double _beta_star;
    /** @brief von Kármán constant \f$\kappa = 0.41\f$ (default). */
    double _kappa;
    /** @brief Model constant \f$\beta_1\f$. */
    double _beta_1;
    /** @brief Model constant \f$\beta_2\f$. */
    double _beta_2;
    /** @brief Model constant \f$\sigma_{w1}\f$. */
    double _sigma_w1;
    /** @brief Model constant \f$\sigma_{w2}\f$. */
    double _sigma_w2;
    /** @brief Derived constant \f$\gamma_1 = \beta_1/\beta^* -
     * \sigma_{w1}\kappa^2/\sqrt{\beta^*}\f$. */
    double _gamma_1;
    /** @brief Derived constant \f$\gamma_2 = \beta_2/\beta^* -
     * \sigma_{w2}\kappa^2/\sqrt{\beta^*}\f$. */
    double _gamma_2;
    /** @brief Flag to enable limiting of the production term. */
    bool _limit_production;

  public:
    // Evaluated fields
    // ---------------------------------------------------------

    /** @brief Source term for the turbulent kinetic energy equation. */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k_source;
    /** @brief Production term for the turbulent kinetic energy equation. */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k_prod;
    /** @brief Destruction term for the turbulent kinetic energy equation. */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k_dest;
    /** @brief Source term for the specific dissipation rate equation. */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _w_source;
    /** @brief Production term for the specific dissipation rate equation. */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _w_prod;
    /** @brief Destruction term for the specific dissipation rate equation. */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _w_dest;
    /** @brief Cross‑diffusion term for the specific dissipation rate equation.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _w_cross;
};

/** @} */ // end of ClosureModel

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_INCOMPRESSIBLESSTSOURCE_HPP