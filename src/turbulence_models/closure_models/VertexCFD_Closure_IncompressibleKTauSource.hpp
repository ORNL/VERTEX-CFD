#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEKTAUSOURCE_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEKTAUSOURCE_HPP

/**
 * @file IncompressibleKTauSource.hpp
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
 * @brief Turbulence closure models for VertexCFD.
 * @{
 */

/**
 * @class IncompressibleKTauSource
 * @brief Evaluator for source terms of the incompressible K‑Tau turbulence
 * model.
 *
 * Computes production, destruction, and cross‑diffusion contributions for the
 * turbulent kinetic energy \f$k\f$ and specific dissipation rate \f$\tau\f$
 * equations. Model coefficients and limiter flags are read from a parameter
 * list.
 *
 * @tparam EvalType  Type providing the scalar type for the evaluation
 *                   (e.g. Residual, Jacobian).
 * @tparam Traits    Traits class required by the Phalanx evaluator framework.
 * @tparam NumSpaceDim Number of spatial dimensions (e.g. 2 or 3).
 */
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleKTauSource : public panzer::EvaluatorWithBaseImpl<Traits>,
                                 public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    /**
     * @brief Constructor.
     *
     * Registers dependent and evaluated fields and reads model coefficients
     * from the supplied parameter list.
     *
     * @param ir          Integration rule providing the data layout.
     * @param turb_params Parameter list containing turbulence model
     * coefficients and limiter flags.
     */
    IncompressibleKTauSource(const panzer::IntegrationRule& ir,
                             const Teuchos::ParameterList& turb_params);

    /**
     * @brief Evaluate the source terms over the workset.
     *
     * Launches a Kokkos parallel kernel that computes the production,
     * destruction, and cross‑diffusion contributions for each cell and
     * quadrature point.
     *
     * @param workset Evaluation data containing the number of cells.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /**
     * @brief Kokkos functor operator executed by each team.
     *
     * Computes the source terms for a single cell. The implementation follows
     * the standard K‑Tau model equations with optional limiting.
     *
     * @param team Team policy member representing the current cell.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    // Dependent fields
    // ---------------------------------------------------------

    /** @brief Molecular kinematic viscosity, \f$\nu\f$. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu;
    /** @brief Turbulent eddy viscosity, \f$\nu_t\f$. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu_t;
    /** @brief Turbulent kinetic energy, \f$k\f$. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_kinetic_energy;
    /** @brief Specific dissipation rate, \f$\tau\f$. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_specific_dissipation_rate;

    /** @brief Gradient of turbulent kinetic energy, \f$\nabla k\f$. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_turb_kinetic_energy;
    /** @brief Gradient of specific dissipation rate, \f$\nabla \tau\f$. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_turb_specific_dissipation_rate;
    /** @brief Velocity gradient tensor, \f$\partial u_i / \partial x_j\f$. */
    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_velocity;

    // Model coefficients
    // -------------------------------------------------------

    /** @brief Model constant \f$\beta^{*}=0.09\f$ (default, can be
     * overridden).
     */
    double _beta_star;
    /** @brief Model constant \f$\gamma=0.52\f$ (default, can be overridden).
     */
    double _gamma;
    /** @brief Model constant \f$\beta_{0}=0.0708\f$ (default, can be
     * overridden). */
    double _beta_0;
    /** @brief Model constant \f$\sigma_{d}=0.125\f$ (default, can be
     * overridden). */
    double _sigma_d;
    /** @brief Model constant \f$\sigma_{t}=0.5\f$ (default, can be
     * overridden).
     */
    double _sigma_t;
    /** @brief Flag to enable limiting of the production term. */
    bool _limit_production;
    /** @brief Flag to enable limiting of the destruction term. */
    bool _limit_destruction;

  public:
    // Evaluated fields
    // ---------------------------------------------------------

    /** @brief Source term for the turbulent kinetic energy equation,
     * \f$S_k\f$.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k_source;
    /** @brief Production term for \f$k\f$, \f$P_k\f$. */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k_prod;
    /** @brief Destruction term for \f$k\f$, \f$D_k\f$. */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _k_dest;
    /** @brief Source term for the specific dissipation rate equation,
     * \f$S_{\tau}\f$. */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _t_source;
    /** @brief Production term for \f$\tau\f$, \f$P_{\tau}\f$. */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _t_prod;
    /** @brief Destruction term for \f$\tau\f$ proportional to \nu_t,
     * \f$D_{\tau}^{\nu_t}\f$. */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _t_dest_nut;
    /** @brief Destruction term for \f$\tau\f$ proportional to molecular
     * viscosity, \f$D_{\tau}^{\nu}\f$. */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _t_dest_nu;
    /** @brief Cross‑diffusion term for \f$\tau\f$, \f$C_{\tau}\f$. */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _t_cross;
};

/** @} */ // end of ClosureModel

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_INCOMPRESSIBLEKTAUSOURCE_HPP