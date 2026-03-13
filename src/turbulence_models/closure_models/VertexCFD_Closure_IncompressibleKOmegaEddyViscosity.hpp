#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEKOMEGAEDDYVISCOSITY_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEKOMEGAEDDYVISCOSITY_HPP

/**
 * @file IncompressibleKOmegaEddyViscosity.hpp
 */

/**
 * @defgroup ClosureModel Closure Model
 * @brief Turbulence closure models.
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
//--------------------------------//
// Turbulent eddy viscosity for the Wilcox (2006) K‑Omega turbulence model
//--------------------------------//

/**
 * @class IncompressibleKOmegaEddyViscosity
 * @brief Incompressible K‑Omega eddy‑viscosity closure model.
 *
 * This evaluator computes the turbulent eddy viscosity \f$\nu_t\f$ from the
 * turbulent kinetic energy \f$k\f$ and the specific dissipation rate
 * \f$\omega\f$ using the Wilcox (2006) formulation.  The model also
 * incorporates a limiter based on the strain‑rate magnitude to ensure
 * numerical stability.
 *
 * @tparam EvalType   Evaluation type (e.g., Residual, Jacobian) providing the
 * scalar type.
 * @tparam Traits     Traits class required by the Phalanx evaluator framework.
 * @tparam NumSpaceDim Number of spatial dimensions (e.g., 2 or 3).
 */
template<class EvalType, class Traits, int NumSpaceDim>
class IncompressibleKOmegaEddyViscosity
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;
    static constexpr int num_space_dim = NumSpaceDim;

    // --------------------------------------------------------------
    /**
     * @brief Constructor.
     *
     * Registers dependent fields (turbulent kinetic energy and specific
     * dissipation rate) and the evaluated field (eddy viscosity).  Model
     * coefficients \f$C_{\text{lim}}\f$ and \f$\beta^{*}\f$ can be overridden
     * via the supplied parameter list.
     *
     * @param ir           Integration rule providing the data layout.
     * @param turb_params  Parameter list containing optional model
     * coefficients.
     */
    // --------------------------------------------------------------
    IncompressibleKOmegaEddyViscosity(const panzer::IntegrationRule& ir,
                                      const Teuchos::ParameterList& turb_params);

    // --------------------------------------------------------------
    /**
     * @brief Evaluate the eddy‑viscosity field over the workset.
     *
     * This method launches a Kokkos team‑parallel kernel that computes
     *
     * \f[
     * \nu_t = \frac{k}{\max\!\bigl(\omega,\;
     * C_{\text{lim}}\sqrt{\frac{2\,S_{ij}S_{ij}}{\beta^{*}}},\;
     * \varepsilon\bigr)}
     * \f]
     *
     * where \f$S_{ij}\f$ is the strain‑rate tensor and
     * \f$\varepsilon\f$ is a small tolerance to avoid division by zero.
     */
    // --------------------------------------------------------------
    void evaluateFields(typename Traits::EvalData workset) override;

    // --------------------------------------------------------------
    /**
     * @brief Kokkos team functor.
     *
     * Computes the strain‑rate magnitude and the eddy viscosity for each
     * cell/point in the workset.
     *
     * @param team  Kokkos team policy member representing a cell.
     */
    // --------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    // --------------------------------------------------------------
    /**
     * @brief Turbulent kinetic energy field \f$k\f$ (input).
     */
    // --------------------------------------------------------------
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_kinetic_energy;

    // --------------------------------------------------------------
    /**
     * @brief Specific dissipation rate field \f$\omega\f$ (input).
     */
    // --------------------------------------------------------------
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_specific_dissipation_rate;

    // --------------------------------------------------------------
    /**
     * @brief Gradient of the velocity field \f$\partial u_i/\partial x_j\f$
     * (input).
     *
     * Stored as an array over spatial dimensions: \f$\nabla \mathbf{u}\f$.
     */
    // --------------------------------------------------------------
    Kokkos::Array<
        PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>,
        num_space_dim>
        _grad_velocity;

    // --------------------------------------------------------------
    /**
     * @brief Limiter coefficient \f$C_{\text{lim}}\f$ (model parameter).
     */
    // --------------------------------------------------------------
    double _C_lim;

    // --------------------------------------------------------------
    /**
     * @brief Model constant \f$\beta^{*}\f$ (typically 0.09).
     */
    // --------------------------------------------------------------
    double _beta_star;

  public:
    // --------------------------------------------------------------
    /**
     * @brief Computed turbulent eddy viscosity \f$\nu_t\f$ (output).
     */
    // --------------------------------------------------------------
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _nu_t;
};

//-----------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

/** @} */ // end of ClosureModel

#endif // end
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLEKOMEGAEDDYVISCOSITY_HPP