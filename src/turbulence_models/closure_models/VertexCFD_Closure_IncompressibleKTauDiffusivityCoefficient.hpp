#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEKTAUDIFFUSIVITYCOEFFICIENT_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEKTAUDIFFUSIVITYCOEFFICIENT_HPP

/**
 * @file IncompressibleKTauDiffusivityCoefficient.hpp
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
 * @brief Turbulence closure models.
 * @{
 */

/**
 * @class IncompressibleKTauDiffusivityCoefficient
 * @brief Provides turbulent diffusivity coefficients for the incompressible
 * K‑Tau model.
 *
 * The diffusivity for the turbulent kinetic energy \f$k\f$ and the specific
 * dissipation rate \f$\omega\f$ are computed as
 *
 * \f[
 *   \nu_{k} = \nu + \sigma_{k}\,k\,\omega ,\qquad
 *   \nu_{\omega} = \nu + \sigma_{\omega}\,k\,\omega ,
 * \f]
 *
 * where \f$\nu\f$ is the molecular kinematic viscosity and \f$\sigma_{k}\f$,
 * \f$\sigma_{\omega}\f$ are model constants that may be overridden via the
 * input parameter list.
 * @tparam EvalType Evaluation type providing the scalar type.
 * @tparam Traits   Traits class required by the Phalanx evaluator framework.
 */
template<class EvalType, class Traits>
class IncompressibleKTauDiffusivityCoefficient
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    /**
     * @brief Constructor.
     * @param ir          Integration rule defining the data layout.
     * @param turb_params Parameter list containing turbulence model
     * coefficients (e.g., "sigma_k", "sigma_t").
     */
    IncompressibleKTauDiffusivityCoefficient(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& turb_params);

    /**
     * @brief Evaluate the diffusivity fields for the current workset.
     * @param workset Evaluation data containing cell information.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /**
     * @brief Kokkos functor executed for each team (cell) in parallel.
     * @param team Kokkos team member representing a cell.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    // Input fields -------------------------------------------------------
    /** @brief Turbulent kinetic energy \f$k\f$ at quadrature points. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_kinetic_energy;

    /** @brief Specific dissipation rate \f$\omega\f$ at quadrature points. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_specific_dissipation_rate;

    /** @brief Molecular kinematic viscosity \f$\nu\f$ at quadrature points. */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu;

    // Model constants ----------------------------------------------------
    /** @brief Constant \f$\sigma_{k}\f$ used in the diffusivity for \f$k\f$.
     */
    double _sigma_k;

    /** @brief Constant \f$\sigma_{\omega}\f$ (named _sigma_t) used in the
     * diffusivity for \f$\omega\f$. */
    double _sigma_t;

  public:
    // Output fields -------------------------------------------------------
    /** @brief Diffusivity for turbulent kinetic energy \f$k\f$. */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _diffusivity_var_k;

    /** @brief Diffusivity for specific dissipation rate \f$\omega\f$. */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _diffusivity_var_t;
};

/** @} */ // end of ClosureModel

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_INCOMPRESSIBLEKTAUDIFFUSIVITYCOEFFICIENT_HPP