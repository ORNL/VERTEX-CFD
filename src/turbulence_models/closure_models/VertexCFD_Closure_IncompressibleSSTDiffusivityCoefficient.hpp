#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLESSTDIFFUSIVITYCOEFFICIENT_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLESSTDIFFUSIVITYCOEFFICIENT_HPP

/**
 * @file IncompressibleSSTDiffusivityCoefficient.hpp
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
 * @class IncompressibleSSTDiffusivityCoefficient
 * @brief Diffusivity coefficients for Menter's SST K‑Omega turbulence model.
 *
 * This evaluator computes the effective diffusivity for the turbulent kinetic
 * energy (k) and the specific dissipation rate (ω) based on the SST blending
 * function and the turbulent eddy viscosity. The formulation follows the
 * standard SST model where the diffusivity is a linear combination of the
 * molecular kinematic viscosity \f$\nu\f$ and the turbulent viscosity
 * \f$\nu_t\f$ weighted by model constants \f$\sigma_{k1},\sigma_{k2},
 * \sigma_{\omega1},\sigma_{\omega2}\f$ and the blending function
 * \f$F_{1}\f$.
 */
template<class EvalType, class Traits>
class IncompressibleSSTDiffusivityCoefficient
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    /**
     * @brief Constructor.
     * @param ir Integration rule providing the data layout for fields.
     * @param turb_params Parameter list that may contain user‑defined values
     * for the model constants \f$\sigma_{k1},\sigma_{k2},
     *        \sigma_{\omega1},\sigma_{\omega2}\f$.
     */
    IncompressibleSSTDiffusivityCoefficient(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& turb_params);

    /**
     * @brief Evaluate the diffusivity fields for a workset.
     *
     * This method launches a Kokkos parallel kernel that computes the
     * diffusivities at each integration point of each cell in the workset.
     *
     * @param workset Evaluation data containing the number of cells.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /**
     * @brief Kokkos functor executed by each team.
     *
     * Computes the diffusivity for all integration points belonging to a
     * single cell. The implementation uses the blending function
     * \f$_{sst\_blending\_function}\f$ to blend the two sets of model
     * constants.
     *
     * @param team Kokkos team policy member representing a cell.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    // Input fields
    // --------------------------------------------------------------

    /**
     * @brief Turbulent eddy viscosity \f$\nu_t\f$ (m\(^2\)/s).
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu_t;

    /**
     * @brief SST blending function \f$F_{1}\f$ (dimensionless, 0–1).
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _sst_blending_function;

    /**
     * @brief Molecular kinematic viscosity \f$\nu\f$ (m\(^2\)/s).
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu;

    // Model constants
    // -------------------------------------------------------

    /**
     * @brief \f$\sigma_{k1}\f$ constant for the k‑equation in the inner
     * (viscous) region.
     */
    double _sigma_k1;

    /**
     * @brief \f$\sigma_{k2}\f$ constant for the k‑equation in the outer
     * (turbulent) region.
     */
    double _sigma_k2;

    /**
     * @brief \f$\sigma_{\omega1}\f$ constant for the ω‑equation in the inner
     * region.
     */
    double _sigma_w1;

    /**
     * @brief \f$\sigma_{\omega2}\f$ constant for the ω‑equation in the outer
     * region.
     */
    double _sigma_w2;

  public:
    // Output fields
    // -------------------------------------------------------

    /**
     * @brief Effective diffusivity for turbulent kinetic energy \f$k\f$:
     *
     * \f[
     * D_k = \nu + \bigl(\sigma_{k1} F_{1}
     *      + \sigma_{k2}(1-F_{1})\bigr)\nu_t
     * \f]
     *
     * (m\(^2\)/s).
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _diffusivity_var_k;

    /**
     * @brief Effective diffusivity for specific dissipation rate \f$\omega\f$:
     *
     * \f[
     * D_\omega = \nu + \bigl(\sigma_{\omega1} F_{1}
     *      + \sigma_{\omega2}(1-F_{1})\bigr)\nu_t
     * \f]
     *
     * (m\(^2\)/s).
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _diffusivity_var_w;
};

/** @} */ // end of ClosureModel

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_INCOMPRESSIBLESSTDIFFUSIVITYCOEFFICIENT_HPP
       // VERTEXCFD_CLOSURE_INCOMPRESSIBLEKOMEGADIFFUSIVITYCOEFFICIENT_HPP