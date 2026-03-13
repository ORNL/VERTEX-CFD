#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEKOMEGADIFFUSIVITYCOEFFICIENT_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEKOMEGADIFFUSIVITYCOEFFICIENT_HPP

/**
 * @file IncompressibleKOmegaDiffusivityCoefficient.hpp
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
 * @class IncompressibleKOmegaDiffusivityCoefficient
 * @brief Evaluator that computes turbulent diffusivity coefficients for
 *        incompressible K‑Omega model (Wilcox 2006).
 *
 * The class provides the diffusivity fields for the turbulent kinetic energy
 * \f$k\f$ and the specific dissipation rate \f$\omega\f$ based on the local
 * turbulent quantities and model constants \f$\sigma_k\f$ and
 * \f$\sigma_\omega\f$.
 *
 * @tparam EvalType Evaluation type (e.g., Residual, Jacobian).
 * @tparam Traits   Traits class providing evaluation data types.
 */
template<class EvalType, class Traits>
class IncompressibleKOmegaDiffusivityCoefficient
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    /** @brief Scalar type used for the evaluation (e.g., double or AD type).
     */
    using scalar_type = typename EvalType::ScalarT;

    /**
     * @brief Constructor registers dependent and evaluated fields and reads
     * model parameters.
     *
     * @param ir          Integration rule that provides the data layout for
     * fields.
     * @param turb_params Parameter list that may contain user‑defined values
     * for "sigma_k" and "sigma_w".
     */
    IncompressibleKOmegaDiffusivityCoefficient(
        const panzer::IntegrationRule& ir,
        const Teuchos::ParameterList& turb_params);

    /**
     * @brief Evaluate the diffusivity fields for all cells in the workset.
     *
     * @param workset Evaluation data containing the number of cells and other
     *                context needed for field evaluation.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /**
     * @brief Kokkos functor executed in parallel over cells/teams.
     *
     * Computes the diffusivity contributions for each quadrature point.
     *
     * @param team Kokkos team member representing a cell.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    // Dependent fields -------------------------------------------------

    /**
     * @brief Turbulent kinetic energy \f$k\f$ at each cell and quadrature
     * point.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_kinetic_energy;

    /**
     * @brief Specific dissipation rate \f$\omega\f$ at each cell and
     * quadrature point.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_specific_dissipation_rate;

    /**
     * @brief Molecular (kinematic) viscosity \f$\nu\f$ at each cell and point.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu;

    // Model coefficients ------------------------------------------------

    /**
     * @brief Model constant \f$\sigma_k\f$ (default 0.6) used for the \f$k\f$
     * diffusivity.
     */
    double _sigma_k;

    /**
     * @brief Model constant \f$\sigma_\omega\f$ (default 0.5) used for the
     * \f$\omega\f$ diffusivity.
     */
    double _sigma_w;

  public:
    // Evaluated fields -------------------------------------------------

    /**
     * @brief Diffusivity field for turbulent kinetic energy \f$k\f$.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _diffusivity_var_k;

    /**
     * @brief Diffusivity field for specific dissipation rate \f$\omega\f$.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _diffusivity_var_w;
};

/** @} */ // end of ClosureModel

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_INCOMPRESSIBLEKOMEGADIFFUSIVITYCOEFFICIENT_HPP