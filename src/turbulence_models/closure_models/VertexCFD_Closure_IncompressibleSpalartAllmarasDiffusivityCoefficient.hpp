#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLESPALARTALLMARASDIFFUSIVITYCOEFFICIENT_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLESPALARTALLMARASDIFFUSIVITYCOEFFICIENT_HPP

/**
 * @file IncompressibleSpalartAllmarasDiffusivityCoefficient.hpp
 */

/**
 * @defgroup ClosureModel Closure Model
 * @brief Components that provide closure relations for turbulence models.
 *
 * This module contains evaluators that compute model-specific quantities
 * required by the turbulence closure models.
 * @{
 */

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <Kokkos_Core.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//--------------------------------//
/**
 * @class IncompressibleSpalartAllmarasDiffusivityCoefficient
 * @brief Diffusivity coefficient evaluator for the incompressible
 * Spalart‑Allmaras turbulence model (SA‑neg).
 *
 * This class computes the diffusivity associated with the Spalart‑Allmaras
 * turbulence variable.  The formulation follows the standard SA‑neg model
 * where the diffusivity is given by the expression documented in the file
 * description.
 *
 * @tparam EvalType Evaluation type providing the scalar type.
 * @tparam Traits   Traits class required by the Phalanx evaluator framework.
 */
template<class EvalType, class Traits>
class IncompressibleSpalartAllmarasDiffusivityCoefficient
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    /**
     * @brief Constructor.
     *
     * Registers the dependent fields (SA variable and kinematic viscosity)
     * and the evaluated diffusivity field.
     *
     * @param[in] ir Integration rule providing the data layout and spatial
     * dimension.
     */
    IncompressibleSpalartAllmarasDiffusivityCoefficient(
        const panzer::IntegrationRule& ir);

    /**
     * @brief Evaluate the diffusivity field over the workset.
     *
     * This method launches a Kokkos parallel kernel that computes the
     * diffusivity at each integration point.
     *
     * @param[in] workset Evaluation data containing the number of cells.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /**
     * @brief Kokkos functor operator executed by each team.
     *
     * Computes the diffusivity for all points in a given cell.
     *
     * @param[in] team Team policy member representing a cell.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    /** SA turbulence variable \f$\tilde{\nu}\f$ (input). */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _sa_var;
    /** Kinematic viscosity \f$\nu\f$ (input). */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu;

    /** Model constant \f$c_{n1}=16.0\f$. */
    const double _cn1;
    /** Model constant \f$\sigma = 2/3\f$. */
    const double _sigma;
    /** Constant one of type scalar_type. */
    const scalar_type _one;
    /** Number of spatial dimensions (gradient dimension). */
    const int _num_grad_dim;

  public:
    /** Diffusivity field \f$D\f$ (output). */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _diffusivity_var;
};

//--------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

/** @} */ // end of ClosureModel

#endif // VERTEXCFD_CLOSURE_INCOMPRESSIBLESPALARTALLMARASDIFFUSIVITYCOEFFICIENT_HPP