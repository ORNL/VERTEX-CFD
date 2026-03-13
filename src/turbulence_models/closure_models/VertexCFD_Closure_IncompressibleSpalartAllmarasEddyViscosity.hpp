#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLESPALARTALLMARASEDDYVISCOSITY_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLESPALARTALLMARASEDDYVISCOSITY_HPP

/**
 * @file IncompressibleSpalartAllmarasEddyViscosity.hpp
 */

/**
 * @defgroup ClosureModel Closure Model
 * @brief Turbulence closure models for VertexCFD.
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
//-------------------------------//
// Turbulent eddy viscosity for Spalart-Allmaras turbulence model (SA‑neg)
//----------------------------------------------//

/**
 * @class IncompressibleSpalartAllmarasEddyViscosity
 * @brief Closure model that computes the turbulent eddy viscosity for the
 *        incompressible Spalart‑Allmaras (SA‑neg) turbulence model.
 *
 * The evaluator extracts the Spalart‑Allmaras working variable
 * \f$\tilde{\nu}\f$ and the kinematic viscosity \f$\nu\f$, applies the
 * model constants, and produces the turbulent eddy viscosity \f$\nu_t\f$.
 *
 * @tparam EvalType Evaluation type providing the scalar type (e.g., Residual,
 * Jacobian).
 * @tparam Traits   Traits class required by the Phalanx evaluator framework.
 */
template<class EvalType, class Traits>
class IncompressibleSpalartAllmarasEddyViscosity
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    //============================================================
    // Type aliases
    //============================================================
    using scalar_type = typename EvalType::ScalarT;

    //============================================================
    // Constructor
    //============================================================
    /**
     * @brief Construct the evaluator and bind required fields.
     *
     * @param ir Integration rule providing cell topology and quadrature
     * information.
     */
    IncompressibleSpalartAllmarasEddyViscosity(const panzer::IntegrationRule& ir);

    //============================================================
    // Evaluation routine
    //============================================================
    /**
     * @brief Compute the turbulent eddy viscosity at each integration point.
     *
     * @param workset Evaluation data containing the current workset of cells.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    //============================================================
    // Parallel functor
    //============================================================
    /**
     * @brief Kokkos functor executed by a team policy.
     *
     * @param team Team member used for hierarchical parallelism.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    //============================================================
    // Dependent fields
    //============================================================
    /**
     * @brief Spalart‑Allmaras working variable \f$\tilde{\nu}\f$
     * (dimensionless).
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _sa_var;
    /**
     * @brief Kinematic viscosity \f$\nu\f$
     * (m\textsuperscript{2}\,s\textsuperscript{-1}).
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _nu;

    //============================================================
    // Model constants
    //============================================================
    /**
     * @brief Model constant \f$C_{v1}=7.1\f$ used in the viscosity limiter.
     */
    const double _cv1;
    /**
     * @brief Number of spatial dimensions (e.g., 2 or 3).
     */
    const int _num_grad_dim;

  public:
    //============================================================
    // Evaluated field
    //============================================================
    /**
     * @brief Turbulent eddy viscosity \f$\nu_t\f$
     * (m\textsuperscript{2}\,s\textsuperscript{-1}).
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _nu_t;
};

//----------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

/** @} */ // end of ClosureModel

#endif // VERTEXCFD_CLOSURE_INCOMPRESSIBLESPALARTALLMARASEDDYVISCOSITY_HPP