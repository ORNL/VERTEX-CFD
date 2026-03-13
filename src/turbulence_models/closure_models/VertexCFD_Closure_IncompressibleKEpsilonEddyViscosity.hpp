#ifndef VERTEXCFD_CLOSURE_INCOMPRESSIBLEKEPSILONEDDYVISCOSITY_HPP
#define VERTEXCFD_CLOSURE_INCOMPRESSIBLEKEPSILONEDDYVISCOSITY_HPP

/**
 * @file IncompressibleKEpsilonEddyViscosity.hpp
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
/** @defgroup ClosureModel Closure Model
 *  @brief Turbulence closure models used in VertexCFD.
 *  @{
 */

/**
 * @class IncompressibleKEpsilonEddyViscosity
 * @brief Closure model that computes the turbulent eddy viscosity using the
 * incompressible K‑Epsilon formulation.
 *
 * The model evaluates
 *
 * \f[
 *   \nu_t = C_\nu \frac{k^2}{\varepsilon}
 * \f]
 *
 * where \f$k\f$ is the turbulent kinetic energy, \f$\varepsilon\f$ is the
 * turbulent dissipation rate, and \f$C_\nu = 0.09\f$ is a model constant.
 *
 * @tparam EvalType Evaluation type providing the scalar type.
 * @tparam Traits   Traits class required by the Phalanx evaluator framework.
 */
template<class EvalType, class Traits>
class IncompressibleKEpsilonEddyViscosity
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    /** @brief Scalar type used for the evaluation (e.g., double or AD type).
     */
    using scalar_type = typename EvalType::ScalarT;

    /**
     * @brief Constructor that registers dependent and evaluated fields.
     *
     * @param[in] ir Integration rule providing the data layout for fields.
     */
    IncompressibleKEpsilonEddyViscosity(const panzer::IntegrationRule& ir);

    /**
     * @brief Evaluate the eddy viscosity field for all cells in the workset.
     *
     * @param[in] workset Evaluation data containing cell information.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /**
     * @brief Kokkos functor invoked for each team (cell) to compute
     * \f$\nu_t\f$.
     *
     * @param[in] team Team policy member representing a cell.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  private:
    /** @brief Turbulent kinetic energy field \f$k\f$ (input). */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_kinetic_energy;

    /** @brief Turbulent dissipation rate field \f$\varepsilon\f$ (input). */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _turb_dissipation_rate;

    /** @brief Model constant \f$C_\nu = 0.09\f$. */
    double _C_nu;

    /** @brief Number of spatial dimensions (gradient dimension). */
    int _num_grad_dim;

  public:
    /** @brief Computed turbulent eddy viscosity field \f$\nu_t\f$ (output). */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _nu_t;
};

/** @} */ // end of ClosureModel group

} // end namespace ClosureModel
} // end namespace VertexCFD

#endif // VERTEXCFD_CLOSURE_INCOMPRESSIBLEKEPSILONEDDYVISCOSITY_HPP