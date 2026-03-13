#ifndef VERTEXCFD_CLOSURE_CONDUCTIONTIMEDERIVATIVE_HPP
#define VERTEXCFD_CLOSURE_CONDUCTIONTIMEDERIVATIVE_HPP

/**
 * @file VertexCFD_Closure_ConductionTimeDerivative.hpp
 * @brief Conduction time‑derivative closure model.
 *
 * Provides the evaluator that computes the time‑derivative term in the
 * conduction energy equation.
 */

/**
 * @defgroup VertexCFD_ClosureModel Conduction Closure Models
 * @brief Closure models for conduction physics.
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
//--------------------------------------------------------------------//
template<class EvalType, class Traits>
class ConductionTimeDerivative : public panzer::EvaluatorWithBaseImpl<Traits>,
                                 public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    /**
     * @brief Construct with an integration rule.
     *
     * @param ir Integration rule defining the quadrature points.
     */
    ConductionTimeDerivative(const panzer::IntegrationRule& ir);

    /**
     * @brief Compute the field values.
     *
     * @param d Evaluation data supplied by the framework.
     */
    void evaluateFields(typename Traits::EvalData d) override;

    /**
     * @brief Kokkos functor for parallel evaluation over teams.
     *
     * @param team Team member for Kokkos parallel execution.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    /**
     * @brief Time‑derivative of the energy field.
     *
     * This field is written by the evaluator.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _dqdt_energy;

  private:
    /**
     * @brief Density field \f$\rho\f$.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _density;

    /**
     * @brief Specific heat capacity \f$c_p\f$.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _specific_heat;

    /**
     * @brief Temperature time derivative \f$\partial T/\partial t\f$.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _dxdt_temperature;
};

//--------------------------------------------------------------------//

} // end namespace ClosureModel
} // namespace VertexCFD

/** @} */ // end of VertexCFD_ClosureModel

#endif // VERTEXCFD_CLOSURE_CONDUCTIONTIMEDERIVATIVE_HPP