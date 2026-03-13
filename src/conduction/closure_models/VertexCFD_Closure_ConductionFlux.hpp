#ifndef VERTEXCFD_CLOSURE_CONDUCTIONFLUX_HPP
#define VERTEXCFD_CLOSURE_CONDUCTIONFLUX_HPP

/**
 * @file VertexCFD_Closure_ConductionFlux.hpp
 * @brief Evaluator for conductive heat flux in closure models.
 *
 * Provides the ConductionFlux evaluator that computes the conductive heat
 * flux \f$\mathbf{q} = -k \nabla T\f$ for each integration point.
 */

/**
 * @defgroup ClosureModel Conduction Flux Closure Model
 * @brief Conduction flux evaluation within VertexCFD.
 *
 * Contains evaluators related to thermal conduction.
 * @{
 */

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_IntegrationRule.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_KokkosDeviceTypes.hpp>
#include <Phalanx_MDField.hpp>

#include <Kokkos_Core.hpp>

#include <string>

namespace VertexCFD
{
namespace ClosureModel
{
//-----------------------------------------------------------------//
// Thermal flux evaluation for conduction.
//-----------------------------------------------------------------//

/**
 * @class ConductionFlux
 * @brief Evaluator that computes the conductive heat flux
 *        \f$\mathbf{q} = -k \nabla T\f$.
 *
 * The evaluator is templated on the evaluation type (e.g., Residual,
 * Jacobian) and the Traits class required by the Panzer/Phalanx framework.
 *
 * @tparam EvalType Evaluation type providing the scalar type.
 * @tparam Traits   Traits class defining the evaluation data layout.
 */
template<class EvalType, class Traits>
class ConductionFlux : public panzer::EvaluatorWithBaseImpl<Traits>,
                       public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    /**
     * @brief Constructor.
     *
     * Registers the fields required for the evaluation of the conductive
     * flux. The flux field name is formed by concatenating \c flux_prefix
     * with the string "CONDUCTION_FLUX_energy". The temperature gradient
     * field name is formed analogously using \c gradient_prefix.
     *
     * @param[in] ir               Integration rule providing data layouts.
     * @param[in] flux_prefix      Optional prefix for the flux field name.
     * @param[in] gradient_prefix  Optional prefix for the temperature gradient
     *                             field name.
     */
    ConductionFlux(const panzer::IntegrationRule& ir,
                   const std::string& flux_prefix = "",
                   const std::string& gradient_prefix = "");

    /**
     * @brief Evaluate the conductive flux for each cell/point.
     *
     * This method is called by the Phalanx evaluation engine. It loops over
     * the workset and computes \f$\mathbf{q}_i = -k\,\nabla T_i\f$ for each
     * integration point \f$i\f$.
     *
     * @param[in] workset Evaluation data containing the current workset.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /**
     * @brief Kokkos functor invoked for a team of threads.
     *
     * The functor computes the flux for a subset of cells assigned to the
     * given team. It is intended for use with a Kokkos::TeamPolicy.
     *
     * @param[in] team Team member provided by Kokkos.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    /**
     * @brief Field that stores the computed conductive flux.
     *
     * The field has layout (Cell, Point, Dim) and is evaluated by this
     * evaluator.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _conduction_flux;

  private:
    /**
     * @brief Temperature gradient field (input).
     *
     * Layout: (Cell, Point, Dim). Marked as const because it is not
     * modified by this evaluator.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_temperature;

    /**
     * @brief Thermal conductivity field (input).
     *
     * Layout: (Cell, Point). Represents the scalar conductivity \f$k\f$
     * at each integration point.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point>
        _thermal_conductivity;

    /** @brief Number of spatial dimensions (e.g., 2 or 3). */
    int _num_space_dim;
};

//-----------------------------------------------------------------//

} // end namespace ClosureModel
} // end namespace VertexCFD

/** @} */ // end of ClosureModel group

#endif // VERTEXCFD_CLOSURE_CONDUCTIONFLUX_HPP