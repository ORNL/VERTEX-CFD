#ifndef VERTEXCFD_BOUNDARYSTATE_ADIABATICWALL_HPP
#define VERTEXCFD_BOUNDARYSTATE_ADIABATICWALL_HPP

/**
 * @file VertexCFD_BoundaryState_AdiabaticWall.hpp
 * @brief Declaration of the adiabatic wall boundary condition evaluator.
 *
 * Provides the @c AdiabaticWall class which implements a no‑heat‑flux
 * (adiabatic) wall condition for the VertexCFD solver.
 */

/**
 * @defgroup VertexCFD_BoundaryCondition Boundary Condition
 * @brief Boundary condition evaluators for VertexCFD.
 *
 * This module contains evaluators that impose various physical
 * conditions on domain boundaries.
 * @{
 */

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_IntegrationRule.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_KokkosDeviceTypes.hpp>
#include <Phalanx_MDField.hpp>

#include <Teuchos_ParameterList.hpp>

#include <Kokkos_Core.hpp>

namespace VertexCFD
{
namespace BoundaryCondition
{

/**
 * @class AdiabaticWall
 * @brief Boundary condition evaluator for an adiabatic wall.
 *
 * This evaluator imposes an adiabatic (no heat flux) condition on a wall.
 * It provides the boundary temperature and its gradient required by the
 * surrounding physics.  The class follows the Panzer/Phalanx evaluator
 * pattern and can be executed in a Kokkos parallel region.
 *
 * @tparam EvalType Evaluation type providing the scalar type.
 * @tparam Traits   Traits class defining the evaluation data.
 */
template<class EvalType, class Traits>
class AdiabaticWall : public panzer::EvaluatorWithBaseImpl<Traits>,
                      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    /** @brief Alias for the scalar type used by the evaluation. */
    using scalar_type = typename EvalType::ScalarT;

    /**
     * @brief Constructor.
     *
     * Registers the fields that will be evaluated or used as dependencies.
     *
     * The adiabatic wall enforces a zero normal heat flux:
     * \f[ \mathbf{n}\cdot\nabla T = 0 \f],
     * where \f$\mathbf{n}\f$ is the outward normal vector at the boundary.
     *
     * @param ir Integration rule that supplies the data layout for the
     *           fields (scalar and vector layouts).
     */
    AdiabaticWall(const panzer::IntegrationRule& ir);

    /**
     * @brief Evaluate the fields for a given workset.
     *
     * Computes the boundary temperature and its gradient according to the
     * adiabatic condition.
     *
     * @param workset Evaluation data for the current workset.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /**
     * @brief Kokkos functor invoked for each team in a parallel region.
     *
     * Implements the actual per‑cell computation of the boundary state.
     *
     * @param team Team member provided by Kokkos::TeamPolicy.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    /** @brief Evaluated field: temperature prescribed on the boundary.
     *
     * Indexed by cell and integration point.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_temperature;

    /** @brief Evaluated field: gradient of the boundary temperature.
     *
     * Indexed by cell, integration point, and spatial dimension.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _boundary_grad_temperature;

  private:
    /** @brief Dependent field: temperature in the interior domain.
     *
     * Read‑only field used to compute the boundary value.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _temperature;

    /** @brief Dependent field: gradient of the interior temperature.
     *
     * Read‑only field required for the adiabatic condition.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_temperature;

    /** @brief Dependent field: outward normal vectors at the boundary.
     *
     * Used to enforce the zero normal heat flux condition.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _normals;
};

} // end namespace BoundaryCondition
} // end namespace VertexCFD

/** @} */ // end of VertexCFD_BoundaryCondition

#endif // VERTEXCFD_BOUNDARYSTATE_ADIABATICWALL_HPP
