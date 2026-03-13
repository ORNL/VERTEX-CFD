#ifndef VERTEXCFD_BOUNDARYSTATE_VARIABLEFLUX_HPP
#define VERTEXCFD_BOUNDARYSTATE_VARIABLEFLUX_HPP

/**
 * @file VertexCFD_BoundaryState_VariableFlux.hpp
 * @brief Declaration of the constant flux boundary condition evaluator.
 *
 * Provides the @c VariableFlux class which implements a constant flux
 * boundary condition for the VertexCFD solver.
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
 * @class VariableFlux
 * @brief Boundary condition evaluator for constant wall flux condition.
 *
 * This evaluator imposes a constant flux condition on a wall.
 * It provides the boundary variable and its gradient required by the
 * surrounding physics. The class follows the Panzer/Phalanx evaluator
 * pattern and can be executed in a Kokkos parallel region.
 *
 * @tparam EvalType Evaluation type providing the scalar type.
 * @tparam Traits   Traits class defining the evaluation data.
 */
template<class EvalType, class Traits>
class VariableFlux : public panzer::EvaluatorWithBaseImpl<Traits>,
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
     * The constant flux enforces a constant flux on the wall such that:
     * \f[ \mathbf{n}\cdot\nabla \phi = \frac{q}{k} \f],
     * where \f$\mathbf{n}\f$ is the outward normal vector at the boundary,
     * \f$q\f$ is the prescribed flux, and \f$k\f$ is the dependent variable
     * (if exists).
     *
     * @param ir Integration rule that supplies the data layout for the
     *           fields (scalar and vector layouts).
     */
    VariableFlux(const panzer::IntegrationRule& ir,
                 const Teuchos::ParameterList& bc_params,
                 const std::string variable_name,
                 const std::string coefficient_name = "");

    /**
     * @brief Evaluate the fields for a given workset.
     *
     * Computes the boundary variable and its gradient according to the
     * constant flux condition.
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
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_variable;

    /** @brief Evaluated field: gradient of the boundary variable.
     *
     * Indexed by cell, integration point, and spatial dimension.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _boundary_grad_variable;

  private:
    /** @brief Dependent field: variable in the interior domain.
     *
     * Read‑only field used to compute the boundary value.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _variable;
    /** @brief Dependent field: gradient of the interior variable.
     *
     * Read‑only field required for the constant flux condition.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_variable;

    /** @brief Dependent field: outward normal vectors at the boundary.
     *
     * Used to enforce the flux condition in wall normal direction.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _normals;
    /** @brief Dependent field: dependent variable at the boundary.
     *
     * Used to compute the flux from the gradient (optional).
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _dependent;

    /** @brief Dependent field: boolean for flux dependence.
     *
     * Used to decide whether to compute the flux by dividing to a dependent
     * variable.
     */
    bool _flux_dependence;

    /** @brief Dependent field: flux value at the boundary.
     *
     * Used to compute the flux.
     */
    double _flux;
};

} // end namespace BoundaryCondition
} // end namespace VertexCFD

/** @} */ // end of VertexCFD_BoundaryCondition

#endif // VERTEXCFD_BOUNDARYSTATE_VARIABLEFLUX_HPP