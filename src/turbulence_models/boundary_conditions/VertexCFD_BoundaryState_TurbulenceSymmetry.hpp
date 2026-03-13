#ifndef VERTEXCFD_BOUNDARYSTATE_TURBULENCESYMMETRY_HPP
#define VERTEXCFD_BOUNDARYSTATE_TURBULENCESYMMETRY_HPP

/**
 * @file TurbulenceSymmetry.hpp
 */

/**
 * @defgroup BoundaryCondition Boundary Condition
 * @brief Module for boundary condition evaluators.
 * @{
 */

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <string>

namespace VertexCFD
{
namespace BoundaryCondition
{

/**
 * @class TurbulenceSymmetry
 * @brief Symmetry boundary condition for turbulence quantities.
 *
 * This evaluator enforces a zero normal gradient for a specified
 * turbulence variable on a boundary face, consistent with a
 * symmetry plane.
 *
 * @tparam EvalType Evaluation type providing the scalar type.
 * @tparam Traits   Phalanx traits class.
 */
template<class EvalType, class Traits>
class TurbulenceSymmetry : public panzer::EvaluatorWithBaseImpl<Traits>,
                           public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    /**
     * @brief Alias for the scalar type associated with the evaluation type.
     */
    using scalar_type = typename EvalType::ScalarT;

    /**
     * @brief Constructor.
     *
     * @param ir            Integration rule defining the side topology and
     * data layout.
     * @param variable_name Name of the turbulence variable (e.g., "k",
     * "omega").
     */
    TurbulenceSymmetry(const panzer::IntegrationRule& ir,
                       const std::string variable_name);

    /**
     * @brief Evaluate the boundary state for the given workset.
     *
     * @param workset Evaluation data containing cell and point information.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /**
     * @brief Functor invoked by Kokkos @c parallel_for to apply the symmetry
     * condition.
     *
     * @param team Kokkos team member representing a cell (league rank).
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    /**
     * @brief Boundary value of the turbulence variable.
     *
     * Indexed as \f$(\text{cell},\text{point})\f$.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_variable;

    /**
     * @brief Boundary gradient of the turbulence variable.
     *
     * Indexed as \f$(\text{cell},\text{point},\text{dim})\f$.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _boundary_grad_variable;

  private:
    /**
     * @brief Interior value of the turbulence variable.
     *
     * Indexed as \f$(\text{cell},\text{point})\f$.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _variable;

    /**
     * @brief Interior gradient of the turbulence variable.
     *
     * Indexed as \f$(\text{cell},\text{point},\text{dim})\f$.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _grad_variable;

    /**
     * @brief Outward‑pointing normal vectors on the side.
     *
     * Indexed as \f$(\text{cell},\text{point},\text{dim})\f$.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _normals;

    /**
     * @brief Number of spatial dimensions for the gradient (typically 2 or 3).
     */
    int _num_grad_dim;
};

} // namespace BoundaryCondition
} // namespace VertexCFD

/** @} */ // end of BoundaryCondition group

#endif // VERTEXCFD_BOUNDARYSTATE_TURBULENCESYMMETRY_HPP