#ifndef VERTEXCFD_BOUNDARYSTATE_TURBULENCEBOUNDARYEDDYVISCOSITY_HPP
#define VERTEXCFD_BOUNDARYSTATE_TURBULENCEBOUNDARYEDDYVISCOSITY_HPP

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_FieldManager.hpp>
#include <Phalanx_config.hpp>

#include <string>

/**
 * @file TurbulenceBoundaryEddyViscosity.hpp
 */

namespace VertexCFD
{
namespace BoundaryCondition
{

/**
 * @class TurbulenceBoundaryEddyViscosity
 * @brief Evaluator that populates boundary turbulent eddy viscosity fields.
 * @tparam EvalType Evaluation type providing the scalar type.
 * @tparam Traits   Traits class defining evaluation data and field handling.
 */
template<class EvalType, class Traits>
class TurbulenceBoundaryEddyViscosity
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    /**
     * @brief Constructor.
     * @param ir          Integration rule providing cell topology and
     * quadrature.
     * @param bc_params   Parameter list describing the boundary condition.
     * @param flux_prefix Prefix used to name the evaluated field.
     */
    TurbulenceBoundaryEddyViscosity(const panzer::IntegrationRule& ir,
                                    const Teuchos::ParameterList& bc_params,
                                    const std::string& flux_prefix);

    /**
     * @brief Evaluate the boundary eddy viscosity for each cell in the
     * workset.
     * @param workset Evaluation data containing cell information.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    KOKKOS_INLINE_FUNCTION
    /**
     * @brief Kokkos functor executed in parallel over cells and points.
     * @param team Team policy member representing a cell.
     */
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

  public:
    /**
     * @brief Boundary eddy viscosity field to be evaluated.
     * @details Indexed by cell and quadrature point.
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _boundary_nu_t;

  private:
    /**
     * @brief Interior eddy viscosity field (input) used when no wall function
     * is present.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _interior_nu_t;
    /**
     * @brief Wall‑function eddy viscosity field (input) used when a
     * wall‑function BC is active.
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point> _wall_func_nu_t;

    /**
     * @brief Flag indicating whether a wall‑function boundary condition is
     * applied.
     */
    bool _wall_func;
};

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_TURBULENCEBOUNDARYEDDYVISCOSITY_HPP