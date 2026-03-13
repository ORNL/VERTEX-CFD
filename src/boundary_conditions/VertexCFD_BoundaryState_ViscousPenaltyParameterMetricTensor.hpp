#ifndef VERTEXCFD_BOUNDARYSTATE_VISCOUSPENALTYPARAMETERMETRICTENSOR_HPP
#define VERTEXCFD_BOUNDARYSTATE_VISCOUSPENALTYPARAMETERMETRICTENSOR_HPP

/**
 * @file
 */

#include <Panzer_Dimension.hpp>
#include <Panzer_Evaluator_WithBaseImpl.hpp>
#include <Panzer_IntegrationRule.hpp>
#include <Panzer_PureBasis.hpp>

#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_KokkosDeviceTypes.hpp>
#include <Phalanx_MDField.hpp>

#include <Kokkos_Core.hpp>

#include <string>

namespace VertexCFD
{
namespace BoundaryCondition
{
//---------------------------------------------------------------------------//
/**
 * @class ViscousPenaltyParameterMetricTensor
 * @brief Evaluate the penalty parameter \f$h_b\f$ from the tensor metric
 * formulation \f$\mathbf{M}\f$ and the normal vector \f$\mathbf{n}\f$:
 * \f$h_b = \sqrt{ \mathbf{n}^T \mathbf{M} \mathbf{n}}\f$
 *
 * @tparam EvalType  Automatic‑differentiation type used by the evaluator.
 * @tparam Traits    Traits class providing the evaluation data type.
 */
template<class EvalType, class Traits>
class ViscousPenaltyParameterMetricTensor
    : public panzer::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalType, Traits>
{
  public:
    using scalar_type = typename EvalType::ScalarT;

    /**
     * @brief Constructor.
     *
     * Registers dependent and evaluated fields with the evaluator framework.
     *
     * @param[in] ir Integration rule providing the data layout for fields.
     */
    ViscousPenaltyParameterMetricTensor(const panzer::IntegrationRule& ir);

    /**
     * @brief Evaluate the source terms over the workset.
     *
     * This method launches a Kokkos parallel kernel that computes the
     * production, destruction and total source contributions for each cell
     * and quadrature point.
     *
     * @param[in] workset Evaluation data containing cell information.
     */
    void evaluateFields(typename Traits::EvalData workset) override;

    /**
     * @brief Kokkos functor executed by a team policy.
     *
     * @param team Team member used for hierarchical parallelism.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const Kokkos::TeamPolicy<PHX::exec_space>::member_type& team) const;

    /**
     * @brief Field storing the penalty parameter
     */
    PHX::MDField<scalar_type, panzer::Cell, panzer::Point> _penalty_param;

  private:
    /** @brief Mesh dimension. */
    int _num_space_dim;

    /**
     * @brief Normal vector
     */
    PHX::MDField<const scalar_type, panzer::Cell, panzer::Point, panzer::Dim>
        _normals;
    /**
     * @brief Metric tensor
     */
    PHX::MDField<const double, panzer::Cell, panzer::Point, panzer::Dim, panzer::Dim>
        _metric_tensor;
};

//---------------------------------------------------------------------------//

} // end namespace BoundaryCondition
} // end namespace VertexCFD

#endif // VERTEXCFD_BOUNDARYSTATE_VISCOUSPENALTYPARAMETERMETRICTENSOR_HPP
